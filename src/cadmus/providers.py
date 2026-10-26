"""Model adapters: a deterministic fake for credential-free development and an OpenAI-compatible HTTP adapter.

Retries are owned by the inference executor so every attempt is ledgered and budgeted; adapters never retry.
"""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from cadmus.config import ProviderConfig
from cadmus.contracts import canonical_json, sha256_text
from cadmus.errors import ConfigError, ProviderError


@dataclass
class ImagePart:
    mime_type: str
    data: bytes


@dataclass
class TextPart:
    text: str


@dataclass
class ModelRequest:
    system: str
    parts: list[TextPart | ImagePart]
    response_schema: dict[str, Any] | None = None
    schema_name: str = "classification"
    max_output_tokens: int = 400
    temperature: float | None = 0.0

    def text_chars(self) -> int:
        return len(self.system) + sum(len(p.text) for p in self.parts if isinstance(p, TextPart))

    def image_count(self) -> int:
        return sum(isinstance(p, ImagePart) for p in self.parts)


@dataclass
class ModelResponse:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    provider_request_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class ModelAdapter(Protocol):
    supports_images: bool

    def identity(self) -> dict[str, Any]: ...

    def complete(self, request: ModelRequest) -> ModelResponse: ...


def model_hash(adapter: ModelAdapter) -> str:
    return sha256_text(canonical_json(adapter.identity()))[:12]


_RULE = re.compile(r"""if the evidence mentions\s+["“'](.+?)["”'],?\s+classify as\s+["“'](.+?)["”']""", re.I)
_TAG = re.compile(r"<(classification_guidance|document_evidence)>(.*?)</\1>", re.S)
_EXAMPLE = re.compile(r"<example>\s*<evidence>(.*?)</evidence>\s*<answer>(.*?)</answer>\s*</example>", re.S)
_WORD = re.compile(r"[a-z][a-z0-9]{2,}")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


class FakeModel:
    """Deterministic rule-following stand-in for a classifier model. Text-only: it never inspects pixels.

    It applies guidance lines of the form: If the evidence mentions "<keyword>", classify as "<label>".
    Without a matching rule it copies the labels of a near-identical few-shot example, otherwise abstains.
    `fail_on` lets tests inject provider failures for specific evidence keywords.
    """

    supports_images = True

    def __init__(self, multilabel: bool = False, fail_on: dict[str, ProviderError] | None = None,
                 hook: Callable[[ModelRequest], None] | None = None) -> None:
        self.multilabel = multilabel
        self.fail_on = fail_on or {}
        self.hook = hook
        self.calls = 0

    def identity(self) -> dict[str, Any]:
        return {"kind": "fake", "model": "fake-classifier-v1", "multilabel": self.multilabel}

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        if self.hook:
            self.hook(request)
        user = "\n".join(p.text for p in request.parts if isinstance(p, TextPart))
        guidance_blocks = [m.group(2) for m in _TAG.finditer(request.system) if m.group(1) == "classification_guidance"]
        evidence_blocks = [m.group(2) for m in _TAG.finditer(user) if m.group(1) == "document_evidence"]
        evidence = evidence_blocks[-1].lower() if evidence_blocks else ""
        for keyword, err in self.fail_on.items():
            if keyword in evidence:
                raise err
        rules = _RULE.findall(guidance_blocks[-1] if guidance_blocks else "")
        labels: list[str] = []
        for keyword, label in rules:
            if re.search(r"\b" + re.escape(keyword.lower()) + r"\b", evidence):
                if label not in labels:
                    labels.append(label)
                if not self.multilabel:
                    break
        if not labels:
            ev_tokens = _tokens(evidence)
            best, best_score = None, 0.0
            for ex_ev, ex_answer in _EXAMPLE.findall(user):
                ex_tokens = _tokens(ex_ev)
                union = ev_tokens | ex_tokens
                score = len(ev_tokens & ex_tokens) / len(union) if union else 0.0
                if score > best_score:
                    best, best_score = ex_answer, score
            if best is not None and best_score >= 0.6:
                labels = json.loads(best).get("labels", [])
        if labels:
            out = {"status": "success", "labels": labels, "evidence": [], "explanation": "rule match",
                   "confidence": 0.9}
        else:
            out = {"status": "abstained", "labels": [], "evidence": [], "explanation": "no rule matched",
                   "confidence": 0.0}
        text = json.dumps(out)
        in_tokens = request.text_chars() // 4 + 85 * request.image_count()
        return ModelResponse(text=text, input_tokens=in_tokens, output_tokens=len(text) // 4,
                             provider_request_id=f"fake-{self.calls}")


class OpenAICompatibleModel:
    """Chat Completions-compatible endpoint (OpenAI, Azure OpenAI v1-style endpoints, many local servers)."""

    def __init__(self, cfg: ProviderConfig) -> None:
        if cfg.kind != "openai_compatible" or not cfg.base_url:
            raise ConfigError("OpenAICompatibleModel requires provider.kind = 'openai_compatible'")
        self.cfg = cfg
        self.supports_images = cfg.supports_images
        self._url = cfg.base_url.rstrip("/") + "/chat/completions"

    def identity(self) -> dict[str, Any]:
        return self.cfg.identity()

    def _headers(self) -> dict[str, str]:
        key = os.environ.get(self.cfg.api_key_env)
        if not key:
            raise ProviderError(f"Credential environment variable {self.cfg.api_key_env} is not set",
                                "credentials_missing")
        auth = {"Authorization": f"Bearer {key}"} if self.cfg.auth_header == "bearer" else {"api-key": key}
        return {"Content-Type": "application/json", **auth}

    def _body(self, request: ModelRequest) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        for part in request.parts:
            if isinstance(part, TextPart):
                content.append({"type": "text", "text": part.text})
            else:
                b64 = base64.b64encode(part.data).decode("ascii")
                content.append({"type": "image_url", "image_url": {
                    "url": f"data:{part.mime_type};base64,{b64}", "detail": self.cfg.image_detail}})
        body: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": [{"role": "system", "content": request.system}, {"role": "user", "content": content}],
            self.cfg.max_tokens_param: request.max_output_tokens,
        }
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.response_schema is not None and self.cfg.structured_output == "json_schema":
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": request.schema_name, "schema": request.response_schema, "strict": True}}
        elif self.cfg.structured_output in ("json_object", "json_schema"):
            body["response_format"] = {"type": "json_object"}
        return body

    def complete(self, request: ModelRequest) -> ModelResponse:
        if request.image_count() and not self.supports_images:
            raise ProviderError("Configured model does not accept images", "capability_mismatch")
        data = json.dumps(self._body(request)).encode("utf-8")
        req = urllib.request.Request(self._url, data=data, headers=self._headers(), method="POST")  # noqa: S310
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:  # noqa: S310
                payload = json.loads(resp.read().decode("utf-8"))
                request_id = resp.headers.get("x-request-id") or resp.headers.get("apim-request-id")
        except urllib.error.HTTPError as exc:
            request_id = exc.headers.get("x-request-id") if exc.headers else None
            transient = exc.code in (408, 409, 429, 500, 502, 503, 504)
            code = "rate_limited" if exc.code == 429 else f"http_{exc.code}"
            raise ProviderError(f"Provider returned HTTP {exc.code}", code, transient=transient,
                                request_id=request_id) from None
        except TimeoutError:
            raise ProviderError("Provider request timed out", "timeout", transient=True,
                                completion_unknown=True) from None
        except urllib.error.URLError:
            raise ProviderError("Provider connection failed", "connection_error", transient=True) from None
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise ProviderError("Provider returned a non-JSON body", "provider_bad_body") from None
        try:
            choice = payload["choices"][0]
            text = choice["message"].get("content") or ""
            if choice["message"].get("refusal"):
                raise ProviderError("Model refused the request", "model_refusal")
        except (KeyError, IndexError, TypeError):
            raise ProviderError("Provider response missing choices", "provider_bad_body") from None
        usage = payload.get("usage") or {}
        return ModelResponse(text=text, input_tokens=usage.get("prompt_tokens"),
                             output_tokens=usage.get("completion_tokens"),
                             provider_request_id=request_id or payload.get("id"))


def make_model(cfg: ProviderConfig, multilabel: bool = False) -> ModelAdapter:
    if cfg.kind == "fake":
        return FakeModel(multilabel=multilabel)
    return OpenAICompatibleModel(cfg)
