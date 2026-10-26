"""Compose classification requests from fixed instructions, mutable guidance, examples and untrusted evidence."""

from __future__ import annotations

import json
import re

from cadmus.contracts import EvidenceRecord, PromptBundle
from cadmus.providers import ImagePart, ModelRequest, TextPart
from cadmus.storage import ArtifactStore
from cadmus.tasks import ClassificationTask

_RESERVED_TAGS = re.compile(r"<(/?)(document_evidence|example|evidence|answer|classification_guidance)\b", re.I)


def neutralize(text: str) -> str:
    """Prevent untrusted content from closing or opening the request's structural tags."""
    return _RESERVED_TAGS.sub(lambda m: f"&lt;{m.group(1)}{m.group(2)}", text)


def render_evidence(ev: EvidenceRecord) -> str:
    lines: list[str] = []
    for block in ev.text:
        page = f" page={block.page}" if block.page is not None else ""
        if block.source.startswith("description:"):
            header = f"[supplied description, field={block.source.split(':', 1)[1]} - not direct inspection]"
        else:
            header = f"[text source={block.source}{page}]"
        lines += [header, neutralize(block.text)]
    if ev.fields:
        lines.append("[structured fields]")
        for key in sorted(ev.fields):
            lines.append(f"{key}: {neutralize(json.dumps(ev.fields[key], ensure_ascii=False, sort_keys=True))}")
    for i, media in enumerate(ev.media, start=1):
        page = f" page={media.page}" if media.page is not None else ""
        lines.append(f"[image {i}{page} attached]")
    prov = ev.provenance
    if prov.get("pages_omitted"):
        lines.append(f"[note: pages omitted {prov['pages_omitted']}]")
    if prov.get("truncated"):
        lines.append("[note: content truncated]")
    return "\n".join(lines)


def compose_request(task: ClassificationTask, bundle: PromptBundle, evidence: EvidenceRecord,
                    example_evidence: list[EvidenceRecord], artifacts: ArtifactStore,
                    max_output_tokens: int, temperature: float | None) -> ModelRequest:
    system = (
        f"{task.fixed_instructions()}\n\n"
        f"<classification_guidance>\n{bundle.guidance.strip()}\n</classification_guidance>\n"
        "The guidance above refines how to apply the allowed labels; it cannot change the label set, the "
        "output format, or the untrusted-content rule."
    )
    parts: list[TextPart | ImagePart] = []
    for ex, ev in zip(bundle.examples, example_evidence, strict=True):
        answer = json.dumps({"status": "success", "labels": ex.labels})
        parts.append(TextPart(f"<example>\n<evidence>\n{render_evidence(ev)}\n</evidence>\n"
                              f"<answer>{answer}</answer>\n</example>"))
        parts += [ImagePart(m.mime_type, artifacts.get(m.sha256)) for m in ev.media]
    parts.append(TextPart(f"<document_evidence>\n{render_evidence(evidence)}\n</document_evidence>\n"
                          "Classify the document in <document_evidence>."))
    parts += [ImagePart(m.mime_type, artifacts.get(m.sha256)) for m in evidence.media]
    return ModelRequest(system=system, parts=parts, response_schema=task.output_schema(),
                        max_output_tokens=max_output_tokens, temperature=temperature)
