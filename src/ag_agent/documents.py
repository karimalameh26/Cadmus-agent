"""Direct-file extraction: UTF-8 text, images and PDFs into normalized evidence with provenance."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from ag_agent.config import DocumentConfig
from ag_agent.contracts import MediaRef, TextBlock
from ag_agent.errors import ExtractionError
from ag_agent.storage import ArtifactStore

TEXT_EXTENSIONS = {".txt", ".md"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | IMAGE_EXTENSIONS | PDF_EXTENSIONS


class Extracted:
    def __init__(self) -> None:
        self.text: list[TextBlock] = []
        self.media: list[MediaRef] = []
        self.provenance: dict[str, Any] = {"transforms": [], "truncated": False}


def _require(module: str) -> Any:
    try:
        return __import__(module)
    except ImportError:
        raise ExtractionError(f"Optional dependency '{module}' is required; install 'ag-agent[files]'",
                              "dependency_missing") from None


def _limit_text(out: Extracted, blocks: list[TextBlock], cfg: DocumentConfig) -> None:
    remaining = cfg.max_text_chars
    for block in blocks:
        if remaining <= 0:
            if cfg.oversize_policy == "reject":
                raise ExtractionError("Document text exceeds max_text_chars", "text_too_large")
            out.provenance["truncated"] = True
            continue
        if len(block.text) > remaining:
            if cfg.oversize_policy == "reject":
                raise ExtractionError("Document text exceeds max_text_chars", "text_too_large")
            block = TextBlock(source=block.source, page=block.page, text=block.text[:remaining])
            out.provenance["truncated"] = True
        remaining -= len(block.text)
        out.text.append(block)


def normalize_image(image: Any, cfg: DocumentConfig, artifacts: ArtifactStore, page: int | None,
                    transforms: list[str]) -> MediaRef:
    _require("PIL")
    from PIL import Image, ImageOps

    w, h = image.size
    if w * h > cfg.max_image_pixels:
        raise ExtractionError("Image exceeds max_image_pixels", "image_too_large")
    orientation = image.getexif().get(0x0112, 1)
    if orientation != 1:
        image = ImageOps.exif_transpose(image)
        transforms.append(f"exif_transpose:{orientation}")
    if image.mode != "RGB":
        image = image.convert("RGB")
        transforms.append("convert_rgb")
    if max(image.size) > cfg.image_max_side:
        image = image.copy()
        image.thumbnail((cfg.image_max_side, cfg.image_max_side), Image.Resampling.LANCZOS)
        transforms.append(f"resize:{image.size[0]}x{image.size[1]}")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=cfg.jpeg_quality, optimize=False)
    digest = artifacts.put(buf.getvalue())
    return MediaRef(sha256=digest, mime_type="image/jpeg", width=image.size[0], height=image.size[1], page=page)


def _extract_text_file(data: bytes, cfg: DocumentConfig, out: Extracted) -> None:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ExtractionError("Text file is not valid UTF-8", "text_decode_error") from None
    _limit_text(out, [TextBlock(source="file_text", text=text)], cfg)


def _extract_image(data: bytes, cfg: DocumentConfig, artifacts: ArtifactStore, out: Extracted) -> None:
    _require("PIL")
    from PIL import Image, UnidentifiedImageError

    try:
        image = Image.open(io.BytesIO(data))
        frames = getattr(image, "n_frames", 1)
        image.load()
    except Image.DecompressionBombError:
        raise ExtractionError("Image exceeds safe pixel limits", "image_too_large") from None
    except (UnidentifiedImageError, OSError, SyntaxError):
        raise ExtractionError("Image could not be decoded", "image_corrupt") from None
    out.provenance["original_size"] = list(image.size)
    out.provenance["pages_total"] = frames
    out.provenance["pages_included"] = [1]
    out.provenance["pages_omitted"] = list(range(2, frames + 1))
    out.media.append(normalize_image(image, cfg, artifacts, None, out.provenance["transforms"]))


def _extract_pdf(data: bytes, cfg: DocumentConfig, artifacts: ArtifactStore, out: Extracted) -> None:
    _require("pypdf")
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ExtractionError("PDF is encrypted", "pdf_encrypted")
        total = len(reader.pages)
    except ExtractionError:
        raise
    except (PdfReadError, ValueError, OSError, KeyError):
        raise ExtractionError("PDF could not be parsed", "pdf_corrupt") from None
    if total == 0:
        raise ExtractionError("PDF has no pages", "pdf_empty")
    if total > cfg.max_pages and cfg.oversize_policy == "reject":
        raise ExtractionError("PDF exceeds max_pages", "too_many_pages")
    selected = list(range(min(total, cfg.max_pages)))
    out.provenance.update(pages_total=total, pages_included=[i + 1 for i in selected],
                          pages_omitted=list(range(len(selected) + 1, total + 1)),
                          page_policy=f"first_{cfg.max_pages}")
    if total > len(selected):
        out.provenance["truncated"] = True
    blocks: list[TextBlock] = []
    scanned: list[int] = []
    for i in selected:
        try:
            text = reader.pages[i].extract_text() or ""
        except Exception:  # pypdf raises many types on malformed content streams
            text = ""
        if len(text.strip()) >= cfg.pdf_min_text_chars:
            blocks.append(TextBlock(source="pdf_text", page=i + 1, text=text.strip()))
        else:
            scanned.append(i)
    _limit_text(out, blocks, cfg)
    if scanned:
        if cfg.pdf_scan_strategy == "fail":
            raise ExtractionError("PDF has pages without extractable text", "pdf_scanned_unsupported")
        try:
            import pypdfium2 as pdfium
        except ImportError:
            raise ExtractionError("Rendering scanned PDF pages requires pypdfium2 ('ag-agent[files]')",
                                  "pdf_render_unavailable") from None
        out.provenance["rendered_pages"] = [i + 1 for i in scanned]
        try:
            doc = pdfium.PdfDocument(data)
            for i in scanned:
                bitmap = doc[i].render(scale=cfg.pdf_render_scale)
                out.media.append(normalize_image(bitmap.to_pil(), cfg, artifacts, i + 1,
                                                 out.provenance["transforms"]))
            doc.close()
        except ExtractionError:
            raise
        except Exception:
            raise ExtractionError("PDF page rendering failed", "pdf_render_failed") from None
        out.provenance["transforms"].append(f"pdf_render_scale:{cfg.pdf_render_scale}")


def extract_file(path: Path, cfg: DocumentConfig, artifacts: ArtifactStore) -> tuple[Extracted, bytes]:
    try:
        size = path.stat().st_size
    except OSError:
        raise ExtractionError("File is not readable", "file_unreadable") from None
    if size > cfg.max_file_bytes:
        raise ExtractionError("File exceeds max_file_bytes", "file_too_large")
    data = path.read_bytes()
    out = Extracted()
    suffix = path.suffix.lower()
    if suffix in TEXT_EXTENSIONS:
        _extract_text_file(data, cfg, out)
    elif suffix in IMAGE_EXTENSIONS:
        _extract_image(data, cfg, artifacts, out)
    elif suffix in PDF_EXTENSIONS:
        _extract_pdf(data, cfg, artifacts, out)
    else:
        raise ExtractionError("Unsupported file type", "unsupported_format")
    out.provenance["extractor"] = suffix.lstrip(".")
    return out, data
