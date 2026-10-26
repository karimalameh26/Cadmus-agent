from __future__ import annotations

import io
import os
from pathlib import Path

import pytest
from PIL import Image

from cadmus.config import AppConfig
from cadmus.inputs import DirectFileAdapter
from cadmus.storage import ArtifactStore
from tests.conftest import base_config


def make_adapter(tmp_path: Path, root: Path, **documents) -> DirectFileAdapter:
    raw = base_config(tmp_path, input={"mode": "direct_files", "roots": [str(root)]}, documents=documents)
    cfg = AppConfig.model_validate(raw)
    return DirectFileAdapter(cfg, ArtifactStore(tmp_path / "state", cfg.client_id))


def pdf_bytes(pages: list[str], image_page: bool = False) -> bytes:
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    for text in pages:
        c.drawString(72, 720, text)
        c.showPage()
    if image_page:
        img = Image.new("RGB", (200, 100), "white")
        c.drawImage(ImageReader(img), 72, 500, 200, 100)
        c.showPage()
    c.save()
    return buf.getvalue()


def test_text_image_and_pdf_extraction(tmp_path: Path) -> None:
    root = tmp_path / "in"
    root.mkdir()
    (root / "a.txt").write_text("Invoice remittance details", encoding="utf-8")
    img = Image.new("RGB", (3000, 1000), "white")
    exif = img.getexif()
    exif[0x0112] = 6
    img.save(root / "b.jpg", exif=exif)
    (root / "c.pdf").write_bytes(pdf_bytes(["Agreement between the parties hereby signed today"], image_page=True))
    (root / "d.docx").write_bytes(b"x")
    items = {i.name: i for i in make_adapter(tmp_path, root).load([root])}
    assert items["a.txt"].evidence.text[0].text.startswith("Invoice")
    b = items["b.jpg"].evidence
    assert b.media and max(b.media[0].width, b.media[0].height) == 1536
    assert any(t.startswith("exif_transpose") for t in b.provenance["transforms"])
    assert b.media[0].height > b.media[0].width  # orientation 6 rotates landscape to portrait
    c = items["c.pdf"].evidence
    assert c.text[0].page == 1 and "Agreement" in c.text[0].text
    assert c.provenance["rendered_pages"] == [2] and c.media[0].page == 2
    assert items["d.docx"].error_code == "unsupported_format"


def test_pdf_page_limit_reports_omissions(tmp_path: Path) -> None:
    root = tmp_path / "in"
    root.mkdir()
    (root / "long.pdf").write_bytes(pdf_bytes([f"Page number {i} with enough text here" for i in range(1, 5)]))
    [item] = make_adapter(tmp_path, root, max_pages=2).load([root])
    assert item.evidence.provenance["pages_included"] == [1, 2]
    assert item.evidence.provenance["pages_omitted"] == [3, 4] and item.evidence.provenance["truncated"]
    [item] = make_adapter(tmp_path, root, max_pages=2, oversize_policy="reject").load([root])
    assert item.error_code == "too_many_pages"


def test_corrupt_encrypted_and_undecodable_inputs(tmp_path: Path) -> None:
    from pypdf import PdfReader, PdfWriter

    root = tmp_path / "in"
    root.mkdir()
    (root / "bad.pdf").write_bytes(b"%PDF-1.4 garbage")
    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(pdf_bytes(["Secret page text that is long enough"]))))
    writer.encrypt("pw")
    with (root / "enc.pdf").open("wb") as fh:
        writer.write(fh)
    (root / "bad.png").write_bytes(b"\x89PNG broken")
    (root / "latin.txt").write_bytes("caf\xe9".encode("latin-1"))
    codes = {i.name: i.error_code for i in make_adapter(tmp_path, root).load([root])}
    assert codes == {"bad.pdf": "pdf_corrupt", "bad.png": "image_corrupt", "enc.pdf": "pdf_encrypted",
                     "latin.txt": "text_decode_error"}


def test_paths_outside_roots_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "in"
    root.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("secret")
    [item] = make_adapter(tmp_path, root).load([outside])
    assert item.error_code == "outside_input_roots"
    link = root / "link.txt"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted on this system")
    [item] = make_adapter(tmp_path, root).load([root])
    assert item.error_code == "outside_input_roots"
