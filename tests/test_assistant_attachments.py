"""Local text extraction for note attachments. Fixtures are built in-test."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from echolingo.assistant import attachments
from echolingo.assistant.attachments import (
    NO_TEXT_WARNING,
    extract_attachments,
    extract_file,
    parse_attachment_items,
)

A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
NOTES_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide"


def make_pdf(pages: list[list[str]]) -> bytes:
    """A minimal valid PDF: one Helvetica text object per page (empty = no text)."""

    def escape(text: str) -> str:
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    count = len(pages)
    page_ids = [4 + 2 * index for index in range(count)]
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        (
            "<< /Type /Pages /Kids ["
            + " ".join(f"{page_id} 0 R" for page_id in page_ids)
            + f"] /Count {count} >>"
        ).encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    ]
    for index, lines in enumerate(pages):
        content_id = page_ids[index] + 1
        objects.append(
            (
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
            ).encode()
        )
        if lines:
            ops = ["BT", "/F1 20 Tf", "72 700 Td"]
            for number, line in enumerate(lines):
                if number:
                    ops.append("0 -28 Td")
                ops.append(f"({escape(line)}) Tj")
            ops.append("ET")
            stream = "\n".join(ops).encode("latin-1")
        else:
            stream = b""
        objects.append(
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"
        )
    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    ).encode()
    return bytes(out)


def slide_xml(*paragraphs: str) -> str:
    body = "".join(
        f"<a:p><a:r><a:t>{text}</a:t></a:r></a:p>" for text in paragraphs
    )
    return (
        f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<p:sld xmlns:a="{A_NS}" xmlns:p="{P_NS}"><p:cSld><p:spTree><p:sp><p:txBody>'
        f"{body}</p:txBody></p:sp></p:spTree></p:cSld></p:sld>"
    )


def notes_xml(*paragraphs: str) -> str:
    body = "".join(f"<a:p><a:r><a:t>{text}</a:t></a:r></a:p>" for text in paragraphs)
    return (
        f'<p:notes xmlns:a="{A_NS}" xmlns:p="{P_NS}"><p:cSld><p:spTree><p:sp><p:txBody>'
        f"{body}</p:txBody></p:sp></p:spTree></p:cSld></p:notes>"
    )


def make_pptx(path: Path, slides: dict[int, list[str]], notes: dict[int, list[str]] | None = None) -> Path:
    notes = notes or {}
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("ppt/presentation.xml", "<p:presentation/>")
        for number, paragraphs in slides.items():
            archive.writestr(f"ppt/slides/slide{number}.xml", slide_xml(*paragraphs))
            if number in notes:
                archive.writestr(
                    f"ppt/slides/_rels/slide{number}.xml.rels",
                    f'<Relationships xmlns="{REL_NS}"><Relationship Id="rId2" '
                    f'Type="{NOTES_REL}" Target="../notesSlides/notesSlide{number}.xml"/>'
                    "</Relationships>",
                )
                archive.writestr(
                    f"ppt/notesSlides/notesSlide{number}.xml",
                    notes_xml(*notes[number], str(number)),
                )
    return path


def make_docx(path: Path, body: str) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr(
            "word/document.xml",
            f'<w:document xmlns:w="{W_NS}"><w:body>{body}</w:body></w:document>',
        )
    return path


def paragraph(text: str) -> str:
    return f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>"


# ---------------------------------------------------------------------- PDF


def test_pdf_text_is_extracted_per_page_with_markers(tmp_path) -> None:
    pdf = tmp_path / "lecture.pdf"
    pdf.write_bytes(make_pdf([["Texture Perception", "Bela Julesz textons"], ["Feature Integration Theory"]]))
    result = extract_file(pdf, attachment_id="a1")
    assert result.pages == 2
    assert result.warning is None
    assert "--- page 1 ---" in result.text and "--- page 2 ---" in result.text
    assert "Julesz" in result.text and "Feature Integration Theory" in result.text
    assert result.segments[0].title == "Texture Perception"
    report = result.report()
    assert report == {
        "id": "a1",
        "name": "lecture.pdf",
        "pages": 2,
        "chars": result.chars,
        "truncated": False,
        "warning": None,
    }
    assert report["chars"] > 0


def test_pdf_without_a_text_layer_warns(tmp_path) -> None:
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(make_pdf([[], []]))
    result = extract_file(pdf)
    assert result.pages == 2
    assert result.chars == 0
    assert result.warning == NO_TEXT_WARNING
    assert "no extractable text" in result.warning.lower()


def test_pdf_with_mostly_empty_pages_warns_about_scans(tmp_path) -> None:
    pdf = tmp_path / "mixed.pdf"
    pdf.write_bytes(make_pdf([["Title page"], [], []]))
    result = extract_file(pdf)
    assert result.chars > 0
    assert "2 of 3 pages" in result.warning


def test_damaged_pdf_reports_a_warning(tmp_path) -> None:
    pdf = tmp_path / "broken.pdf"
    pdf.write_bytes(b"%PDF-1.4\nnot really a pdf")
    result = extract_file(pdf)
    assert result.segments == []
    assert result.warning


# ---------------------------------------------------------------------- PPTX


def test_pptx_orders_slides_numerically_and_includes_notes(tmp_path) -> None:
    deck = make_pptx(
        tmp_path / "deck.pptx",
        {
            10: ["Summary of pop-out"],
            2: ["Pre-attentive vision", "Parallel search"],
            1: ["Texture Perception", "CS180 Lecture 5"],
        },
        notes={2: ["Mention Treisman and Gelade 1980"]},
    )
    result = extract_file(deck, name="deck.pptx")
    assert result.pages == 3
    assert [segment.number for segment in result.segments] == [1, 2, 10]
    text = result.text
    assert text.index("--- slide 1 ---") < text.index("--- slide 2 ---") < text.index("--- slide 10 ---")
    assert "Notes: Mention Treisman and Gelade 1980" in text
    # The slide-number placeholder on the notes page is dropped.
    assert "\n2\n" not in text
    assert result.segments[1].title == "Pre-attentive vision"


def test_pptx_zip_bomb_guards(tmp_path, monkeypatch) -> None:
    many = tmp_path / "many.pptx"
    with zipfile.ZipFile(many, "w") as archive:
        for index in range(attachments.MAX_ARCHIVE_ENTRIES + 1):
            archive.writestr(f"ppt/slides/slide{index + 1}.xml", "")
    result = extract_file(many)
    assert result.segments == []
    assert "too many parts" in result.warning

    monkeypatch.setattr(attachments, "MAX_ARCHIVE_UNCOMPRESSED", 100_000)
    large = tmp_path / "large.pptx"
    with zipfile.ZipFile(large, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("ppt/slides/slide1.xml", slide_xml("x" * 200_000))
    result = extract_file(large)
    assert result.segments == []
    assert "safety limit" in result.warning


def test_xml_with_doctype_is_refused(tmp_path) -> None:
    deck = tmp_path / "entity.pptx"
    with zipfile.ZipFile(deck, "w") as archive:
        archive.writestr(
            "ppt/slides/slide1.xml",
            '<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY lol "lol">]>' + slide_xml("&lol;"),
        )
    result = extract_file(deck)
    assert result.segments == []
    assert "unsupported XML" in result.warning


def test_doctype_after_a_long_comment_is_refused(tmp_path) -> None:
    deck = tmp_path / "hidden-entity.pptx"
    with zipfile.ZipFile(deck, "w") as archive:
        archive.writestr(
            "ppt/slides/slide1.xml",
            '<?xml version="1.0"?><!--' + "x" * 10_000 + '--><!DOCTYPE lolz [<!ENTITY lol "lol">]>'
            + slide_xml("&lol;"),
        )
    result = extract_file(deck)
    assert result.segments == []
    assert "unsupported XML" in result.warning


def test_not_a_zip_is_reported(tmp_path) -> None:
    fake = tmp_path / "fake.docx"
    fake.write_bytes(b"plain text pretending to be a docx")
    result = extract_file(fake)
    assert result.warning == "The file is not a valid Office document."


# ---------------------------------------------------------------------- DOCX


def test_docx_paragraphs_and_table_cells(tmp_path) -> None:
    table = (
        "<w:tbl>"
        "<w:tr><w:tc>" + paragraph("Term") + "</w:tc><w:tc>" + paragraph("Meaning") + "</w:tc></w:tr>"
        "<w:tr><w:tc>" + paragraph("saccade") + "</w:tc><w:tc>" + paragraph("rapid eye movement") + "</w:tc></w:tr>"
        "</w:tbl>"
    )
    document = make_docx(
        tmp_path / "handout.docx",
        paragraph("Handout: Visual Search") + table + paragraph("Read chapter 3."),
    )
    result = extract_file(document)
    assert result.warning is None
    lines = result.text.splitlines()
    assert lines[0] == "Handout: Visual Search"
    assert "saccade | rapid eye movement" in lines
    assert lines[-1] == "Read chapter 3."


# ---------------------------------------------------------------------- text


def test_text_files_decode_utf8_and_legacy_chinese(tmp_path) -> None:
    utf8 = tmp_path / "notes.md"
    utf8.write_text("# 视觉感知\n\n预注意视觉 = pre-attentive vision\n", encoding="utf-8")
    assert "预注意视觉" in extract_file(utf8).text

    legacy = tmp_path / "legacy.txt"
    legacy.write_bytes("纹理感知与视觉搜索，特征整合理论。".encode("gb18030"))
    assert "特征整合理论" in extract_file(legacy).text

    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    assert extract_file(empty).warning == "The file is empty."


def test_char_budget_truncates_and_flags(tmp_path) -> None:
    long_text = tmp_path / "long.txt"
    long_text.write_text("word " * 5000, encoding="utf-8")
    result = extract_file(long_text, char_budget=1000)
    assert result.chars == 1000
    assert result.truncated is True
    assert "1,000" in result.warning


def test_input_size_limit_and_unsupported_types(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(attachments, "MAX_INPUT_BYTES", 100)
    big = tmp_path / "big.txt"
    big.write_text("x" * 101, encoding="utf-8")
    assert extract_file(big).warning == "The file is larger than 25 MB."

    exe = tmp_path / "tool.exe"
    exe.write_bytes(b"MZ")
    assert "Unsupported file type" in extract_file(exe).warning

    missing = extract_file(tmp_path / "missing.pdf")
    assert missing.warning == "The file could not be opened."


def test_extension_falls_back_to_the_display_name(tmp_path) -> None:
    staged = tmp_path / "attachment-7f3a"
    staged.write_text("Julesz textons", encoding="utf-8")
    result = extract_file(staged, name="slides.txt")
    assert result.kind == "txt"
    assert "Julesz" in result.text


async def test_extract_attachments_keeps_order_and_ids(tmp_path) -> None:
    first = tmp_path / "a.txt"
    first.write_text("first file", encoding="utf-8")
    second = tmp_path / "b.pdf"
    second.write_bytes(make_pdf([["second file"]]))
    items = parse_attachment_items(
        [{"id": "x1", "path": str(first), "name": "A"}, {"id": "x2", "path": str(second)}]
    )
    results = await extract_attachments(items)
    assert [item.id for item in results] == ["x1", "x2"]
    assert [item.name for item in results] == ["A", "b.pdf"]
    assert "second file" in results[1].text


@pytest.mark.parametrize(
    "value",
    [
        "not a list",
        [{"id": "1"}],
        [{"path": ""}],
        ["path"],
        [{"path": f"/tmp/{index}.txt"} for index in range(6)],
    ],
)
def test_parse_attachment_items_rejects_malformed_values(value) -> None:
    with pytest.raises(ValueError):
        parse_attachment_items(value)


def test_parse_attachment_items_accepts_missing_list() -> None:
    assert parse_attachment_items(None) == []
