"""Local text extraction for note attachments (slides, handouts, notes).

Supported: PDF (pypdf), PPTX and DOCX (zipfile + ElementTree, no Office
dependency) and plain text (txt, md, markdown, tex, csv). Extraction runs in
a worker thread, never touches the network and enforces size limits:

* at most ``MAX_INPUT_BYTES`` per file,
* for OOXML archives at most ``MAX_ARCHIVE_ENTRIES`` entries and
  ``MAX_ARCHIVE_UNCOMPRESSED`` declared uncompressed bytes (zip-bomb guard),
* at most ``char_budget`` extracted characters per attachment (``truncated``).

A file that cannot be read yields a report with a ``warning`` instead of
failing the whole notes request.
"""

from __future__ import annotations

import asyncio
import io
import logging
import re
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

logger = logging.getLogger("echolingo.assistant.attachments")

SUPPORTED_EXTENSIONS = frozenset({"pdf", "pptx", "docx", "txt", "md", "markdown", "tex", "csv"})
TEXT_EXTENSIONS = frozenset({"txt", "md", "markdown", "tex", "csv"})
MAX_ATTACHMENTS = 5
MAX_INPUT_BYTES = 25 * 1024 * 1024
MAX_ARCHIVE_UNCOMPRESSED = 50 * 1024 * 1024
MAX_ARCHIVE_ENTRIES = 2000
MAX_PDF_PAGES = 1000
CHAR_BUDGET = 60_000
TITLE_MAX_CHARS = 80

NO_TEXT_WARNING = "No extractable text: the PDF may contain scanned images only."

_A_NS = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_REL_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"
_NOTES_REL_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide"
_SLIDE_RE = re.compile(r"^ppt/slides/slide(\d+)\.xml$")
_NOTES_RE = re.compile(r"^ppt/notesSlides/notesSlide(\d+)\.xml$")
_BLANK_LINES = re.compile(r"\n{3,}")
_TRAILING_SPACE = re.compile(r"[ \t]+\n")


class AttachmentError(Exception):
    """A per-file problem reported as a warning (safe to show the user)."""


@dataclass(slots=True)
class AttachmentSegment:
    """One page or slide (or the whole document for flat formats)."""

    label: str  # "page 3", "slide 3" or "" for flat documents
    number: int
    text: str

    @property
    def title(self) -> str:
        return first_line_title(self.text)

    def render(self) -> str:
        if not self.label:
            return self.text
        return f"--- {self.label} ---\n{self.text}".rstrip()


@dataclass(slots=True)
class ExtractedAttachment:
    id: str
    name: str
    kind: str
    pages: int | None = None
    truncated: bool = False
    warning: str | None = None
    segments: list[AttachmentSegment] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(segment.render() for segment in self.segments if segment.text.strip())

    @property
    def chars(self) -> int:
        return sum(len(segment.text) for segment in self.segments)

    def report(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "pages": self.pages,
            "chars": self.chars,
            "truncated": self.truncated,
            "warning": self.warning,
        }


def first_line_title(text: str, limit: int = TITLE_MAX_CHARS) -> str:
    """First meaningful line of a page or slide, trimmed to ``limit`` chars."""
    for raw in text.splitlines():
        line = " ".join(raw.split()).strip(" -–—•·*#|")
        if len(line) < 2 or line.replace(".", "").isdigit():
            continue
        if line.lower().startswith("notes:"):
            continue
        if len(line) > limit:
            cut = line[:limit].rsplit(" ", 1)[0]
            line = (cut if len(cut) >= limit // 2 else line[:limit]).rstrip() + "…"
        return line
    return ""


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = _TRAILING_SPACE.sub("\n", text)
    return _BLANK_LINES.sub("\n\n", text).strip()


class _Budget:
    def __init__(self, limit: int) -> None:
        self.left = max(0, int(limit))
        self.truncated = False

    def take(self, text: str) -> str | None:
        """Return the part of ``text`` that fits; ``None`` once exhausted."""
        if self.left <= 0:
            if text.strip():
                self.truncated = True
            return None
        if len(text) > self.left:
            self.truncated = True
            text = text[: self.left]
        self.left -= len(text)
        return text


# ---------------------------------------------------------------------- PDF


def _extract_pdf(data: bytes, result: ExtractedAttachment, budget: _Budget) -> None:
    try:
        from pypdf import PdfReader
    except ImportError as error:  # pragma: no cover - packaging defect
        raise AttachmentError("PDF support is not installed in this build.") from error
    # pypdf reports structural oddities of real-world PDFs as warnings; they are
    # not actionable in logs/sidecar.log.
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
        if reader.is_encrypted:
            try:
                if not reader.decrypt(""):
                    raise AttachmentError("The PDF is password-protected.")
            except AttachmentError:
                raise
            except Exception as error:
                raise AttachmentError("The PDF is password-protected.") from error
        pages = reader.pages
        total = len(pages)
    except AttachmentError:
        raise
    except Exception as error:
        raise AttachmentError("The PDF could not be read (damaged or unsupported).") from error
    result.pages = total
    empty = 0
    failed = 0
    for index in range(min(total, MAX_PDF_PAGES)):
        if budget.left <= 0:
            budget.truncated = True
            break
        try:
            text = _clean(pages[index].extract_text() or "")
        except Exception:
            failed += 1
            text = ""
        if not text:
            empty += 1
            continue
        kept = budget.take(text)
        if kept is None:
            break
        result.segments.append(AttachmentSegment(f"page {index + 1}", index + 1, kept))
    if total > MAX_PDF_PAGES:
        budget.truncated = True
    if total and not result.segments:
        result.warning = NO_TEXT_WARNING
    elif total >= 2 and empty * 2 > total:
        result.warning = (
            f"{empty} of {total} pages have no extractable text (they may be scanned images)."
        )
    elif failed:
        result.warning = f"{failed} page(s) could not be read."


# --------------------------------------------------------------------- OOXML


def _open_archive(data: bytes) -> zipfile.ZipFile:
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except (zipfile.BadZipFile, OSError, ValueError) as error:
        raise AttachmentError("The file is not a valid Office document.") from error
    infos = archive.infolist()
    if len(infos) > MAX_ARCHIVE_ENTRIES:
        archive.close()
        raise AttachmentError("The document has too many parts to read safely.")
    if sum(info.file_size for info in infos) > MAX_ARCHIVE_UNCOMPRESSED:
        archive.close()
        raise AttachmentError("The document expands beyond the 50 MB safety limit.")
    return archive


def _read_member(archive: zipfile.ZipFile, name: str) -> bytes | None:
    try:
        info = archive.getinfo(name)
    except KeyError:
        return None
    if info.file_size > MAX_ARCHIVE_UNCOMPRESSED:
        raise AttachmentError("The document expands beyond the 50 MB safety limit.")
    try:
        with archive.open(info) as handle:
            # ZipExtFile stops at the declared size; the extra byte detects a lie.
            data = handle.read(info.file_size + 1)
    except (zipfile.BadZipFile, OSError, RuntimeError, ValueError, EOFError) as error:
        raise AttachmentError("A part of the document is damaged.") from error
    if len(data) > info.file_size:
        raise AttachmentError("A part of the document is damaged.")
    return data


def _parse_xml(data: bytes) -> ElementTree.Element:
    # The whole part is searched: comments or processing instructions may
    # push a DTD past any fixed-size prefix. (Markup keywords are uppercase;
    # literal "<" in text is always escaped.)
    if b"<!DOCTYPE" in data or b"<!ENTITY" in data:
        # OOXML never uses DTDs; refuse entity expansion outright.
        raise AttachmentError("The document contains unsupported XML declarations.")
    try:
        return ElementTree.fromstring(data)
    except ElementTree.ParseError as error:
        raise AttachmentError("A part of the document is damaged.") from error


def _drawing_paragraphs(root: ElementTree.Element) -> list[str]:
    lines: list[str] = []
    for paragraph in root.iter(f"{_A_NS}p"):
        parts: list[str] = []
        for node in paragraph.iter():
            if node.tag == f"{_A_NS}t" and node.text:
                parts.append(node.text)
            elif node.tag == f"{_A_NS}br":
                parts.append("\n")
        line = "".join(parts).strip()
        if line:
            lines.append(line)
    return lines


def _slide_notes_target(archive: zipfile.ZipFile, number: int) -> str | None:
    rels = _read_member(archive, f"ppt/slides/_rels/slide{number}.xml.rels")
    if rels is None:
        return None
    for rel in _parse_xml(rels).iter(f"{_REL_NS}Relationship"):
        if rel.get("Type") == _NOTES_REL_TYPE and rel.get("Target"):
            target = rel.get("Target", "")
            name = target.rsplit("/", 1)[-1]
            return f"ppt/notesSlides/{name}"
    return None


def _extract_pptx(data: bytes, result: ExtractedAttachment, budget: _Budget) -> None:
    with _open_archive(data) as archive:
        slides = sorted(
            (int(match.group(1)), info.filename)
            for info in archive.infolist()
            if (match := _SLIDE_RE.match(info.filename))
        )
        if not slides:
            raise AttachmentError("The presentation contains no slides.")
        result.pages = len(slides)
        notes_by_number = {
            int(match.group(1)): info.filename
            for info in archive.infolist()
            if (match := _NOTES_RE.match(info.filename))
        }
        for number, filename in slides:
            if budget.left <= 0:
                budget.truncated = True
                break
            slide_xml = _read_member(archive, filename)
            lines = _drawing_paragraphs(_parse_xml(slide_xml)) if slide_xml else []
            notes_name = _slide_notes_target(archive, number) or notes_by_number.get(number)
            notes: list[str] = []
            if notes_name:
                notes_xml = _read_member(archive, notes_name)
                if notes_xml:
                    # Drop slide-number placeholders ("12") from the notes page.
                    notes = [
                        line for line in _drawing_paragraphs(_parse_xml(notes_xml))
                        if not line.isdigit()
                    ]
            text = "\n".join(lines)
            if notes:
                text = f"{text}\nNotes: " + "\n".join(notes) if text else "Notes: " + "\n".join(notes)
            text = _clean(text)
            if not text:
                continue
            kept = budget.take(text)
            if kept is None:
                break
            result.segments.append(AttachmentSegment(f"slide {number}", number, kept))
    if not result.segments:
        result.warning = "The presentation contains no text."


def _docx_paragraph(node: ElementTree.Element) -> str:
    parts: list[str] = []
    for child in node.iter():
        if child.tag == f"{_W_NS}t" and child.text:
            parts.append(child.text)
        elif child.tag == f"{_W_NS}tab":
            parts.append("\t")
        elif child.tag in (f"{_W_NS}br", f"{_W_NS}cr"):
            parts.append("\n")
    return "".join(parts).strip()


def _docx_table(table: ElementTree.Element) -> list[str]:
    rows: list[str] = []
    for row in table.iter(f"{_W_NS}tr"):
        cells = []
        for cell in row.findall(f"{_W_NS}tc"):
            text = " ".join(
                value for value in (_docx_paragraph(p) for p in cell.iter(f"{_W_NS}p")) if value
            )
            cells.append(text)
        if any(cells):
            rows.append(" | ".join(cells))
    return rows


def _extract_docx(data: bytes, result: ExtractedAttachment, budget: _Budget) -> None:
    with _open_archive(data) as archive:
        document = _read_member(archive, "word/document.xml")
        if document is None:
            raise AttachmentError("The file is not a Word document.")
        root = _parse_xml(document)
    body = root.find(f"{_W_NS}body")
    lines: list[str] = []
    for child in list(body) if body is not None else []:
        if child.tag == f"{_W_NS}p":
            text = _docx_paragraph(child)
            if text:
                lines.append(text)
        elif child.tag == f"{_W_NS}tbl":
            lines.extend(_docx_table(child))
        elif child.tag == f"{_W_NS}sdt":
            for paragraph in child.iter(f"{_W_NS}p"):
                text = _docx_paragraph(paragraph)
                if text:
                    lines.append(text)
    text = _clean("\n".join(lines))
    kept = budget.take(text) if text else None
    if kept:
        result.segments.append(AttachmentSegment("", 1, kept))
    else:
        result.warning = "The document contains no text."


# ---------------------------------------------------------------------- text


def decode_text(data: bytes) -> str:
    """UTF-8 first, then BOM-marked UTF-16, a detector if available, then legacy CJK."""
    if data.startswith(b"\xef\xbb\xbf"):
        return data[3:].decode("utf-8", "replace")
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", "replace")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        pass
    try:  # optional dependency (present via httpx/requests stacks)
        from charset_normalizer import from_bytes

        best = from_bytes(data[:200_000]).best()
        if best is not None and best.encoding:
            return data.decode(best.encoding, "replace")
    except Exception:
        pass
    for encoding in ("gb18030", "shift_jis", "euc-kr"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def _extract_text(data: bytes, result: ExtractedAttachment, budget: _Budget) -> None:
    text = _clean(decode_text(data))
    kept = budget.take(text) if text else None
    if kept:
        result.segments.append(AttachmentSegment("", 1, kept))
    else:
        result.warning = "The file is empty."


_EXTRACTORS = {
    "pdf": _extract_pdf,
    "pptx": _extract_pptx,
    "docx": _extract_docx,
    **{extension: _extract_text for extension in TEXT_EXTENSIONS},
}


# ------------------------------------------------------------------- public


def extract_file(
    path: str | Path,
    *,
    attachment_id: str = "",
    name: str | None = None,
    char_budget: int = CHAR_BUDGET,
) -> ExtractedAttachment:
    """Extract one file synchronously (call through ``extract_attachments``)."""
    path = Path(path)
    display_name = (name or path.name or "attachment").strip() or "attachment"
    extension = path.suffix.lower().lstrip(".")
    if extension not in SUPPORTED_EXTENSIONS:
        # The shell may stage a picked file under an opaque name; the display
        # name keeps the original extension.
        named = Path(display_name).suffix.lower().lstrip(".")
        if named in SUPPORTED_EXTENSIONS:
            extension = named
    result = ExtractedAttachment(id=attachment_id, name=display_name, kind=extension)
    budget = _Budget(char_budget)
    try:
        if extension not in SUPPORTED_EXTENSIONS:
            raise AttachmentError(f"Unsupported file type .{extension or '?'}.")
        try:
            size = path.stat().st_size
        except OSError as error:
            raise AttachmentError("The file could not be opened.") from error
        if not path.is_file():
            raise AttachmentError("The file could not be opened.")
        if size > MAX_INPUT_BYTES:
            raise AttachmentError("The file is larger than 25 MB.")
        try:
            with path.open("rb") as handle:
                data = handle.read(MAX_INPUT_BYTES + 1)
        except OSError as error:
            raise AttachmentError("The file could not be opened.") from error
        if len(data) > MAX_INPUT_BYTES:
            raise AttachmentError("The file is larger than 25 MB.")
        _EXTRACTORS[extension](data, result, budget)
    except AttachmentError as error:
        result.warning = str(error)
        result.segments.clear()
    except Exception as error:  # defensive: parser bugs must not fail the job
        logger.warning("attachment extraction failed (%s, %s)", extension, type(error).__name__)
        result.warning = "The file could not be read."
        result.segments.clear()
    result.truncated = budget.truncated
    if result.truncated and result.warning is None:
        result.warning = f"Only the first {char_budget:,} characters are used."
    return result


def parse_attachment_items(items: Any) -> list[dict[str, str]]:
    """Validate the ``attachments`` task field (``[{id, path, name}]``)."""
    if items is None:
        return []
    if not isinstance(items, list):
        raise ValueError("attachments must be a list")
    if len(items) > MAX_ATTACHMENTS:
        raise ValueError(f"at most {MAX_ATTACHMENTS} attachments are allowed")
    parsed: list[dict[str, str]] = []
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ValueError("each attachment must be an object")
        path = item.get("path")
        if not isinstance(path, str) or not path.strip():
            raise ValueError("each attachment needs a path")
        attachment_id = item.get("id")
        name = item.get("name")
        parsed.append(
            {
                "id": attachment_id if isinstance(attachment_id, str) else str(index + 1),
                "path": path,
                "name": name if isinstance(name, str) and name.strip() else Path(path).name,
            }
        )
    return parsed


async def extract_attachments(
    items: Iterable[Mapping[str, str]], *, char_budget: int = CHAR_BUDGET
) -> list[ExtractedAttachment]:
    """Extract every attachment in a worker thread, preserving order."""
    results: list[ExtractedAttachment] = []
    for item in items:
        results.append(
            await asyncio.to_thread(
                extract_file,
                item["path"],
                attachment_id=item.get("id", ""),
                name=item.get("name"),
                char_budget=char_budget,
            )
        )
    return results
