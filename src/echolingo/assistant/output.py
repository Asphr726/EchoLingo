"""Clean-up of model output: titles, preambles, code fences and headings."""

from __future__ import annotations

import re

TITLE_MAX_CHARS = 60

_TITLE_LABEL_RE = re.compile(
    r"^(?:title|session title|标题|標題|题目|題目|タイトル|제목)\s*[:：]\s*", re.IGNORECASE
)
_WRAPPERS = {
    '"': '"',
    "'": "'",
    "“": "”",
    "‘": "’",
    "「": "」",
    "『": "』",
    "《": "》",
    "«": "»",
    "*": "*",
    "_": "_",
}
_TRAILING_PUNCTUATION = " \t.。!！?？,，;；:：、…·-–—~～"
_BULLET_RE = re.compile(r"^(?:[-*•·]\s+|\d{1,2}[.)、]\s+)")
_H1_RE = re.compile(r"^#[ \t]+(.+?)[ \t#]*$", re.MULTILINE)
_H2_RE = re.compile(r"^##[ \t]+(.+?)[ \t#]*$", re.MULTILINE)
_FENCE_OPEN_RE = re.compile(r"\A\s*```[a-zA-Z]*[ \t]*\n")
_FENCE_CLOSE_RE = re.compile(r"\n```[ \t]*\s*\Z")
_TIME_RANGE_RE = re.compile(r"\(\s*\d{1,3}:\d{2}\s*[–—-]\s*\d{1,3}:\d{2}\s*\)")


def _cut(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    if space >= limit // 2:
        cut = cut[:space]
    return cut


def sanitize_title(text: str | None, limit: int = TITLE_MAX_CHARS) -> str:
    """A single-line title: no heading marks, labels, quotes, markdown or
    trailing punctuation, at most ``limit`` characters."""
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("```"):
            continue
        line = re.sub(r"^#+\s*", "", line)
        line = _BULLET_RE.sub("", line)
        line = line.replace("**", "").replace("__", "").replace("`", "")
        line = _TITLE_LABEL_RE.sub("", line.strip())
        line = " ".join(line.split())
        changed = True
        while changed and len(line) >= 2:
            changed = False
            closing = _WRAPPERS.get(line[0])
            if closing is not None and line.endswith(closing):
                line = line[1:-1].strip()
                changed = True
        line = line.strip("\"“”")
        line = line.rstrip(_TRAILING_PUNCTUATION).strip()
        if not line:
            continue
        line = _cut(line, limit).rstrip(_TRAILING_PUNCTUATION).strip()
        if line:
            return line
    return ""


def extract_title(markdown: str) -> str:
    """The sanitized text of the first ``# `` heading, or ''."""
    match = _H1_RE.search(markdown or "")
    return sanitize_title(match.group(1)) if match else ""


def section_headings(markdown: str) -> list[str]:
    """The ``## `` headings of ``markdown`` in order (time ranges kept)."""
    return [" ".join(match.group(1).split()) for match in _H2_RE.finditer(markdown or "")]


def heading_key(heading: str) -> str:
    """Heading text for duplicate detection (no time range, case-folded)."""
    return " ".join(_TIME_RANGE_RE.sub("", heading).split()).casefold()


def strip_code_fence(markdown: str) -> str:
    """Remove a code fence wrapped around the whole answer."""
    text = markdown or ""
    opened = _FENCE_OPEN_RE.match(text)
    if opened:
        text = text[opened.end() :]
        text = _FENCE_CLOSE_RE.sub("", text)
    return text


def clean_markdown(markdown: str) -> str:
    return strip_code_fence(markdown).strip()


_FENCE_LINE_RE = re.compile(r"^[ \t]*```", re.MULTILINE)


def _opens_fence(preamble: str) -> bool:
    """Whether discarded text leaves a code fence open (odd fence count)."""
    return len(_FENCE_LINE_RE.findall(preamble)) % 2 == 1


def _holdable(line: str, *, complete: bool) -> bool:
    """A line that may belong to a closing code fence at the end of the answer."""
    stripped = line.strip()
    if not stripped:
        return True
    if set(stripped) != {"`"}:
        return False
    return len(stripped) >= 3 or not complete


class PreambleGate:
    """Drop chatter a model writes before the first heading of its answer.

    Text is held back until a heading line of the expected level appears
    (anything before it is discarded) or ``limit`` characters arrive without
    one (then everything is released unchanged, minus an opening code fence).

    When the discarded text opened a code fence (the model wrapped its answer
    in ```` ```markdown ````), the matching closing fence at the end of the
    answer is dropped too; otherwise it would turn everything that follows
    (the next part of a long document) into a code block.
    """

    def __init__(self, *, sections_only: bool, limit: int) -> None:
        # A whole document starts at "# " (or "## " without a title); a part of
        # a long document starts at its first "## " section.
        self._pattern = re.compile(
            r"^##[ \t]" if sections_only else r"^#{1,2}[ \t]", re.MULTILINE
        )
        self._limit = limit
        self._pending = ""
        self._fenced = False
        self._held = ""
        self.open = False

    def feed(self, text: str) -> str:
        if self.open:
            return self._pass(text)
        self._pending += text
        match = self._pattern.search(self._pending)
        if match is not None:
            preamble = self._pending[: match.start()]
            body = self._pending[match.start() :]
            self._pending = ""
            self.open = True
            self._fenced = _opens_fence(preamble)
            return self._pass(body)
        if len(self._pending) > self._limit:
            return self._open_without_heading()
        return ""

    def finish(self) -> str:
        """End of the answer: whatever is still held back, minus a closing
        fence that matches a dropped opening fence."""
        released = "" if self.open else self._open_without_heading()
        if not self._fenced:
            return released
        held = released + self._held
        self._held = ""
        lines = held.splitlines(keepends=True)
        for index in range(len(lines) - 1, -1, -1):
            if lines[index].strip():
                if set(lines[index].strip()) == {"`"}:
                    del lines[index]
                break
        return "".join(lines)

    def _open_without_heading(self) -> str:
        pending = self._pending
        self._pending = ""
        self.open = True
        opened = _FENCE_OPEN_RE.match(pending)
        if opened is None:
            return pending.lstrip()
        body = pending[opened.end() :]
        closed = _FENCE_CLOSE_RE.search(body)
        if closed is not None:
            return body[: closed.start()].lstrip()
        # The closing fence is still to come.
        self._fenced = True
        return self._pass(body.lstrip())

    def _pass(self, text: str) -> str:
        """Release ``text``, holding back a possible closing fence at its end."""
        if not self._fenced:
            return text
        buffer = self._held + text
        lines = buffer.splitlines(keepends=True)
        keep = len(lines)
        while keep > 0:
            line = lines[keep - 1]
            complete = line.endswith(("\n", "\r"))
            if not _holdable(line, complete=complete):
                break
            keep -= 1
        released = "".join(lines[:keep])
        self._held = buffer[len(released) :]
        return released


_H1_LINE_RE = re.compile(r"^#[ \t]", re.MULTILINE)


def trim_preamble(markdown: str) -> str:
    """The answer from its first ``# `` heading on, without chatter before it
    or a code fence wrapped around it; unchanged (minus a whole-answer fence)
    when it has no such heading."""
    text = markdown or ""
    match = _H1_LINE_RE.search(text)
    if match is None:
        return clean_markdown(text)
    preamble, body = text[: match.start()], text[match.start() :]
    if _opens_fence(preamble):
        body = _FENCE_CLOSE_RE.sub("", body.rstrip())
    return body.strip()


def split_header(markdown: str) -> tuple[str, str]:
    """``(head, rest)``: text before the first ``## `` heading and the rest."""
    text = clean_markdown(markdown)
    match = _H2_RE.search(text)
    if match is None:
        return text, ""
    return text[: match.start()].strip(), text[match.start() :].strip()


def drop_repeated_sections(markdown: str, known_headings: list[str]) -> str:
    """Remove ``## `` blocks whose heading was already written."""
    known = {heading_key(heading) for heading in known_headings}
    text = markdown.strip()
    if not text:
        return ""
    matches = list(_H2_RE.finditer(text))
    if not matches:
        return text
    kept: list[str] = []
    prefix = text[: matches[0].start()].strip()
    if prefix:
        kept.append(prefix)
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[match.start() : end].strip()
        if heading_key(match.group(1)) in known:
            continue
        kept.append(block)
    return "\n\n".join(kept)
