"""Fixed-size (strategy A) and structure-aware (strategies B/C) chunking, per page.

Every chunk keeps document_id, source, page and the `chapter` in force at the chunk start,
so strategies differ only in how text is split (A vs B) or whether the filter is used (B vs C).
"""
import re

from .ingestion import chapter_at
from .models import Chunk, DocumentPage

HEADING_RE = re.compile(r"^(Chương\s|Mục\s+\d|Điều\s+\d+\s*\.)", re.IGNORECASE)
CONTAINER_RE = re.compile(r"^(Chương\s|Mục\s*\d)", re.IGNORECASE)
SENTENCE_RE =re.compile(r"(?<=[.;:])\s+")


def _make(page: DocumentPage, index: int, text: str, offset: int) -> Chunk:
    metadata = {}
    chapter = chapter_at(page, offset)
    if chapter:  # unknown chapter (preamble before Chương I) is omitted, never guessed
        metadata["chapter"] = chapter
    return Chunk(chunk_id=f"p{page.page:03d}-c{index:02d}", document_id=page.document_id,
                 source=page.source, page=page.page, text=text, metadata=metadata)


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Overlapping character windows over one page."""
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and < size")
    text, chunks, start = page.text, [], 0
    while start < len(text):
        piece = text[start:start + size].strip()
        if piece:
            chunks.append(_make(page, len(chunks), piece, start))
        if start + size >= len(text):
            break
        start += size - overlap
    return chunks


def _paragraphs(text: str) -> list[tuple[int, str]]:
    """(offset, paragraph) split on blank lines; OCR keeps blank lines between paragraphs."""
    return [(m.start(), m.group().strip()) for m in re.finditer(r"\S[\s\S]*?(?=\n\s*\n|\Z)", text)]


def _split_long(text: str, max_size: int) -> list[str]:
    """Split an oversized paragraph at sentence/clause ends, hard-cutting only as last resort."""
    parts, current = [], ""
    for sentence in SENTENCE_RE.split(text):
        while len(sentence) > max_size:  # single sentence longer than max_size
            if current:
                parts.append(current)
                current = ""
            cut = sentence.rfind(" ", 0, max_size)
            if cut <= 0:
                cut = max_size
            parts.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if current and len(current) + 1 + len(sentence) > max_size:
            parts.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        parts.append(current)
    return parts


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Group whole paragraphs up to max_size; a Chương/Mục/Điều heading always starts a new chunk."""
    chunks: list[Chunk] = []
    group, group_offset = "", 0

    def flush() -> None:
        nonlocal group
        if group:
            chunks.append(_make(page, len(chunks), group, group_offset))
        group = ""

    def only_container_heading() -> bool:
        # a lone 'Chương ...'/'Mục ...' title must stay with the Điều that follows it
        return bool(CONTAINER_RE.match(group)) and len(group) < 160

    for offset, paragraph in _paragraphs(page.text):
        pieces = _split_long(paragraph, max_size) if len(paragraph) > max_size else [paragraph]
        for k, piece in enumerate(pieces):
            starts_section = k == 0 and HEADING_RE.match(piece) is not None
            if group and ((starts_section and not only_container_heading())
                          or len(group) + 2 + len(piece) > max_size):
                flush()
            if not group:
                group_offset = offset
            group = f"{group}\n\n{piece}" if group else piece
    flush()
    return chunks
