"""Bounded, overlapping chunks that never cross physical PDF pages."""


def split_text(text: str, size: int = 800, overlap: int = 100) -> list[str]:
    if size < 100 or not 0 <= overlap < size:
        raise ValueError("Chunk size must be >=100; overlap must be >=0 and less than size")
    text = text.strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # Prefer a nearby whitespace boundary without creating tiny chunks.
            boundary = max(
                text.rfind("\n", start + size // 2, end), text.rfind(" ", start + size // 2, end)
            )
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks
