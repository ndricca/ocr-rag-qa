import re

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    id: int = Field(..., description="The ID of the chunk")
    text: str = Field(..., description="The text of the chunk")


class ChunkWithEmbedding(Chunk):
    embedding: list[float] | None = Field(..., description="The embedding of the chunk")


def split_markdown_into_chunks(lines) -> list[Chunk]:
    """
    Split a list of lines from a Markdown file into chunks based on headers.
    Each chunk will contain all lines up to the next header, and headers will be
    included in the chunk as prefixes.

    Parameters:
        lines (list): List of lines from a Markdown file.

    Returns:
        list: List of strings, where each string is a chunk of Markdown text.
    """
    chunks = []
    current_chunk_lines = []
    header_stack = []
    header_pattern = re.compile(r'^(#+)\s+(.*)$')

    def start_new_chunk():
        if current_chunk_lines:
            chunks.append(''.join(current_chunk_lines))
            current_chunk_lines.clear()

    for line in lines:
        # check if the line is a header
        match = header_pattern.match(line)
        if match:
            start_new_chunk()  # store all previous lines whe a new header is found
            level = len(match.group(1))
            text = match.group(2)
            while header_stack and header_stack[-1][0] >= level:
                header_stack.pop()
            header_stack.append((level, text))
            prefix = '\n'.join(["#" * h[0] + " " + h[1] for h in header_stack[:-1]]) + '\n'
            current_chunk_lines.append(prefix)
        if line not in ['\n', '\r\n']:
            # update list of lines in current chunk (will be stored as chunk when a new header is found)
            current_chunk_lines.append(line)

    start_new_chunk()  # Add a last chunk with the remaining current lines from last iteration
    chunks_list = [Chunk(id=i, text=c) for i, c in enumerate(chunks)]
    return chunks_list

