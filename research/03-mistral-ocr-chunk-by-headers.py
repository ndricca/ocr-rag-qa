import json
import logging
import os.path
import re
from argparse import ArgumentParser

from pydantic import BaseModel, Field

from llm_handlers.jina_handler import JinaHandler
from utils.logger import filter_loggers, LOG_CONFIG
from llm_handlers.mistral_handler import MistralHandler

filter_loggers({'httpcore': 'ERROR', 'httpx': 'ERROR'})
logging.basicConfig(**LOG_CONFIG)

logger = logging.getLogger(__name__)

parser = ArgumentParser(description="Process Mistral OCR output to create chunks.")
parser.add_argument('--md_file', type=str,
                    help="Path to the Markdown formatted file from OCR")  # data/processed/ocr_result_9d277797-a704-406c-bd99-a9803d0cf8f5.md


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


if __name__ == "__main__":
    args = parser.parse_args()
    md_file = args.md_file

    # m_handler = MistralHandler(chat_model="mistral-small-latest", embed_model="mistral-embed")
    j_handler = JinaHandler(embed_model="jina-clip-v2")

    with open(md_file, "r", encoding='utf-8') as f:
        file_lines = f.readlines()

    chunks = split_markdown_into_chunks(file_lines)

    chunks_with_embeddings = []
    for i, chunk in enumerate(chunks):
        logger.info(f'Chunking chunk {chunk.id:02} with {len(chunk.text)} characters')
        emb_resp = j_handler.invoke_with_retry("embed", [chunk.model_dump()], to_embed_key="text")
        chunk_with_emb = ChunkWithEmbedding(id=chunk.id, text=chunk.text, embedding=emb_resp.data[0].embedding)
        chunks_with_embeddings.append(chunk_with_emb)

    # Save the chunks with embeddings to a JSON file
    file, ext = os.path.splitext(md_file)
    path, filename = os.path.split(file)
    json_chunk_file = os.path.join(path, "jinaai_embeddings_" + filename + ".json")
    logger.info(f"Saving chunks with embeddings to {json_chunk_file}")
    with open(json_chunk_file, "w", encoding="utf-8") as json_file:
        json.dump([ce.model_dump() for ce in chunks_with_embeddings], json_file)

