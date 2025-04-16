import json
import logging
import os.path
import re
from argparse import ArgumentParser

from pydantic import BaseModel, Field

from llm_handlers.jina_handler import JinaHandler
from preprocessing.chunking import split_markdown_into_chunks, ChunkWithEmbedding
from utils.logger import filter_loggers, LOG_CONFIG
from llm_handlers.mistral_handler import MistralHandler

filter_loggers({'httpcore': 'ERROR', 'httpx': 'ERROR'})
logging.basicConfig(**LOG_CONFIG)

logger = logging.getLogger(__name__)

parser = ArgumentParser(description="Process Mistral OCR output to create chunks.")
parser.add_argument('--md_file', type=str,
                    help="Path to the Markdown formatted file from OCR")  # data/processed/ocr_result_9d277797-a704-406c-bd99-a9803d0cf8f5.md


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

