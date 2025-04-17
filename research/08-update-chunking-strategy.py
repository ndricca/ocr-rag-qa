import datetime
import json
import logging
import os.path
from argparse import ArgumentParser

from mistralai import OCRResponse
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

from llm_handlers.jina_handler import JinaHandler
from preprocessing.chunking import ChunkWithEmbedding, add_overlap_to_chunks
from utils.logger import filter_loggers, LOG_CONFIG

filter_loggers({'httpcore': 'ERROR', 'httpx': 'ERROR'})
logging.basicConfig(**LOG_CONFIG)

logger = logging.getLogger(__name__)

parser = ArgumentParser(description="Process Mistral OCR output to create chunks.")
parser.add_argument('--ocr_output_file', type=str,
                    help="Path to the OCR output resposne")  # data/processed/ocr_result_9d277797-a704-406c-bd99-a9803d0cf8f5.json

qdrant_url = os.getenv('QDRANT_URL')
qdrant_port = os.getenv('QDRANT_PORT')
if qdrant_url:
    qdrant_client = QdrantClient(url=f"{qdrant_url}:{qdrant_port}")
else:
    logger.debug(f'loading local Qdrant client at "{os.getenv("QDRANT_PATH_TO_DB")}"')
    os.makedirs(os.getenv("QDRANT_PATH_TO_DB"), exist_ok=True)
    qdrant_client = QdrantClient(path=os.getenv("QDRANT_PATH_TO_DB"))

if __name__ == "__main__":
    args = parser.parse_args()
    ocr_output_file = args.ocr_output_file
    file_id = "9d277797-a704-406c-bd99-a9803d0cf8f5"  # TODO make argument

    j_handler = JinaHandler(embed_model="jina-clip-v2")

    with open(ocr_output_file, "r", encoding='utf-8') as jf:
        ocr_response = OCRResponse(**json.load(jf))

    chunks = add_overlap_to_chunks(ocr_response)

    # Save the chunks with embeddings to a JSON file
    orig_file, ext = os.path.splitext(ocr_output_file)
    path, orig_filename = os.path.split(orig_file)

    json_chunk_file = os.path.join(path, "chunked_" + orig_filename + ".json")
    logger.info(f"Saving chunks with embeddings to {json_chunk_file}")
    with open(json_chunk_file, "w", encoding="utf-8") as json_file:
        json.dump([ce.model_dump() for ce in chunks], json_file)


    chunks_with_embeddings = []
    for chunk_to_embed in chunks:
        logger.info(f'Chunking chunk {chunk_to_embed.id:02} with {len(chunk_to_embed.text)} characters')
        if len(chunk_to_embed.text) == 0:
            logger.warning('Empty chunk, skipping...')
            continue
        emb_resp = j_handler.invoke_with_retry("embed", messages=[chunk_to_embed.model_dump(include={"text"})], to_embed_key="text")
        chunk_with_emb = ChunkWithEmbedding(id=chunk_to_embed.id, text=chunk_to_embed.text, embedding=emb_resp.data[0].embedding)
        chunks_with_embeddings.append(chunk_with_emb)



    logger.info("# 5. Create a collection in Qdrant")
    # Create a collection in Qdrant
    create_response = qdrant_client.create_collection(
        collection_name=file_id,
        vectors_config=VectorParams(size=len(chunks_with_embeddings[0].embedding), distance=Distance.COSINE),
    )
    logger.info(f"Collection '{file_id}' created successfully")


    json_chunk_file = os.path.join(path, f"jinaai_embeddings_{orig_filename}_{datetime.date.today().strftime('%Y%m%d')}.json")
    logger.info(f"Saving chunks with embeddings to {json_chunk_file}")
    with open(json_chunk_file, "w", encoding="utf-8") as json_file:
        json.dump([ce.model_dump() for ce in chunks_with_embeddings], json_file)

    point_structs = [chunk.to_qdrant_point_struct(filename=json_chunk_file) for chunk in chunks_with_embeddings]
    operation_info = qdrant_client.upsert(
        collection_name=file_id,
        wait=True,
        points=point_structs
    )
    logger.info(f"upsert data output: {operation_info}")