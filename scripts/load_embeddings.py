import json
import logging
import os
from argparse import ArgumentParser

from mistralai import Mistral
from qdrant_client import QdrantClient
from qdrant_client.conversions.common_types import Distance
from qdrant_client.grpc import VectorParams

from llm_handlers.jina_handler import JinaHandler
from preprocessing.chunking import split_markdown_into_chunks, ChunkWithEmbedding
from utils.logger import filter_loggers
from utils.ocr import get_combined_markdown

filter_loggers({'httpcore': 'ERROR'})

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

qdrant_url = os.getenv('QDRANT_URL')
qdrant_port = os.getenv('QDRANT_PORT')
if qdrant_url:
    qdrant_client = QdrantClient(url=f"{qdrant_url}:{qdrant_port}")
else:
    logger.debug(f'loading local Qdrant client at "{os.getenv("QDRANT_PATH_TO_DB")}"')
    os.makedirs(os.getenv("QDRANT_PATH_TO_DB"), exist_ok=True)
    qdrant_client = QdrantClient(path=os.getenv("QDRANT_PATH_TO_DB"))

api_key = os.environ["MISTRAL_API_KEY"]

mistral_ocr_client = Mistral(api_key=api_key)
jina_handler = JinaHandler(embed_model="jina-clip-v2")

parser = ArgumentParser(description="""
Document preprocessing pipeline.

Steps:
1. Upload the file to Mistral
2. Perform OCR on the uploaded file
3. Split the markdown into chunks
4. Create a collection in Qdrant
5. Embed the chunks
6. Load embeddings in Qdrant

Use --file to specify the raw file to upload.
""")
parser.add_argument('--file_path', type=str, help="Path to the file to upload", required=False)
# TODO implement --from_step and --to_step in code
# parser.add_argument('--from_step', type=int, help="Step to start from", required=False, default=0)
# parser.add_argument('--to_step', type=int, help="Step to end at", required=False, default=6)

if __name__ == "__main__":
    args = parser.parse_args()
    file_path = args.file_path

    logger.info("# 1. Upload the file to Mistral")
    uploaded_pdf = mistral_ocr_client.files.upload(
        file={
            "file_name": os.path.basename(file_path),
            "content": open(file_path, "rb"),
        },
        purpose="ocr"
    )
    # Get the file ID
    file_id = uploaded_pdf.id
    logger.info(f"File uploaded successfully with ID: {file_id}")

    # Get the file URL
    signed_url = mistral_ocr_client.files.get_signed_url(file_id=file_id)

    # Perform OCR
    logger.info("# 2. Perform OCR on the uploaded file")
    try:
        ocr_result = mistral_ocr_client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": signed_url.url,
            })
    except Exception as e:
        logger.error(f"Error processing OCR for file with id {file_id}: {e}")
        exit(1)

    print(ocr_result.model_dump_json(indent=2))

    # Store ocr_result as a json file with pages
    with open(f"data/processed/ocr_result_{file_id}.json", "w", encoding="utf-8") as json_file:
        json_file.write(ocr_result.model_dump_json(indent=2))

    # Store the result in a md file
    result_file_name = f"data/processed/ocr_result_{file_id}.md"
    with open(result_file_name, "w", encoding="utf-8") as result_file:
        result_file.write(get_combined_markdown(ocr_result))

    with open(result_file_name, "r", encoding='utf-8') as f:
        file_lines = f.readlines()

    logger.info("# 3. Split the markdown into chunks")
    # Split the markdown into chunks
    chunks = split_markdown_into_chunks(file_lines)

    logger.info("# 4. Create a collection in Qdrant")
    # Create a collection in Qdrant
    create_response = qdrant_client.create_collection(
        collection_name=file_id,
        vectors_config=VectorParams(size=len(chunks[0].text), distance=Distance.COSINE),
    )
    logger.info(f"Collection '{file_id}' created successfully")

    logger.info("# 5. Embed the chunks")
    chunks_with_embeddings = []
    for i, chunk in enumerate(chunks):
        logger.info(f'Chunking chunk {chunk.id:02} with {len(chunk.text)} characters')
        emb_resp = jina_handler.invoke_with_retry("embed", messages=[chunk.model_dump(include={'text'})], to_embed_key="text")
        chunk_with_emb = ChunkWithEmbedding(id=chunk.id, text=chunk.text, embedding=emb_resp.data[0].embedding)
        chunks_with_embeddings.append(chunk_with_emb)

    # Save the chunks with embeddings to a JSON file
    file, ext = os.path.splitext(result_file_name)
    path, filename = os.path.split(file)

    json_chunk_file = os.path.join(path, "jinaai_embeddings_" + filename + ".json")
    logger.info(f"Saving chunks with embeddings to {json_chunk_file}")
    with open(json_chunk_file, "w", encoding="utf-8") as json_file:
        json.dump([ce.model_dump() for ce in chunks_with_embeddings], json_file)

    logger.info("# 6. Load the embeddings into Qdrant")
    # Load embeddings in Qdrant
    with open(json_chunk_file, "r", encoding='utf-8') as f:
        all_embeddings = json.load(f)

    embedded_chunks = [ChunkWithEmbedding.from_json_elem(elem) for elem in all_embeddings]
    point_structs = [chunk.to_qdrant_point_struct(filename=json_chunk_file) for chunk in embedded_chunks]
    operation_info = qdrant_client.upsert(
        collection_name=file_id,
        wait=True,
        points=point_structs
    )
