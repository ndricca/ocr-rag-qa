import argparse
import logging
import os

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from utils.logger import filter_loggers, LOG_CONFIG
filter_loggers({'httpcore': 'ERROR', 'httpx': 'ERROR'})

logging.basicConfig(**LOG_CONFIG)
logger = logging.getLogger(__name__)

qdrant_url = os.getenv('QDRANT_URL')
qdrant_port = os.getenv('QDRANT_PORT')
if qdrant_url:
    client = QdrantClient(url=f"{qdrant_url}:{qdrant_port}")
else:
    logger.debug(f'loading local Qdrant client at "{os.getenv("QDRANT_PATH_TO_DB")}"')
    os.makedirs(os.getenv("QDRANT_PATH_TO_DB"), exist_ok=True)
    client = QdrantClient(path=os.getenv("QDRANT_PATH_TO_DB"))

parser = argparse.ArgumentParser(description="Create a Qdrant vector store.")
parser.add_argument('--collection_name', type=str, help="Name of the collection to create", required=True)


if __name__ == "__main__":
    args = parser.parse_args()
    collection_name = args.collection_name

    create_response = client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )
    logger.info(f"Collection '{collection_name}' created successfully")
