import json
import logging
import os.path
import re
from argparse import ArgumentParser

from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

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

parser = ArgumentParser(description="Load chunks into Qdrant.")
parser.add_argument('--embeddings_file', type=str,
                    help="Path to the embeddings file.")  # data/processed/jinaai_embeddings_ocr_result_9d277797-a704-406c-bd99-a9803d0cf8f5.json
parser.add_argument('--collection_name', type=str, help="Name of the collection to create", required=True)


class Chunk(BaseModel):
    id: int = Field(..., description="The ID of the chunk")
    text: str = Field(..., description="The text of the chunk")


class ChunkWithEmbedding(Chunk):
    embedding: list[float] | None = Field(..., description="The embedding of the chunk")

    @classmethod
    def from_json_elem(cls, json_elem: dict) -> "ChunkWithEmbedding":
        """
        Convert a JSON dictionaries toa ChunkWithEmbedding object.

        Args:
            json_elem (dict): The JSON dictionary to convert.

        Returns:
            ChunkWithEmbedding: The converted ChunkWithEmbedding object.
        """
        return cls(**json_elem)

    def to_qdrant_point_struct(self, **file_metadata_kwargs) -> PointStruct:
        """
        Convert the ChunkWithEmbedding object to a Qdrant PointStruct.

        Parameters:
            **file_metadata_kwargs: Additional metadata to include in the payload.

        Returns:
            PointStruct: The Qdrant PointStruct object.
        """
        return PointStruct(
            id=self.id,
            vector=self.embedding,
            payload={
                "text": self.text,
                **file_metadata_kwargs,
            }
        )


if __name__ == "__main__":
    args = parser.parse_args()
    embeddings_file = args.embeddings_file
    collection_name = args.collection_name

    with open(embeddings_file, "r", encoding='utf-8') as f:
        all_embeddings = json.load(f)

    embedded_chunks = [ChunkWithEmbedding.from_json_elem(elem) for elem in all_embeddings]

    point_structs = [chunk.to_qdrant_point_struct(filename=embeddings_file) for chunk in embedded_chunks]

    operation_info = client.upsert(
        collection_name=collection_name,
        wait=True,
        points=point_structs
    )

    logger.info(f'loaded {len(embedded_chunks)} files: {operation_info}')

