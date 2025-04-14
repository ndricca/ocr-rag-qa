import json
import logging
import re
from argparse import ArgumentParser
from enum import StrEnum

from mistralai import OCRResponse
from pydantic import BaseModel, Field

from utils.logger import filter_loggers, LOG_CONFIG
from utils.mistral_handler import MistralCompletionHandler

filter_loggers({'httpcore': 'ERROR', 'httpx': 'ERROR'})
logging.basicConfig(**LOG_CONFIG)

logger = logging.getLogger(__name__)

parser = ArgumentParser(description="Process Mistral OCR output to create chunks.")
parser.add_argument('--md_file', type=str, help="Path to the Markdown formatted file from OCR")  # data/processed/ocr_result_9d277797-a704-406c-bd99-a9803d0cf8f5.md


class ChunkingEvaluation(BaseModel):
    """
    Class to evaluate the chunking process.

    Contains the internal reasoning needed to check if the second page continues the first one and is_continuing boolean indicating if the text continues.
    """
    internal_reasoning: str = Field(..., description="Reason for continuing text across chunks or not.")
    is_continuing: bool = Field(..., description="True if the text in second page is the continuation of the what's in the first page. False otherwise.")


class ChunkMetadata(BaseModel):
    """
    Class to represent metadata for a chunk.

    Contains the IDs of the pages that are part of this chunk.
    """
    ids: list[int] = Field(..., description="List of page IDs that are part of this chunk.")


class Chunk(BaseModel):
    """
    Class to represent a chunk of text.

    Contains the text, metadata, and the chunk number.
    """
    num: int = Field(..., description="Chunk number")
    text: str = Field(..., description="Text of the chunk")
    metadata: ChunkMetadata = Field(..., description="Metadata associated with the chunk")


def chunk_ocr_pages_using_mistral_check(ocr_response: OCRResponse, m_handler: MistralCompletionHandler) -> list[Chunk]:
    """
    For each chunk - excluding the first - check if current and previous chunk should be merged into one.
    Check is done by calling the Mistral API with the text of both chunks.
    If the API returns a positive response, update previous chunk with current chunk text.
    """
    chunks = []
    for page in ocr_response.pages:
        if len(chunks) == 0:
            # create the first chunk
            chunks.append(Chunk(num=len(chunks), text=page.markdown, metadata=ChunkMetadata(ids=[page.index])))
            continue
        messages = [
            {
                "role": "system",
                "content": """Compare the text extracted from the following two pages and determine if they should be merged into one chunk.
You want to merge the two page if and only if what you see at the top of the second page is a continuation of what you see at the end of the first page, for example:
1. A table that is split into two pages
2. A continuing list
3. A sentence in a paragraph which starts in the first page and continues in the second page

Pages with similar topics but not continuing text should be kept separate.
Pages are separated by triple tilde.
"""
            },
            {
                "role": "user",
                "content": f"""
First page:
~~~~
{chunks[-1].text}
~~~~



Second page:
~~~~
{page.markdown}
~~~~
"""
            }
        ]
        cont_response = m_handler.invoke_with_retry("parse", messages, response_format=ChunkingEvaluation, temperature=0)
        continuing_evaluation: ChunkingEvaluation = cont_response.choices[0].message.parsed
        logger.debug(f"Chunking evaluation response: {continuing_evaluation.model_dump()}")
        if continuing_evaluation.is_continuing:
            # merge the two chunks
            chunks[-1].text += " " + page.markdown
            chunks[-1].metadata.ids.append(page.index)
        else:
            # add the new chunk
            chunks.append(Chunk(num=len(chunks), text=page.markdown, metadata=ChunkMetadata(ids=[page.index])))
    return chunks


if __name__ == "__main__":
    args = parser.parse_args()
    ocr_output_file = args.md_file
    model = args.model

    m_handler = MistralCompletionHandler(model=model)

    with open(ocr_output_file, "r", encoding='utf-8') as jf:
        ocr_response = OCRResponse(**json.load(jf))

    # merge pages
    chunked_docs = chunk_ocr_pages_using_mistral_check(ocr_response, m_handler)
    # checks on partition list
    max_part = max(chunked_docs, key=lambda x: len(x.text))
    logger.info(f"max text length: {len(max_part.text)}")
    logger.debug(f"max text is in partition element with id {max_part.id}")
    logger.debug(f"max text is what follows: \n{max_part.text}")


