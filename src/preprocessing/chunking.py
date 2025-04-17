import logging
import re

from mistralai import OCRResponse
from pydantic import BaseModel, Field

from qdrant_client.models import PointStruct

logger = logging.getLogger(__name__)


class Chunk(BaseModel):
    id: int = Field(..., description="The ID of the chunk")
    text: str = Field(..., description="The text of the chunk")

    def __repr__(self):
        return f"ID: {self.id}\nText:\n{self.text}"

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



def split_markdown_into_chunks(lines: list[str]) -> list[Chunk]:
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


def add_header_overlap_to_chunks(ocr_response: OCRResponse) -> list[Chunk]:
    LAST_HEADER_REGEX = r"(^(#{1,6}) .*?$)(?![\s\S]*^#{1,6} .*?$)"
    FIRST_HEADER_REGEX = r"(^(#{1,6}) .*?$)"

    idx = 0
    chunks = []
    for i, page in enumerate(ocr_response.pages):
        current_md = page.markdown

        # add previous and next header text to current chunk for further context

        if i > 0:  # skip for first page
            # search last header level from previous page and, if present, add text after current chunk
            previous_md = ocr_response.pages[i - 1].markdown
            previous_last_header_match = re.search(LAST_HEADER_REGEX, previous_md, re.MULTILINE)
            if previous_last_header_match:
                current_md += "<\br>PAGE_BREAK<\br>" + previous_md[previous_last_header_match.start():]

        if i < len(ocr_response.pages) - 1:  # skip for last page
            # search first header level from next page and, if present, add text before current chunk
            next_md = ocr_response.pages[i + 1].markdown
            next_first_header_match = re.search(FIRST_HEADER_REGEX, next_md, re.MULTILINE)
            if next_first_header_match:
                current_md += "\n\n<\br>PAGE_BREAK<\br>\n\n" + next_md[:next_first_header_match.start()]

        # split if current page is too long
        if len(current_md) > 20000:  # TODO make this a parameter depending on a tokenizer and MAX_TOKENS
            split_idx = len(current_md) // 2  # simple split using half of the text
            current_md_start = current_md[:split_idx]
            current_md_end = current_md[split_idx:]
            chunks.append(Chunk(id=idx, text=current_md_start))
            chunks.append(Chunk(id=idx+1, text=current_md_end))
        else:
            chunks.append(Chunk(id=idx, text=current_md))
        idx += 1

    return chunks


def split_chunks_using_prev_headers(ocr_response: OCRResponse) -> list[Chunk]:
    chunks = []
    for i, page in enumerate(ocr_response.pages):
        if i == 0:
            chunks.append(Chunk(id=i, text=page.markdown))
        else:
            last_header_regex = r"(^(#{1,6}) .*?$)(?![\s\S]*^#{1,6} .*?$)"
            first_header_regex = r"(^(#{1,6}) .*?$)"
            # take last header level from previous page and the first header level in the current page
            # we use 100 as default for a very inner subparagraph
            previous_md = chunks[-1].text
            current_md = page.markdown
            previous_last_header_match = re.search(last_header_regex, previous_md, re.MULTILINE)
            if previous_last_header_match:
                previous_last_header_level = len(previous_last_header_match.group(2))
                logger.debug(f'previous page last header: "{previous_last_header_match.group(1)}"')
            else:
                previous_last_header_level = 100
                logger.debug('previous page has no header')

            current_first_header_match = re.search(first_header_regex, current_md, re.MULTILINE)
            if current_first_header_match:
                current_first_header_level = len(current_first_header_match.group(2))
                logger.debug(f'current page first header: "{current_first_header_match.group(1)}"')
            else:
                current_first_header_level = 100
                logger.debug('current page has no header')

            # it last header in previous page is < first header in the current page (e.g. ## vs ###) (=prev is more important)
            # then we split current page add the text before first header to previous chunk
            if previous_last_header_level < current_first_header_level:
                text_before_curr_first_header = current_md[
                                                :current_first_header_match.start()] if current_first_header_match else current_md
                text_after_curr_first_header = current_md[
                                               current_first_header_match.start():] if current_first_header_match else ""
                chunks[-1].text += "\n" + text_before_curr_first_header
                if text_after_curr_first_header:
                    chunks.append(Chunk(id=i, text=text_after_curr_first_header))
            # otherwise if last header in previous page is > first header in current page (e.g. ### vs ##) (=prev is less important)
            # then we split previous page and add text after last header in previous page to current page/chunk
            elif previous_last_header_level > current_first_header_level:
                text_before_prev_last_header = previous_md[
                                               :previous_last_header_match.start()] if previous_last_header_match else previous_md
                text_after_prev_last_header = previous_md[
                                              previous_last_header_match.start():] if previous_last_header_match else ""
                chunks[-1].text = text_before_prev_last_header
                chunks.append(Chunk(id=i, text=current_md + "\n" + text_after_prev_last_header))
            else:
                # if both levels are equals we add the text of the current page to the last chunk
                chunks[-1].text += "\n" + current_md


def add_overlap_to_chunks(ocr_response: OCRResponse) -> list[Chunk]:
    idx = 0
    overlap_chars = 1000
    chunks = []
    for i, page in enumerate(ocr_response.pages):
        current_md = page.markdown

        # add previous page overlap to current for further context
        if i > 0:  # skip for first page
            previous_text = ocr_response.pages[i - 1].markdown[overlap_chars:]
            current_md = "\n\n..." + previous_text + "<\br>PAGE_BREAK<\br>" + current_md

        if i < len(ocr_response.pages) - 1:  # skip for last page
            # add next page overlap to current for further context
            next_md = ocr_response.pages[i + 1].markdown[:overlap_chars]
            current_md += "\n\n<\br>PAGE_BREAK<\br>\n\n" + next_md + "...\n\n"

        # split if current page is too long
        if len(current_md) > 20000:  # TODO make this a parameter depending on a tokenizer and MAX_TOKENS
            split_idx = len(current_md) // 2  # simple split using half of the text
            current_md_start = current_md[:split_idx]
            current_md_end = current_md[split_idx:]
            chunks.append(Chunk(id=idx, text=current_md_start))
            chunks.append(Chunk(id=idx + 1, text=current_md_end))
        else:
            chunks.append(Chunk(id=idx, text=current_md))
        idx += 1
    return chunks
