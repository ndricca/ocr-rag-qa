import copy
import logging
from argparse import ArgumentParser
from typing import Iterable

import markdownify
from unstructured.chunking.title import chunk_by_title
from unstructured.chunking.basic import chunk_elements
from unstructured.documents.elements import Table, Element, Text
from unstructured.partition.md import partition_md

from utils.logger import filter_loggers, LOG_CONFIG

filter_loggers({'httpcore': 'ERROR', 'httpx': 'ERROR'})
logging.basicConfig(**LOG_CONFIG)

logger = logging.getLogger(__name__)

parser = ArgumentParser(description="Process Mistral OCR output to create chunks.")
# parser.add_argument('--processed_file', type=str, help="Path to the processed file")  # data/processed/ocr_result_9d277797-a704-406c-bd99-a9803d0cf8f5.json
parser.add_argument('--md_file', type=str,
                    help="Path to the markdown file")  # data/processed/ocr_result_9d277797-a704-406c-bd99-a9803d0cf8f5.md


def table_to_md(part: Table) -> Text:
    """
    Convert a table element text to markdown.
    """
    # Convert the table element to a markdown well-formatted table (string)
    md_table_text = markdownify.markdownify(part.metadata.text_as_html)
    new_part = Text(text=md_table_text, metadata=part.metadata)
    return new_part


def fix_tables_before_chunking(elements: list[Element]) -> Iterable[Element]:
    for e in elements:
        if isinstance(e, Table):
            yield table_to_md(e)
        else:
            yield e


if __name__ == "__main__":
    args = parser.parse_args()
    md_file = args.md_file

    logger.info(f'Partitioning markdown file "{md_file}"')
    # partition_md is a function that partitions a markdown file into its constituent elements
    # it internally uses the markdown library to parse the file and convert it into HTML
    # then it uses the partition_html function to partition the HTML into its constituent elements
    partition_list = partition_md(md_file)

    # checks on partition list
    max_part = max(partition_list, key=lambda x: len(x.text))
    logger.info(f"max text length: {len(max_part.text)}")
    logger.debug(f"max text is in partition element with id {max_part.id}")
    logger.debug(f"max text is a {max_part.__class__.__name__} element")
    max_part_text = table_to_md(max_part).text if isinstance(max_part, Table) else max_part.text
    logger.debug(f"max text is what follows: \n{max_part_text}")

    # chunking
    fixed_partition_list = list(fix_tables_before_chunking(partition_list))
    chunk_list = chunk_elements(fixed_partition_list,  include_orig_elements=True,
                                max_characters=5000, new_after_n_chars=2500)
    chunk_list = chunk_by_title(fixed_partition_list, combine_text_under_n_chars=2000, include_orig_elements=True,
                                max_characters=5000, multipage_sections=True)

    for part in chunk_list:
        print("#" * 50)
        print(part.text)


