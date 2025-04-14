import json
import logging
import re
from argparse import ArgumentParser

from utils.logger import filter_loggers, LOG_CONFIG
from utils.mistral_handler import MistralCompletionHandler

filter_loggers({'httpcore': 'ERROR', 'httpx': 'ERROR'})
logging.basicConfig(**LOG_CONFIG)

logger = logging.getLogger(__name__)

parser = ArgumentParser(description="Process Mistral OCR output to create chunks.")
parser.add_argument('--md_file', type=str, help="Path to the Markdown formatted file from OCR")  # data/processed/ocr_result_9d277797-a704-406c-bd99-a9803d0cf8f5.md


def split_markdown_into_chunks(lines):
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
            prefix = '\n'.join(["#"*h[0] + " " + h[1] for h in header_stack[:-1]]) + '\n'
            current_chunk_lines.append(prefix)
        if line not in ['\n', '\r\n']:
            # update list of lines in current chunk (will be stored as chunk when a new header is found)
            current_chunk_lines.append(line)

    start_new_chunk()  # Add a last chunk with the remaining current lines from last iteration
    return chunks

if __name__ == "__main__":
    args = parser.parse_args()
    md_file = args.md_file

    with open(md_file, "r", encoding='utf-8') as f:
        file_lines = f.readlines()

    chunks = split_markdown_into_chunks(file_lines)

    print("ok")

