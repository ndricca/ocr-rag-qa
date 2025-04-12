import os
import logging
from argparse import ArgumentParser

from mistralai import Mistral, OCRResponse

from utils.logger import filter_loggers
from utils.ocr import get_combined_markdown

filter_loggers({'httpcore': 'ERROR'})

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

api_key = os.environ["MISTRAL_API_KEY"]

client = Mistral(api_key=api_key)

parser = ArgumentParser(description="Upload a file to Mistral for OCR processing.")
parser.add_argument('--file', type=str, help="Path to the file to upload", required=False)
parser.add_argument('--file_id', type=str, help="File ID to use for OCR", required=False)  # 9d277797-a704-406c-bd99-a9803d0cf8f5

if __name__ == "__main__":
    args = parser.parse_args()
    file_path = args.file
    file_id = args.file_id

    if not file_path and not file_id:
        print("Please provide a file path using the --file argument or a file ID using the --file_id argument.")
        exit(1)

    # Upload the file
    if file_id:
        try:
            uploaded_pdf = client.files.retrieve(file_id=file_id)
            logger.debug("File retrieved successfully")
        except Exception as e:
            logger.error(f"Error retrieving file with id {file_id}: {e}")
            exit(1)
    elif file_path:
        if not os.path.exists(file_path):
            print(f"The file {file_path} does not exist.")
            exit(1)
        uploaded_pdf = client.files.upload(
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
    signed_url = client.files.get_signed_url(file_id=file_id)

    # Perform OCR
    try:
        ocr_result = client.ocr.process(
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

