# Document OCR and RAG Question Answering

In this project, we will build a system that can extract text from documents using OCR (Optical Character Recognition) and then use that text to answer questions using a RAG (Retrieval-Augmented Generation) model.
The system will consist of two main components:
1. Document Processing: This offline component is responsible for processing input document using OCR and storing the extracted text in a vector database.
2. Question Answering: This online component is responsible for answering questions based on the text stored in the vector database using an Agentic RAG workflow.

The project will be implemented using the following technologies:
- Mistral OCR: A powerful OCR engine for extracting text from documents.
# TODO add other details

## Usage

To reproduce this code you have to set some environment variables (MistralAI, JinaAI).

You can set the API key in your environment by creating a .env file similar `sample.env` file versioned.
Remember to not add to git your .env file with the API key.

You will also need docker or podman installed to run Qdrant, the vector database. Follow this [link](https://qdrant.tech/documentation/quickstart/#how-to-get-started-with-qdrant-locally) for local setup.
In case of using podman on Windows, you need to create a .qdrant_storage folder in project root and then run the following command:
```bash
podman run -p 6333:6333 -p 6334:6334 -v ".\.qdrant_storage:/qdrant/storage/:z" qdrant
```
Web UI is available at [localhost:6333/dashboard](http://localhost:6333/dashboard)

