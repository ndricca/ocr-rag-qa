import json
import logging
import os

from qdrant_client import QdrantClient

from llm_handlers.base_handler import BaseHandler
from llm_handlers.jina_handler import JinaHandler

logger = logging.getLogger(__name__)


class ToolClient:
    def __init__(self, llm_handler: BaseHandler,
                 embeddings_handler: JinaHandler,
                 collection_name: str,
                 vector_db: QdrantClient | None = None
                 ):
        self.llm_handler = llm_handler
        self.embeddings_handler = embeddings_handler
        if vector_db:
            self.vector_db = vector_db
        else:
            qdrant_url = os.getenv('QDRANT_URL')
            qdrant_port = os.getenv('QDRANT_PORT')
            if qdrant_url:
                self.vector_db = QdrantClient(url=f"{qdrant_url}:{qdrant_port}")
            else:
                logger.debug(f'loading local Qdrant client at "{os.getenv("QDRANT_PATH_TO_DB")}"')
                os.makedirs(os.getenv("QDRANT_PATH_TO_DB"), exist_ok=True)
                self.vector_db = QdrantClient(path=os.getenv("QDRANT_PATH_TO_DB"),
                                              force_disable_check_same_thread=True)
        self.collection_name = collection_name

        # Initialize tools
        self.tool_to_function = {
            "math_reasoning": self.math_reasoning,
            "get_context": self.get_context
        }
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "math_reasoning",
                    "description": "Use this tool to approach the problem using mathematical reasoning.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The question to be answered."
                            },
                            "context": {
                                "type": "string",
                                "description": "Contextual information to help answer the question."
                            }
                        },
                        "required": ["question", "context"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_context",
                    "description": """ Use this tool to search for information in the vector store. Query search expands user input with an hypothetical answer to increase cosine similarity.""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "search_query": {
                                "type": "string",
                                "description": "Expanded sentence generated from user input to increase cosine similarity."
                            },
                            "limit": {
                                "type": "integer",
                                "description": """Max number of results to return, usually 3-5 are enough. Keep it higher for abstractive queries, lower for extractive (factual) queries.""",
                            }
                        },
                        "required": ["question"]
                    }
                }
            }
        ]


    def math_reasoning(self, question: str, context: str) -> str:
        """
        This function is used to perform mathematical reasoning based on the provided question and context.
        It is designed to generate a logical procedure to arrive at a numerical result.

        Parameters:
            question (str): The question to be answered.
            context (str): Contextual information to help answer the question.
        """
        messages = [
            {
                "role": "system",
                "content": f"""Il tuo compito è quello di scrivere il procedimento logico necessario ad ottenere un risultato numerico.
Non concentrarti sul'output finale, ma sul procedimento.

Considera quanto segue per rispondere alla domanda:
{context}
"""
            },
            {"role": "user", "content": question},
        ]
        response = self.llm_handler.invoke_with_retry(
            method="complete",
            messages=messages,
        )
        return response.choices[0].message.content

    def get_context(self, search_query: str, limit: int = 3, max_limit: int = 20, min_limit: int = 3) -> list[dict]:
        """
        This function is used to search for information in the vector store.
        It implements a query search that expands the user input with a hypothetical answer to increase cosine similarity with stored chunks.

        Parameters:
            search_query (str): The query to search for in the vector store.
            limit (int): The maximum number of results to return. Default is 3.
        """
        embedded_query_resp = self.embeddings_handler.invoke_with_retry(
            'embed', messages=[{'text': search_query}],
            to_embed_key="text",
        )
        embedded_query = embedded_query_resp.data[0].embedding
        logger.debug(f'limit {limit} forced to be between {min_limit} and {max_limit}')
        limit = min(limit, max_limit)
        limit = max(limit, min_limit)
        results = self.vector_db.query_points(
            collection_name=self.collection_name,
            query=embedded_query,
            limit=limit,
            with_payload=True,
        )
        result_text = sorted(results.points, key=lambda x: x.score, reverse=True)
        return [r.model_dump() for r in result_text]

    def execute(self, tool_name: str, tool_arguments: str) -> str:
        """
        Execute a tool based on the tool name and arguments provided.

        Parameters:
            tool_name (str): The name of the tool to execute.
            tool_arguments (str): The arguments for the tool in JSON format.
        """
        if tool_name in self.tool_to_function:
            function = self.tool_to_function[tool_name]
            try:
                tool_args = json.loads(tool_arguments)
                logger.debug(f"Calling tool {tool_name} with arguments: {tool_args}")
                result = function(**tool_args)
                logger.debug(f"Tool {tool_name} result: {result}")
                result = json.dumps(result, indent=4)
            except Exception as e:
                result = f"Error executing tool {tool_name}: {str(e)}"
        else:
            result = f"Tool {tool_name} not found in tool_to_function mapping."
        return  result