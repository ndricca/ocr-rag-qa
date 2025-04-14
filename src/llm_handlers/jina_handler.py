import logging
import os
import time
from typing import Literal

import httpx
import requests
from pydantic import BaseModel
from requests import Response

from llm_handlers.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class JinaEmbeddingUsage(BaseModel):
    total_tokens: int = 0
    prompt_tokens: int = 0


class JinaVector(BaseModel):
    object: Literal["embedding"] = "embedding"
    index: int = 0
    embedding: list[float] = []


class JinaEmbeddingResponse(BaseModel):
    usage: JinaEmbeddingUsage
    data: list[JinaVector] = []

    @classmethod
    def from_response(cls, response: Response) -> "JinaEmbeddingResponse":
        """
        Create a JinaEmbeddingResponse object from an HTTP response.

        Args:
            response (httpx.Response): The HTTP response object.

        Returns:
            JinaEmbeddingResponse: The JinaEmbeddingResponse object.
        """
        data = response.json()
        usage = JinaEmbeddingUsage(**data.get("usage", {}))
        vectors = [JinaVector(**vector) for vector in data.get("data", [])]
        return cls(usage=usage, data=vectors)


class JinaHandler(BaseHandler):
    def __init__(self, embed_model: str = "jina-clip-v2"):
        super().__init__()
        self.url = 'https://api.jina.ai/v1/embeddings'
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + os.getenv("JINAAI_API_KEY")
        }
        self.embed_model = embed_model

    def update_state(self, response: JinaEmbeddingResponse):
        """
        Update the state of the handler with the response from the Jina AI API.

        Parameters:

        """
        self.token_usage['total_tokens'] += response.usage.total_tokens
        self.token_usage['prompt_tokens'] += response.usage.prompt_tokens

    def _embed(self, messages: list[dict], to_embed_key: str, **embeddings_create_kwargs) -> JinaEmbeddingResponse:
        data = {
            'model': self.embed_model,
            'input': [{'text': m.get(to_embed_key)} for m in messages],
            **embeddings_create_kwargs
        }
        response = requests.post(self.url, headers=self.headers, json=data)
        response.raise_for_status()
        jina_embedding_response = JinaEmbeddingResponse.from_response(response)
        self.update_state(jina_embedding_response)
        return jina_embedding_response

    def invoke_with_retry(self,
                          method: Literal["complete", "parse", "embed"],
                          messages: list[dict],
                          **invoke_kwargs) -> JinaEmbeddingResponse:
        """
        Return the response from the Jina API with retry logic for rate limits.

        Args:
            method: The method to invoke, either "complete", "parse" or "embed".
            messages: List of message dictionaries.
            *invoke_kwargs: Additional parameters for client methods invocation.

        Returns:
            JinaEmbeddingResponse: The response from the Jina AI embeddings endpoint.
        """
        try:
            match method:
                case "complete":
                    raise NotImplementedError("Chat completion is not implemented for Jina.")
                case "parse":
                    raise NotImplementedError("Chat parsing is not implemented for Jina.")
                case "embed":
                    if "to_embed_key" not in invoke_kwargs:
                        raise ValueError("to_embed_key must be provided for embedding.")
                    to_embed_key = invoke_kwargs.get("to_embed_key")
                    embed_invoke_kwargs = {k: v for k, v in invoke_kwargs.items() if k != "to_embed_key"}
                    response = self._embed(messages, to_embed_key=to_embed_key, **embed_invoke_kwargs)
                case _:
                    raise ValueError(f"Invalid method: {method}. Use 'complete' or 'parse'.")
        except Exception as e:
            try:
                logger.warning(f"Retrying after error: {e}")
                time.sleep(10)
                return self.invoke_with_retry(method, messages, **invoke_kwargs)
            except Exception as retry_exception:
                logger.error(f"Retry failed: {retry_exception}")
                raise retry_exception
        return response
