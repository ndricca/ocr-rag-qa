import logging
import os
import time
from typing import Literal

import mistralai
from mistralai import Mistral, ChatCompletionResponse, EmbeddingResponse

from llm_handlers.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class MistralHandler(BaseHandler):
    """
    This class handles the interaction with the Mistral API for chat and embedding tasks.
    It provides methods to complete, parse, and embed messages, as well as to handle rate limits and token usage.
    These methods are then invoked with retry logic in concrete method `invoke_with_retry`.
    """
    def __init__(self, chat_model: str = "mistral-small-latest", embed_model: str = "mistral-embed"):
        super().__init__(rps_limit=1, tpm_limit=500000)
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

    def update_state(self, response: ChatCompletionResponse | EmbeddingResponse):
        """
        Update the state of the handler with the response from the Mistral API.

        Parameters:
            response (ChatCompletionResponse): The response from the Mistral API.

        """
        self.token_usage['total_tokens'] += response.usage.total_tokens
        self.token_usage['prompt_tokens'] += response.usage.prompt_tokens
        self.token_usage['completion_tokens'] += response.usage.completion_tokens
        request_time = time.time()
        self.update_last_minute_state(request_time)

    def _complete(self, messages: list[dict], **chat_complete_kwargs) -> ChatCompletionResponse:
        response = self.client.chat.complete(
            model=self.chat_model,
            messages=messages,
            **chat_complete_kwargs
        )
        self.update_state(response)
        return response

    def _parse(self, messages: list[dict], **chat_parse_kwargs) -> ChatCompletionResponse:
        response = self.client.chat.parse(
            model=self.chat_model,
            messages=messages,
            **chat_parse_kwargs
        )
        self.update_state(response)
        return response

    def _embed(self, messages: list[dict], **embeddings_create_kwargs) -> EmbeddingResponse:
        if "to_embed_key" not in embeddings_create_kwargs:
            raise ValueError("to_embed_key must be provided for embedding.")
        to_embed_key = embeddings_create_kwargs.pop('to_embed_key')
        sentences = [m.get(to_embed_key) for m in messages]
        response = self.client.embeddings.create(
            model=self.embed_model,
            inputs=sentences,
            **embeddings_create_kwargs
        )
        self.update_state(response)
        return response

    def invoke_with_retry(self,
                          method: Literal["complete", "parse", "embed"],
                          **invoke_kwargs) -> ChatCompletionResponse | EmbeddingResponse:
        """
        Return the response from the Mistral API with retry logic for rate limits.

        Args:
            method: The method to invoke, either "complete", "parse" or "embed".
            messages: List of message dictionaries.
            *invoke_kwargs: Additional parameters for client methods invocation.

        Returns:
            ChatCompletionResponse: The response from the chat completion.
        """
        self.wait_if_rps_limit()
        try:
            if method == "complete":
                response = self._complete(**invoke_kwargs)
            elif method == "parse":
                response = self._parse(**invoke_kwargs)
            elif method == "embed":
                response = self._embed(**invoke_kwargs)
            else:
                raise ValueError(f"Unknown method: {method}")
        except Exception as e:
            logger.warning(f'method {method} failed. Try again (waiting if TPM limit). Current error: {e}')
            self.wait_if_tpm_limit()
            try:
                logger.debug('second invocation (after tpm or not')
                response = self.invoke_with_retry(method, **invoke_kwargs)
            except Exception as retry_exception:
                # try and raise again the error in case it was not a problem of TPM
                logger.error(f"Retry failed: {retry_exception}")
                raise retry_exception
        return response

