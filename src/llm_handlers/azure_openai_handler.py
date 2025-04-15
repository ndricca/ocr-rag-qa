import logging
import os
import time
from typing import Literal

from openai import AzureOpenAI
from openai.types import CreateEmbeddingResponse
from openai.types.chat import ChatCompletion

from llm_handlers.base_handler import BaseHandler


logger = logging.getLogger(__name__)


class AzureOpenaiHandler(BaseHandler):
    def __init__(self, chat_model: str = "crif-genai-gpt-4.1-2025-04-14", embed_model: str = "crif-genai-text-embedding-3-small"):
        super().__init__(rps_limit=250 * 60, tpm_limit=250000)
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
            api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        )

    def update_state(self, response: ChatCompletion):
        """
        Update the state of the handler with the response from the Azure OpenAI API.

        Parameters:
            response (ChatCompletion): The response from the Azure OpenAI API.
        """
        self.token_usage['total_tokens'] += response.usage.total_tokens
        self.token_usage['prompt_tokens'] += response.usage.prompt_tokens
        self.token_usage['completion_tokens'] += response.usage.completion_tokens
        request_time = time.time()
        self.update_tokens_in_minute(request_time)
        self.last_request_time = request_time

    def _complete(self, messages: list[dict], **chat_complete_kwargs) -> ChatCompletion:
        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=messages,
            **chat_complete_kwargs
        )
        self.update_state(response)
        return response

    def _parse(self, messages: list[dict], **chat_parse_kwargs) -> ChatCompletion:
        response = self.client.beta.chat.completions.parse(
            model=self.chat_model,
            messages=messages,
            **chat_parse_kwargs
        )
        self.update_state(response)
        return response

    def _embed(self, messages: list[dict], **embeddings_create_kwargs) ->  CreateEmbeddingResponse:
        if "to_embed_key" not in embeddings_create_kwargs:
            raise ValueError("to_embed_key must be provided for embedding.")
        to_embed_key = embeddings_create_kwargs.pop('to_embed_key')
        sentences = [m.get(to_embed_key) for m in messages]
        response = self.client.embeddings.create(
            input=messages,
            model=self.embed_model,
            **embeddings_create_kwargs
        )
        self.update_state(response)
        return response


    def invoke_with_retry(self,
                          method: Literal["complete", "parse", "embed"],
                          **invoke_kwargs) -> ChatCompletion | CreateEmbeddingResponse:
        """
        Return the response from the Azure OpenAI API with retry logic for rate limits.

        Args:
            method (str): The method to call on the Azure OpenAI API.
            **kwargs: Keyword arguments for the method.

        Returns:
            ChatCompletion: The response from the Azure OpenAI API.
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
                logger.debug('second invocation (after tpm or not)')
                response = self.invoke_with_retry(method, **invoke_kwargs)
            except Exception as retry_exception:
                # try and raise again the error in case it was not a problem of TPM
                logger.error(f"Retry failed: {retry_exception}")
                raise retry_exception
        return response

