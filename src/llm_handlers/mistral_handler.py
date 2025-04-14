import logging
import os
import time
from typing import Literal

import mistralai
from mistralai import Mistral, ChatCompletionResponse, EmbeddingResponse

logger = logging.getLogger(__name__)


class MistralHandler:
    def __init__(self, chat_model: str = "mistral-small-latest", embed_model: str = "mistral-embed"):
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        self.rps_limit = 1
        self.tpm_limit = 500e3
        self.token_usage = {'total_tokens': 0, 'prompt_tokens': 0, 'completion_tokens': 0}
        self.last_request_time: float | None = None
        self.tokens_since_last_request = 0

    def update_state(self, response: ChatCompletionResponse | EmbeddingResponse):
        """
        Update the state of the handler with the response from the Mistral API.

        Parameters:
            response (ChatCompletionResponse): The response from the Mistral API.

        """
        self.token_usage['total_tokens'] += response.usage.total_tokens
        self.token_usage['prompt_tokens'] += response.usage.prompt_tokens
        self.token_usage['completion_tokens'] += response.usage.completion_tokens
        self.last_request_time = time.time()
        self.tokens_since_last_request += response.usage.total_tokens

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

    def _embed(self, messages: list[dict], to_embed_key: str, **embeddings_create_kwargs) -> EmbeddingResponse:
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
                          messages: list[dict],
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
        if self.last_request_time is not None:
            elapsed_time = time.time() - self.last_request_time
            if elapsed_time < self.rps_limit:
                logger.warning(f"Rate limit exceeded. Sleeping for {self.rps_limit - elapsed_time:.2f} seconds.")
                time.sleep(self.rps_limit - elapsed_time)
        try:
            match method:
                case "complete":
                    response = self._complete(messages, **invoke_kwargs)
                case "parse":
                    response = self._parse(messages, **invoke_kwargs)
                case "embed":
                    if "to_embed_key" not in invoke_kwargs:
                        raise ValueError("to_embed_key must be provided for embedding.")
                    to_embed_key = invoke_kwargs.get("to_embed_key")
                    embed_invoke_kwargs = {k: v for k, v in invoke_kwargs.items() if k != "to_embed_key"}
                    response = self._embed(messages, to_embed_key=to_embed_key, **embed_invoke_kwargs)
                case _:
                    raise ValueError(f"Invalid method: {method}. Use 'complete' or 'parse'.")
        except mistralai.models.sdkerror.SDKError as e:
            # if new token + tokens since last request token has become bigger than tpm_limit, wait for a minute since
            # last request self.last_request_time and retry the function
            if self.tokens_since_last_request > self.tpm_limit:
                logger.warning(f"Token limit exceeded. Sleeping for 60 seconds.")
                # wait for 60 seconds since self.last_request_time
                waiting_time = time.time() - self.last_request_time if self.last_request_time is not None else 60
                time.sleep(waiting_time)
                self.tokens_since_last_request = 0
                return self.invoke_with_retry(method, messages, **invoke_kwargs)
            else:
                logger.warning(f"Retrying after 60 seconds after the following error occurred: {e}")
                # if the error is not due to token limit, try again after a longer sleep than raise the error
                try:
                    logger.warning(f"Retrying after error: {e}")
                    time.sleep(30)
                    return self.invoke_with_retry(method, messages, **invoke_kwargs)
                except Exception as e:
                    raise e
        return response
