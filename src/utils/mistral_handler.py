import logging
import os
import time

import mistralai
from mistralai import Mistral, ChatCompletionResponse

logger = logging.getLogger(__name__)


class MistralCompletionHandler:
    def __init__(self, model: str):
        self.model = model
        self.client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        self.rps_limit = 1
        self.tpm_limit = 500e3
        self.token_usage = {'total_tokens': 0, 'prompt_tokens': 0, 'completion_tokens': 0}
        self.last_request_time: float | None = None
        self.tokens_since_last_request = 0

    def _complete(self, messages: list[dict], **chat_complete_kwargs) -> ChatCompletionResponse:
        response = self.client.chat.complete(
            model=self.model,
            messages=messages,
            **chat_complete_kwargs
        )
        self.token_usage['total_tokens'] += response.usage.total_tokens
        self.token_usage['prompt_tokens'] += response.usage.prompt_tokens
        self.token_usage['completion_tokens'] += response.usage.completion_tokens
        self.last_request_time = time.time()
        self.tokens_since_last_request += response.usage.total_tokens
        return response

    def complete_with_retry(self, messages: list[dict], **chat_complete_kwargs) -> ChatCompletionResponse:
        """
        Complete the chat with the given messages and parameters.

        Args:
            messages: List of message dictionaries.
            *chat_complete_kwargs: Additional parameters for chat completion.

        Returns:
            ChatCompletionResponse: The response from the chat completion.
        """
        if self.last_request_time is not None:
            elapsed_time = time.time() - self.last_request_time
            if elapsed_time < self.rps_limit:
                logger.warning(f"Rate limit exceeded. Sleeping for {self.rps_limit - elapsed_time:.2f} seconds.")
                time.sleep(self.rps_limit - elapsed_time)
        try:
            response = self._complete(messages, **chat_complete_kwargs)
        except mistralai.models.sdkerror.SDKError as e:
            # if new token + tokens since last request token has become bigger than tpm_limit, wait for a minute since
            # last request self.last_request_time and retry the function
            if self.tokens_since_last_request > self.tpm_limit:
                logger.warning(f"Token limit exceeded. Sleeping for 60 seconds.")
                # wait for 60 seconds since self.last_request_time
                waiting_time = time.time() - self.last_request_time if self.last_request_time is not None else 60
                time.sleep(waiting_time)
                self.tokens_since_last_request = 0
                return self.complete_with_retry(messages, **chat_complete_kwargs)
            else:
                # if the error is not due to token limit, raise the error
                raise e
        return response
