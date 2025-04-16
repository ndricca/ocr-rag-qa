import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Literal

logger = logging.getLogger(__name__)

class BaseHandler(ABC):
    """
    Abstract base class for LLM handlers.
    It implements two abstract methods: `update_state` and `invoke_with_retry`.

    The `update_state` method is used to update the state of the handler with tokens usage.
    The `invoke_with_retry` method is used to handle the request and return a response.
    Underlying classes should implement these methods to provide specific functionality.
    """

    def __init__(self, rps_limit: int | None = None, tpm_limit: int | None = None):
        self.token_usage = {'total_tokens': 0, 'prompt_tokens': 0, 'completion_tokens': 0}
        self.last_request_time: float | None = None
        self.rps_limit = rps_limit
        self.tpm_limit = tpm_limit
        self.tokens_in_minute: int = 0
        self.requests_in_minute: int = 0


    @abstractmethod
    def update_state(self, response: Any):
        """
        Update the state of the handler with tokens usage.
        """
        pass

    @abstractmethod
    def _complete(self, messages: list[dict], *args, **kwargs): pass

    @abstractmethod
    def _parse(self, messages: list[dict], *args, **kwargs): pass

    @abstractmethod
    def _embed(self, messages: list[dict], *args, **kwargs): pass

    def invoke_with_retry(self, method: Literal["complete", "parse", "embed"], *args, **kwargs) -> Any:
        """
        Handle the request and return a response.
        """
        # TODO understand if this may be generalized here
        pass

    def update_last_minute_state(self, request_time: float):
        """
        Update number of requests and number of  tokens used in the last minute comparing current to last request time.
        """
        if self.last_request_time is not None and (request_time - self.last_request_time) < 60:
            self.requests_in_minute += 1
            self.tokens_in_minute += self.token_usage['total_tokens']
        else:
            self.requests_in_minute = 0
            self.tokens_in_minute = 0
        self.last_request_time = request_time

    def wait_if_tpm_limit(self):
        if self.tpm_limit is None:
            return  # nothing to do here
        if self.tokens_in_minute >= self.tpm_limit:
            elapsed_time = time.time() - self.last_request_time
            logger.warning(f"TPM limit exceeded. Sleeping for {60 - elapsed_time:.2f} seconds.")
            time.sleep(60 - elapsed_time)

    def wait_if_rps_limit(self):
        if self.rps_limit is None:
            return  # nothing to do here
        if self.requests_in_minute > self.rps_limit:
            request_time = time.time()
            elapsed_time = request_time - self.last_request_time
            logger.warning(f"RPS limit exceeded. Sleeping for {60 - elapsed_time:.2f} seconds.")
            time.sleep(60 - elapsed_time)

