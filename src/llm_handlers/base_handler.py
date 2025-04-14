from abc import ABC, abstractmethod
from typing import Any


class BaseHandler(ABC):
    """
    Abstract base class for LLM handlers.
    """

    def __init__(self):
        self.token_usage = {'total_tokens': 0, 'prompt_tokens': 0, 'completion_tokens': 0}

    @abstractmethod
    def update_state(self, response: Any):
        """
        Update the state of the handler with tokens usage.
        """
        pass

    @abstractmethod
    def invoke_with_retry(self, request, *args, **kwargs) -> Any:
        """
        Handle the request and return a response.
        """
        pass

