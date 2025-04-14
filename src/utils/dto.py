import datetime
import uuid

from pydantic import BaseModel, Field


FIXED_FIELDS = {'conversation_id', 'correlation_id'}

class InputMessage(BaseModel):
    correlation_id: str = Field(default_factory=lambda: uuid.uuid4().hex, description="ID for correlation input and output")
    conversation_id: str = Field(..., description="The ID of the conversation")
    user_id: str = Field(..., description="The ID of the user")
    message: str = Field(..., description="The message content")
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().isoformat(), description="The timestamp of the message")

class OutputMessage(BaseModel):
    correlation_id: str = Field(default_factory=lambda: uuid.uuid4().hex, description="ID for correlation input and output")
    conversation_id: str = Field(..., description="The ID of the conversation")
    user_id: str = Field(..., description="The ID of the user")
    message: str = Field(..., description="The message content")
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now().isoformat(), description="The timestamp of the message")
