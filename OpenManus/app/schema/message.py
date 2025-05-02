from typing import Any, List, Optional, Union

from pydantic import BaseModel, Field

from OpenManus.app.schema.common import ToolCall, Role, ROLE_TYPE, ROLE_VALUES

__all__ = ['Message']


class Message(BaseModel):
    """Represents a chat message in the conversation"""

    role: ROLE_TYPE = Field(...)  # type: ignore
    content: Optional[str] = Field(default=None)
    tool_calls: Optional[List[ToolCall]] = Field(default=None)
    name: Optional[str] = Field(default=None)
    tool_call_id: Optional[str] = Field(default=None)
    base64_image: Optional[str] = Field(default=None)

    def __add__(self, other) -> List["Message"]:
        """支持 Message + list 或 Message + Message 的操作"""
        if isinstance(other, list):
            return [self] + other
        elif isinstance(other, Message):
            return [self, other]
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(self).__name__}' and '{type(other).__name__}'"
            )

    def __radd__(self, other) -> List["Message"]:
        """支持 list + Message 的操作"""
        if isinstance(other, list):
            return other + [self]
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(other).__name__}' and '{type(self).__name__}'"
            )

    def to_dict(self) -> dict:
        """Convert message to dictionary format"""
        message = {"role": self.role}
        if self.content is not None:
            message["content"] = self.content
        if self.tool_calls is not None:
            message["tool_calls"] = [tool_call.dict() for tool_call in self.tool_calls]
        if self.name is not None:
            message["name"] = self.name
        if self.tool_call_id is not None:
            message["tool_call_id"] = self.tool_call_id
        if self.base64_image is not None:
            message["base64_image"] = self.base64_image
        return message

    @classmethod
    def user_message(
            cls, content: str, base64_image: Optional[str] = None
    ) -> "Message":
        """Create a user message"""
        return cls(role=Role.USER, content=content, base64_image=base64_image)

    @classmethod
    def system_message(cls, content: str) -> "Message":
        """Create a system message"""
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def assistant_message(
            cls, content: Optional[str] = None, base64_image: Optional[str] = None
    ) -> "Message":
        """Create an assistant message"""
        return cls(role=Role.ASSISTANT, content=content, base64_image=base64_image)

    @classmethod
    def tool_message(
            cls, content: str, name, tool_call_id: str, base64_image: Optional[str] = None
    ) -> "Message":
        """Create a tool message"""
        return cls(
            role=Role.TOOL,
            content=content,
            name=name,
            tool_call_id=tool_call_id,
            base64_image=base64_image,
        )

    @classmethod
    def from_tool_calls(
            cls,
            tool_calls: List[Any],
            content: Union[str, List[str]] = "",
            base64_image: Optional[str] = None,
            **kwargs,
    ):
        """Create ToolCallsMessage from raw tool calls.

        Args:
            tool_calls: Raw tool calls from LLM
            content: Optional message content
            base64_image: Optional base64 encoded image
        """
        formatted_calls = [
            {"id": call.id, "function": call.function.model_dump(), "type": "function"}
            for call in tool_calls
        ]
        return cls(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=formatted_calls,
            base64_image=base64_image,
            **kwargs,
        )


def format_messages(
    messages: List[Union[dict, Message]], supports_images: bool = False
) -> List[dict]:
    """
    Format messages for LLM by converting them to OpenAI message format.

    Args:
        messages: List of messages that can be either dict or Message objects
        supports_images: Flag indicating if the target model supports image inputs

    Returns:
        List[dict]: List of formatted messages in OpenAI format

    Raises:
        ValueError: If messages are invalid or missing required fields
        TypeError: If unsupported message types are provided

    Examples:
        >>> msgs = [
        ...     Message.system_message("You are a helpful assistant"),
        ...     {"role": "user", "content": "Hello"},
        ...     Message.user_message("How are you?")
        ... ]
        >>> formatted = format_messages(msgs)
    """
    formatted_messages = []

    for message in messages:
        # Convert Message objects to dictionaries
        if isinstance(message, Message):
            message = message.to_dict()

        if isinstance(message, dict):
            # If message is a dict, ensure it has required fields
            if "role" not in message:
                raise ValueError("Message dict must contain 'role' field")

            # Process base64 images if present and model supports images
            if supports_images and message.get("base64_image"):
                # Initialize or convert content to appropriate format
                if not message.get("content"):
                    message["content"] = []
                elif isinstance(message["content"], str):
                    message["content"] = [
                        {"type": "text", "text": message["content"]}
                    ]
                elif isinstance(message["content"], list):
                    # Convert string items to proper text objects
                    message["content"] = [
                        (
                            {"type": "text", "text": item}
                            if isinstance(item, str)
                            else item
                        )
                        for item in message["content"]
                    ]

                # Add the image to content
                message["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{message['base64_image']}"
                        },
                    }
                )

                # Remove the base64_image field
                del message["base64_image"]
            # If model doesn't support images but message has base64_image, handle gracefully
            elif not supports_images and message.get("base64_image"):
                # Just remove the base64_image field and keep the text content
                del message["base64_image"]

            if "content" in message or "tool_calls" in message:
                formatted_messages.append(message)
            # else: do not include the message
        else:
            raise TypeError(f"Unsupported message type: {type(message)}")

    # Validate all messages have required fields
    for msg in formatted_messages:
        if msg["role"] not in ROLE_VALUES:
            raise ValueError(f"Invalid role: {msg['role']}")

    return formatted_messages