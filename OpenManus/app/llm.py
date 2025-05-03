from typing import Dict, List, Optional, Union

import tiktoken
from openai import (
    APIError,
    AsyncAzureOpenAI,
    AsyncOpenAI,
    AuthenticationError,
    OpenAIError,
    RateLimitError,
)
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from OpenManus.app.bedrock import BedrockClient
from OpenManus.app.config import LLMSettings, config
from OpenManus.app.exceptions import TokenLimitExceeded
from OpenManus.app.logger import logger  # Assuming a logger is set up in your app
from OpenManus.app.schema.common import (
    TOOL_CHOICE_TYPE,
    TOOL_CHOICE_VALUES,
    ToolChoice,
)
from OpenManus.app.schema.message import Message, format_messages
from OpenManus.app.utils.token_counter import TokenCounter

REASONING_MODELS = ["o1", "o3-mini"]
MULTIMODAL_MODELS = [
    "gpt-4-vision-preview",
    "gpt-4o",
    "gpt-4o-mini",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
]


class LLM:
    _instances: Dict[str, "LLM"] = {}

    def __new__(
            cls, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        if config_name not in cls._instances:
            instance = super().__new__(cls)
            instance.__init__(config_name, llm_config)
            cls._instances[config_name] = instance
        return cls._instances[config_name]

    def __init__(
            self, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        if hasattr(self, "client"):
            return
        llm_config = llm_config or config.llm
        llm_config = llm_config.get(config_name, llm_config["default"])
        self.model = llm_config.model
        self.max_tokens = llm_config.max_tokens
        self.temperature = llm_config.temperature
        self.api_type = llm_config.api_type
        self.api_key = llm_config.api_key
        self.api_version = llm_config.api_version
        self.base_url = llm_config.base_url

        # Add token counting related attributes
        self.total_input_tokens = 0
        self.total_completion_tokens = 0
        self.max_input_tokens = getattr(llm_config, "max_input_tokens", None)

        self.tokenizer = self._init_tokenizer()
        self.client = self._init_client()
        self.token_counter = TokenCounter(self.tokenizer)

    def _init_tokenizer(self):
        if self.model in tiktoken.model.MODEL_TO_ENCODING:
            return tiktoken.encoding_for_model(self.model)
        else:
            return tiktoken.get_encoding("cl100k_base")

    def _init_client(self):
        if self.api_type == "azure":
            return AsyncAzureOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                api_version=self.api_version,
            )
        elif self.api_type == "aws":
            return BedrockClient()
        return AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    def count_tokens(self, text: str) -> int:
        """Calculate the number of tokens in a text"""
        return len(self.tokenizer.encode(text)) if text else 0

    def count_message_tokens(self, messages: List[dict]) -> int:
        return self.token_counter.count_message_tokens(messages)

    def update_token_count(self, input_tokens: int, completion_tokens: int = 0) -> None:
        """Update token counts"""
        # Only track tokens if max_input_tokens is set
        self.total_input_tokens += input_tokens
        self.total_completion_tokens += completion_tokens
        logger.info(
            f"Token usage: Input={input_tokens}, Completion={completion_tokens}, "
            f"Cumulative Input={self.total_input_tokens}, Cumulative Completion={self.total_completion_tokens}, "
            f"Total={input_tokens + completion_tokens}, Cumulative Total={self.total_input_tokens + self.total_completion_tokens}"
        )

    def check_token_limit(self, input_tokens: int) -> bool:
        """Check if token limits are exceeded"""
        return self.max_input_tokens is None or (self.total_input_tokens + input_tokens) <= self.max_input_tokens

    def get_limit_error_message(self, input_tokens: int) -> str:
        """Generate error message for token limit exceeded"""
        return (
            f"Request may exceed input token limit "
            f"(Current: {self.total_input_tokens}, Needed: {input_tokens}, Max: {self.max_input_tokens})"
            if self.max_input_tokens is not None
            else "Token limit exceeded"
        )

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type((OpenAIError, Exception, ValueError)),
    )
    async def ask_tool(
            self,
            messages: List[Union[dict, Message]],
            system_msgs: Optional[List[Union[dict, Message]]] = None,
            timeout: int = 300,
            tools: Optional[List[dict]] = None,
            tool_choice: TOOL_CHOICE_TYPE = ToolChoice.AUTO,  # type: ignore
            temperature: Optional[float] = None,
            **kwargs,
    ) -> ChatCompletionMessage | None:
        """
        Ask LLM using functions/tools and return the response.

        Args:
            messages: List of conversation messages
            system_msgs: Optional system messages to prepend
            timeout: Request timeout in seconds
            tools: List of tools to use
            tool_choice: Tool choice strategy
            temperature: Sampling temperature for the response
            **kwargs: Additional completion arguments

        Returns:
            ChatCompletionMessage: The model's response

        Raises:
            TokenLimitExceeded: If token limits are exceeded
            ValueError: If tools, tool_choice, or messages are invalid
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        try:
            # Validate tool_choice
            if tool_choice not in TOOL_CHOICE_VALUES:
                raise ValueError(f"Invalid tool_choice: {tool_choice}")

            # Check if the model supports images
            supports_images = self.model in MULTIMODAL_MODELS
            # Format messages
            formatted_messages = format_messages((system_msgs or []) + messages, supports_images)
            # Calculate input token count
            input_tokens = self.count_message_tokens(formatted_messages)

            # If there are tools, calculate token count for tool descriptions
            if tools:
                input_tokens += sum(self.count_tokens(str(tool)) for tool in tools)

            # Check if token limits are exceeded
            if not self.check_token_limit(input_tokens):
                raise TokenLimitExceeded(self.get_limit_error_message(input_tokens))

            # Validate tools if provided
            if tools:
                for tool in tools:
                    if not isinstance(tool, dict) or "type" not in tool:
                        raise ValueError("Each tool must be a dict with 'type' field")

            # Set up the completion request
            params = {
                "model": self.model,
                "messages": formatted_messages,
                "tools": tools,
                "tool_choice": tool_choice,
                "timeout": timeout,
                "stream": False,  # Always use non-streaming for tool requests
                **kwargs,
            }

            if self.model in REASONING_MODELS:
                params["max_completion_tokens"] = self.max_tokens
            else:
                params["max_tokens"] = self.max_tokens
                params["temperature"] = (
                    temperature if temperature is not None else self.temperature
                )

            response: ChatCompletion = await self.client.chat.completions.create(**params)

            # Check if response is valid
            if not response.choices or not response.choices[0].message:
                # raise ValueError("Invalid or empty response from LLM")
                return None

            # Update token counts
            self.update_token_count(response.usage.prompt_tokens, response.usage.completion_tokens)
            return response.choices[0].message

        except TokenLimitExceeded:
            # Re-raise token limit errors without logging
            raise
        except ValueError as ve:
            logger.error(f"Validation error in ask_tool: {ve}")
            raise
        except OpenAIError as oe:
            logger.error(f"OpenAI API error: {oe}")
            if isinstance(oe, AuthenticationError):
                logger.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.error(f"API error: {oe}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask_tool: {e}")
            raise
