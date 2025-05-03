class ToolError(Exception):
    """Raised when a tool encounters an error."""

    def __init__(self, message):
        self.message = message


class OpenManusError(Exception):
    """Base exception for all OpenManus errors"""


class TokenLimitExceeded(OpenManusError):
    """Exception raised when the token limit is exceeded"""


class InvalidState(OpenManusError):
    """Exception raised when a state is invalid"""

class UnsupportedMessageRole(OpenManusError):
    """Exception raised when a message role is unsupported"""

class InvalidStartState(OpenManusError):
    """Exception raised when a start state is invalid"""

class McpServerUrlRequired(OpenManusError):
    """Exception raised when a server url is required"""

class McpServerCommandRequired(OpenManusError):
    """Exception raised when a command is required"""

class McpUnsupportedConnectionType(OpenManusError):
    """Exception raised when a connection type is not supported"""