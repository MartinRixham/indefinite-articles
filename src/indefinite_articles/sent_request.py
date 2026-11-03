"""An LLM request whose response is arriving."""

import http.client
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .llm import WaitLimit


class SentRequest:
    """A request whose response headers have arrived, with the limit it was sent under."""

    def __init__(self, connection: http.client.HTTPConnection, stream_socket: Any,
                 response: http.client.HTTPResponse, limit: "WaitLimit"):
        self.connection = connection
        self.stream_socket = stream_socket  # ours: http.client drops connection.sock when the server will close
        self.response = response
        self.limit = limit

    def close(self):
        # Closing the connection makes Ollama and llama-server abandon the request.
        self.response.close()
        self.connection.close()
