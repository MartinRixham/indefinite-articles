import unittest
from unittest import mock

from indefinite_articles.llm import WaitLimit
from indefinite_articles.sent_request import SentRequest


class SentRequestTests(unittest.TestCase):
    def test_close_closes_response_then_connection(self):
        calls = mock.Mock()
        sent = SentRequest(calls.connection, calls.socket, calls.response, WaitLimit(5.0, True))
        sent.close()
        self.assertEqual(calls.mock_calls, [mock.call.response.close(), mock.call.connection.close()])

    def test_keeps_what_it_was_given(self):
        connection, stream_socket, response = mock.Mock(), mock.Mock(), mock.Mock()
        limit = WaitLimit(None, False)
        sent = SentRequest(connection, stream_socket, response, limit)
        self.assertIs(sent.connection, connection)
        self.assertIs(sent.stream_socket, stream_socket)
        self.assertIs(sent.response, response)
        self.assertIs(sent.limit, limit)


if __name__ == "__main__":
    unittest.main()
