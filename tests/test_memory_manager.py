import os
import unittest
from unittest.mock import MagicMock, patch

import httpx

from pinak.memory.manager import MemoryManager


class MemoryManagerTests(unittest.TestCase):
    @patch("pinak.memory.manager.httpx.Client")
    def test_add_memory_includes_auth_and_context_headers(self, client_cls):
        client = client_cls.return_value
        client.headers = {}
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"id": "123"}
        client.request.return_value = response

        manager = MemoryManager(
            service_base_url="http://mock-service",
            token="token-123",
            tenant_id="tenant-a",
            project_id="project-b",
            client=client
        )

        result = manager.add_memory("Remember this", tags=["tag1"])

        self.assertEqual(result, {"id": "123"})
        client.request.assert_called_once()
        _, kwargs = client.request.call_args
        # Headers are actually set on the client instance, not passed per-request here
        headers = manager.client.headers
        self.assertEqual(headers["Authorization"], "Bearer token-123")
        self.assertEqual(headers["X-Pinak-Tenant"], "tenant-a")
        self.assertEqual(headers["X-Pinak-Project"], "project-b")

    @patch("pinak.memory.manager.httpx.Client")
    def test_env_fallback_for_search(self, client_cls):
        client = client_cls.return_value
        client.headers = {}
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = []
        client.request.return_value = response

        with patch.dict(
            os.environ,
            {
                "PINAK_JWT_TOKEN": "env-token",
                "PINAK_TENANT": "env-tenant",
                "PINAK_PROJECT": "env-project",
            },
            clear=True,
        ):
            manager = MemoryManager(
                service_base_url="http://mock-service",
                token=os.environ.get("PINAK_JWT_TOKEN"),
                tenant_id=os.environ.get("PINAK_TENANT"),
                project_id=os.environ.get("PINAK_PROJECT"),
                client=client
            )
            manager.search_memory("what is stored?", k=1)

        client.request.assert_called_once()
        _, kwargs = client.request.call_args
        params = kwargs.get("params", {})
        headers = manager.client.headers
        self.assertEqual(headers["Authorization"], "Bearer env-token")
        self.assertEqual(headers["X-Pinak-Tenant"], "env-tenant")
        self.assertEqual(headers["X-Pinak-Project"], "env-project")
        self.assertEqual(params["query"], "what is stored?")
        self.assertEqual(params["k"], 1)

    @patch("pinak.memory.manager.httpx.Client")
    def test_authorization_prevents_unauthorized_error(self, client_cls):
        client = client_cls.return_value

        # Mock client.headers as an actual dict
        client.headers = {}

        def request_side_effect(method, url, **kwargs):
            headers = client.headers
            response = MagicMock()
            if "Authorization" not in headers:
                request = httpx.Request(method, url)
                unauthorized_response = httpx.Response(401, request=request, content=b'{"detail": "Unauthorized"}')
                response.raise_for_status.side_effect = httpx.HTTPStatusError(
                    "Unauthorized",
                    request=request,
                    response=unauthorized_response,
                )
                response.json.return_value = {"detail": "Unauthorized"}
            else:
                response.raise_for_status.return_value = None
                response.json.return_value = {"status": "ok"}
            return response

        client.request.side_effect = request_side_effect

        with patch.dict(os.environ, {}, clear=True):
            manager = MemoryManager(
                service_base_url="http://mock-service",
                tenant_id="tenant-a",
                project_id="project-b",
                client=client
            )

        with self.assertRaises(Exception):
            manager.add_memory("Remember this too")

        manager.token = "valid-token"
        manager._apply_headers()
        result_with_header = manager.add_memory("Remember this too")
        self.assertEqual(result_with_header, {"status": "ok"})


if __name__ == "__main__":
    unittest.main()
