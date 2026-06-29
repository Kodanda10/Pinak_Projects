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
        )

        result = manager.add_memory("Remember this", tags=["tag1"])

        self.assertEqual(result, {"id": "123"})
        client.request.assert_called_once()

        # Verify headers were set on the client instance properly
        self.assertEqual(client.headers["Authorization"], "Bearer token-123")
        self.assertEqual(client.headers["X-Pinak-Tenant"], "tenant-a")
        self.assertEqual(client.headers["X-Pinak-Project"], "project-b")

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
            manager = MemoryManager(service_base_url="http://mock-service")
            manager.search_memory("what is stored?", k=1)

        client.request.assert_called_once()
        _, kwargs = client.request.call_args

        self.assertEqual(client.headers["Authorization"], "Bearer env-token")
        self.assertEqual(client.headers["X-Pinak-Tenant"], "env-tenant")
        self.assertEqual(client.headers["X-Pinak-Project"], "env-project")

        params = kwargs["params"]
        self.assertEqual(params["query"], "what is stored?")
        self.assertEqual(params["k"], 1)

    @patch("pinak.memory.manager.httpx.Client")
    def test_authorization_prevents_unauthorized_error(self, client_cls):
        client = client_cls.return_value

        def request_side_effect(method, url, **kwargs):
            response = MagicMock()
            if "Authorization" not in client.headers:
                request = httpx.Request(method, url)
                unauthorized_response = httpx.Response(401, request=request)
                unauthorized_response._content = b'{"detail": "Unauthorized"}'
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

        client.headers = {}
        client.request.side_effect = request_side_effect

        with patch.dict(os.environ, {}, clear=True):
            manager = MemoryManager(
                service_base_url="http://mock-service",
                tenant_id="tenant-a",
                project_id="project-b",
            )

        with self.assertRaises(Exception) as context:
            manager.add_memory("Remember this too")
        self.assertIn("401", str(context.exception))

        manager.token = "valid-token"
        manager._apply_headers() # We need to actually apply it on the httpx client
        result_with_header = manager.add_memory("Remember this too")
        self.assertEqual(result_with_header, {"status": "ok"})


if __name__ == "__main__":
    unittest.main()
