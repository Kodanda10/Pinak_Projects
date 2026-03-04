import os
import unittest
from unittest.mock import MagicMock, patch

import httpx

from pinak.memory.manager import MemoryManager


class MemoryManagerTests(unittest.TestCase):
    def test_add_memory_includes_auth_and_context_headers(self):
        client = MagicMock()
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
        headers = client.headers
        self.assertEqual(headers["Authorization"], "Bearer token-123")
        self.assertEqual(headers["X-Pinak-Tenant"], "tenant-a")
        self.assertEqual(headers["X-Pinak-Project"], "project-b")

    def test_env_fallback_for_search(self):
        client = MagicMock()
        client.headers = {}
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = []
        client.request.return_value = response

        manager = MemoryManager(
            service_base_url="http://mock-service",
            token="env-token",
            tenant_id="env-tenant",
            project_id="env-project",
            client=client
        )
        manager.search_memory("what is stored?", k=1)

        client.request.assert_called_once()
        headers = client.headers
        self.assertEqual(headers["Authorization"], "Bearer env-token")
        self.assertEqual(headers["X-Pinak-Tenant"], "env-tenant")
        self.assertEqual(headers["X-Pinak-Project"], "env-project")

    def test_authorization_prevents_unauthorized_error(self):
        client = MagicMock()
        client.headers = {}

        def request_side_effect(method, url, **kwargs):
            headers = client.headers
            response = MagicMock()
            if "Authorization" not in headers:
                request = httpx.Request(method, url)
                unauthorized_response = httpx.Response(401, request=request)
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

        manager = MemoryManager(
            service_base_url="http://mock-service",
            tenant_id="tenant-a",
            project_id="project-b",
            client=client
        )

        from pinak.memory.manager import MemoryManagerError
        with self.assertRaises(MemoryManagerError):
            manager.add_memory("Remember this too")

        manager.configure(token="valid-token")
        result_with_header = manager.add_memory("Remember this too")
        self.assertEqual(result_with_header, {"status": "ok"})


if __name__ == "__main__":
    unittest.main()
