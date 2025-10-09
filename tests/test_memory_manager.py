import os
import unittest
from unittest.mock import MagicMock, patch

import httpx

from src.pinak.memory.manager import MemoryManager


class MemoryManagerTests(unittest.TestCase):
    @patch("src.pinak.memory.manager.httpx.Client")
    def test_add_memory_includes_auth_and_context_headers(self, client_cls):
        client = client_cls.return_value
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"id": "123"}
        client.post.return_value = response

        manager = MemoryManager(
            service_base_url="http://mock-service",
            token="token-123",
            tenant="tenant-a",
            project_id="project-b",
        )

        result = manager.add_memory("Remember this", tags=["tag1"])

        self.assertEqual(result, {"id": "123"})
        client.post.assert_called_once()
        _, kwargs = client.post.call_args
        headers = kwargs["headers"]
        params = kwargs["params"]
        self.assertEqual(headers["Authorization"], "Bearer token-123")
        self.assertEqual(headers["X-Tenant-ID"], "tenant-a")
        self.assertEqual(headers["X-Project-ID"], "project-b")
        self.assertEqual(params["tenant"], "tenant-a")
        self.assertEqual(params["project_id"], "project-b")

    @patch("src.pinak.memory.manager.httpx.Client")
    def test_env_fallback_for_search(self, client_cls):
        client = client_cls.return_value
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = []
        client.get.return_value = response

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

        client.get.assert_called_once()
        _, kwargs = client.get.call_args
        headers = kwargs["headers"]
        params = kwargs["params"]
        self.assertEqual(headers["Authorization"], "Bearer env-token")
        self.assertEqual(headers["X-Tenant-ID"], "env-tenant")
        self.assertEqual(headers["X-Project-ID"], "env-project")
        self.assertEqual(params["tenant"], "env-tenant")
        self.assertEqual(params["project_id"], "env-project")

    @patch("src.pinak.memory.manager.httpx.Client")
    def test_authorization_prevents_unauthorized_error(self, client_cls):
        client = client_cls.return_value

        def post_side_effect(url, **kwargs):
            headers = kwargs.get("headers", {})
            response = MagicMock()
            if "Authorization" not in headers:
                request = httpx.Request("POST", url)
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

        client.post.side_effect = post_side_effect

        with patch.dict(os.environ, {}, clear=True):
            manager = MemoryManager(
                service_base_url="http://mock-service",
                tenant="tenant-a",
                project_id="project-b",
            )

        result_without_header = manager.add_memory("Remember this too")
        self.assertIsNone(result_without_header)

        manager.token = "valid-token"
        result_with_header = manager.add_memory("Remember this too")
        self.assertEqual(result_with_header, {"status": "ok"})


if __name__ == "__main__":
    unittest.main()
