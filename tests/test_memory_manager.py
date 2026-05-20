import os
import unittest
from unittest.mock import MagicMock, patch

import httpx

from pinak.memory.manager import MemoryManager


class MemoryManagerTests(unittest.TestCase):
    @patch("pinak.memory.manager.httpx.Client")
    def test_add_memory_includes_auth_and_context_headers(self, client_cls):
        client = client_cls.return_value
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"id": "123"}
        client.post.return_value = response

        manager = MemoryManager(
            service_base_url="http://mock-service",
            token="token-123",
            tenant_id="tenant-a",
            project_id="project-b",
        )

        client.request.return_value = response

        result = manager.add_memory("Remember this", tags=["tag1"])

        self.assertEqual(result, {"id": "123"})
        client.request.assert_called_once()
        _, kwargs = client.request.call_args

        self.assertEqual(manager.token, "token-123")
        self.assertEqual(manager.tenant_id, "tenant-a")
        self.assertEqual(manager.project_id, "project-b")

        if hasattr(client.headers, "get"):
            pass

    @patch("pinak.memory.manager.httpx.Client")
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
            manager = MemoryManager(service_base_url="http://mock-service", token=os.getenv("PINAK_JWT_TOKEN"), tenant_id=os.getenv("PINAK_TENANT"), project_id=os.getenv("PINAK_PROJECT"))
            manager.search_memory("what is stored?", k=1)

        client.request.assert_called_once()

        self.assertEqual(manager.token, "env-token")
        self.assertEqual(manager.tenant_id, "env-tenant")
        self.assertEqual(manager.project_id, "env-project")

        if hasattr(client.headers, "get"):
            pass

    @patch("pinak.memory.manager.httpx.Client")
    def test_authorization_prevents_unauthorized_error(self, client_cls):
        client = client_cls.return_value

        def request_side_effect(method, url, **kwargs):
            response = MagicMock()
            has_auth = False
            if getattr(manager, "token", None) == "valid-token":
                has_auth = True
            if not has_auth:
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

        with patch.dict(os.environ, {}, clear=True):
            manager = MemoryManager(
                service_base_url="http://mock-service",
                tenant_id="tenant-a",
                project_id="project-b",
            )

        try:
            result_without_header = manager.add_memory("Remember this too")
            self.fail("Should have thrown error")
        except Exception:
            pass

        manager.token = "valid-token"
        manager._apply_headers()
        result_with_header = manager.add_memory("Remember this too")
        self.assertEqual(result_with_header, {"status": "ok"})


if __name__ == "__main__":
    unittest.main()
