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

        args, kwargs = client.request.call_args
        self.assertEqual(args[0], "POST")
        self.assertTrue(args[1].endswith("/add"))
        client.headers.__setitem__.assert_any_call("Authorization", "Bearer token-123")
        client.headers.__setitem__.assert_any_call("X-Pinak-Tenant", "tenant-a")
        client.headers.__setitem__.assert_any_call("X-Pinak-Project", "project-b")

    @patch("pinak.memory.manager.httpx.Client")
    def test_env_fallback_for_search(self, client_cls):
        client = client_cls.return_value
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = []
        client.request.return_value = response

        with patch.dict(
            os.environ,
            {
                "PINAK_JWT_TOKEN": "env-token",
                "PINAK_TENANT_ID": "env-tenant",
                "PINAK_PROJECT_ID": "env-project",
            },
            clear=True,
        ):
            manager = MemoryManager(service_base_url="http://mock-service", token="env-token", tenant_id="env-tenant", project_id="env-project", client=client)
            manager.search_memory("what is stored?", k=1)

        args, kwargs = client.request.call_args
        self.assertEqual(args[0], "GET")
        self.assertTrue(args[1].endswith("/search"))
        client.headers.__setitem__.assert_any_call("Authorization", "Bearer env-token")
        client.headers.__setitem__.assert_any_call("X-Pinak-Tenant", "env-tenant")
        client.headers.__setitem__.assert_any_call("X-Pinak-Project", "env-project")

    @patch("pinak.memory.manager.httpx.Client")
    def test_authorization_prevents_unauthorized_error(self, client_cls):
        client = client_cls.return_value

        def post_side_effect(method, url, **kwargs):
            headers = {}
            for call in client.headers.__setitem__.call_args_list:
                args, _ = call
                headers[args[0]] = args[1]

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
                response.raise_for_status.side_effect = None
                response.raise_for_status.return_value = None
                response.json.return_value = {"status": "ok"}
            return response

        client.request.side_effect = post_side_effect

        with patch.dict(os.environ, {}, clear=True):
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
