import httpx
from pinak.memory.manager import MemoryManager


def test_memory_manager_sets_auth_header():
    token = "abc123"
    # Mock transport to capture request
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("Authorization") == f"Bearer {token}"
        if request.url.path.endswith("/add"):
            return httpx.Response(201, json={"id": "x", "content": "c", "tags": []})
        return httpx.Response(200, json=[])

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    mm = MemoryManager(service_base_url="http://local", token=token, client=client)
    r = mm.add_memory("c", ["t"])  # should not raise
    assert r and r.get("id") == "x"


def test_memory_manager_search_ok():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/search"):
            return httpx.Response(200, json=[{"id": "1", "content": "a", "tags": []}])
        return httpx.Response(200, json={"status": "ok"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    mm = MemoryManager(service_base_url="http://local", client=client)
    res = mm.search_memory("q")
    assert isinstance(res, list) and res and res[0]["content"] == "a"


def test_health_true_when_root_ok():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    mm = MemoryManager(service_base_url="http://local", client=client)
    assert mm.health() is True

