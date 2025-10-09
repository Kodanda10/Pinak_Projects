import httpx
from typing import List, Dict, Any, Optional, Sequence


class MemoryManager:
    """An API client for the Pinak Memory Service."""

    def __init__(self, service_base_url: str = "http://localhost:8001"):
        """Initializes the client with the URL of the running memory service."""
        self.base_url = f"{service_base_url}/api/v1/memory"
        self.client = httpx.Client()
        print(f"MemoryManager client initialized. Pointing to service at: {self.base_url}")

    def add_memory(self, content: str, tags: list = None) -> Optional[Dict[str, Any]]:
        """Sends a request to the Memory Service to add a new memory."""
        try:
            response = self.client.post(
                f"{self.base_url}/add",
                json={"content": content, "tags": tags or []},
                timeout=10.0,
            )
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            return response.json()
        except httpx.RequestError as e:
            print(f"An error occurred while requesting {e.request.url!r}.")
            return None

    def search_memory(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Sends a request to the Memory Service to search for memories."""
        try:
            response = self.client.get(
                f"{self.base_url}/search",
                params={"query": query, "k": k},
                timeout=10.0,
            )
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            print(f"An error occurred while requesting {e.request.url!r}.")
            return []

    def search_v2(
        self,
        query: str,
        *,
        layers: Optional[Sequence[str]] = None,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search across specific memory layers using the v2 multi-layer endpoint."""

        params: Dict[str, Any] = {"query": query, "k": k}
        if layers:
            params["layers"] = ",".join(layers)
        try:
            response = self.client.get(
                f"{self.base_url}/search_v2",
                params=params,
                timeout=10.0,
            )
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            print(f"An error occurred while requesting {e.request.url!r}.")
            return []
