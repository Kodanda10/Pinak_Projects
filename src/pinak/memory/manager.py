import os
import httpx
from typing import List, Dict, Any, Optional, Mapping

class MemoryManager:
    """An API client for the Pinak Memory Service."""

    def __init__(
        self,
        service_base_url: Optional[str] = None,
        token: Optional[str] = None,
        default_headers: Optional[Mapping[str, str]] = None,
        timeout: float = 10.0,
        client: Optional[httpx.Client] = None,
    ):
        """Initializes the client.

        - service_base_url: memory API root (default from PINAK_MEMORY_URL or http://localhost:8001)
        - token: optional bearer token for Authorization
        - default_headers: extra headers to send for all requests
        - timeout: request timeout seconds
        """
        base = service_base_url or os.getenv("PINAK_MEMORY_URL", "http://localhost:8001")
        self.base_url = f"{base}/api/v1/memory"
        self._timeout = timeout
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        env_token = os.getenv("PINAK_TOKEN")
        if env_token and "Authorization" not in headers:
            headers["Authorization"] = f"Bearer {env_token}"
        if default_headers:
            headers.update(default_headers)
        if client is not None:
            # merge headers onto provided client
            try:
                client.headers.update(headers)
            except Exception:
                pass
            self.client = client
        else:
            self.client = httpx.Client(headers=headers, timeout=self._timeout)
        print(f"MemoryManager client initialized. Pointing to service at: {self.base_url}")

    def add_memory(self, content: str, tags: Optional[list] = None) -> Optional[Dict[str, Any]]:
        """Sends a request to the Memory Service to add a new memory."""
        try:
            response = self.client.post(
                f"{self.base_url}/add",
                json={"content": content, "tags": tags or []},
                timeout=self._timeout
            )
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            return response.json()
        except httpx.RequestError as e:
            print(f"An error occurred while requesting {e.request.url!r}.")
            return None
        except httpx.HTTPStatusError as e:
            print(f"HTTP error: {e.response.status_code} {e.response.text}")
            return None

    def search_memory(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Sends a request to the Memory Service to search for memories."""
        try:
            response = self.client.get(
                f"{self.base_url}/search",
                params={"query": query, "k": k},
                timeout=self._timeout
            )
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            print(f"An error occurred while requesting {e.request.url!r}.")
            return []
        except httpx.HTTPStatusError as e:
            print(f"HTTP error: {e.response.status_code} {e.response.text}")
            return []

    def health(self) -> bool:
        try:
            r = self.client.get(self.base_url.rsplit('/api/v1/memory', 1)[0] + "/", timeout=self._timeout)
            return r.status_code == 200
        except Exception:
            return False
