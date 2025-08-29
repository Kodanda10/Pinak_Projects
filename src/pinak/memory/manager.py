import os
import httpx
import sys
from typing import List, Dict, Any, Optional, Mapping
try:
    from ..bridge.context import ProjectContext
except Exception:
    ProjectContext = None  # type: ignore

class MemoryManager:
    """An API client for the Pinak Memory Service."""

    def __init__(
        self,
        service_base_url: Optional[str] = None,
        token: Optional[str] = None,
        project_id: Optional[str] = None,
        default_headers: Optional[Mapping[str, str]] = None,
        timeout: float = 10.0,
        client: Optional[httpx.Client] = None,
    ):
        base = service_base_url or os.getenv("PINAK_MEMORY_URL")
        tok = token or os.getenv("PINAK_TOKEN")
        proj = project_id or os.getenv("PINAK_PROJECT_ID")
        ctx = None
        if ProjectContext and (not base or not tok or not proj):
            ctx = ProjectContext.find()
            if ctx:
                base = base or ctx.memory_url
                tok = tok or ctx.get_token()
                proj = proj or ctx.project_id
        base = base or "http://localhost:8001"
        self.base_url = f"{base}/api/v1/memory"
        self._timeout = timeout
        headers: Dict[str,str] = {}
        if tok:
            headers["Authorization"] = f"Bearer {tok}"
        if proj:
            headers["X-Pinak-Project"] = proj
        if ctx and getattr(ctx, "identity_fingerprint", None):
            headers["X-Pinak-Fingerprint"] = ctx.identity_fingerprint
        if default_headers:
            headers.update(default_headers)
        ca = os.getenv("PINAK_MEMORY_CA")
        verify_opt = ca if ca else True
        self.client = client or httpx.Client(headers=headers, timeout=self._timeout, verify=verify_opt)
        print(f"MemoryManager client initialized. Pointing to service at: {self.base_url}")

    def add_memory(self, content: str, tags: list = None) -> Dict[str, Any]:
        """Sends a request to the Memory Service to add a new memory."""
        try:
            response = self.client.post(
                f"{self.base_url}/add",
                json={"content": content, "tags": tags or []},
                timeout=self._timeout
            )
            if response.status_code == 401:
                print("Auth failed (401). Token may be missing or expired. Run 'pinak token --exp 120 --set' or 'pinak-bridge token rotate' and retry.", file=sys.stderr)
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
                timeout=self._timeout
            )
            if response.status_code == 401:
                print("Auth failed (401). Token may be missing or expired. Run 'pinak token --exp 120 --set' or 'pinak-bridge token rotate' and retry.", file=sys.stderr)
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            print(f"An error occurred while requesting {e.request.url!r}.")
            return []
    def health(self) -> bool:
        try:
            r = self.client.get(self.base_url.rsplit('/api/v1/memory', 1)[0] + "/", timeout=self._timeout)
            return r.status_code == 200
        except Exception:
            return False

    def add_event(self, payload: Dict[str, Any]) -> bool:
        """Send a generic event to the Memory Service (/event).
        Expected keys include: type, ts, project_id, and any domain-specific fields.
        """
        try:
            base = self.base_url.rsplit('/api/v1/memory', 1)[0]
            r = self.client.post(f"{base}/event", json=payload, timeout=self._timeout)
            if r.status_code < 400:
                return True
            return False
        except Exception:
            return False
