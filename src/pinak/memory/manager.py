import logging
import os
from typing import Any, Dict, List, Optional

import httpx


logger = logging.getLogger(__name__)

class MemoryManager:
    """An API client for the Pinak Memory Service."""

    def __init__(
        self,
        service_base_url: str = "http://localhost:8001",
        token: Optional[str] = None,
        tenant: Optional[str] = None,
        project_id: Optional[str] = None,
        client: Optional[httpx.Client] = None,
    ):
        """Initializes the client with the URL of the running memory service.

        Args:
            service_base_url: Base URL for the Pinak memory service.
            token: Optional JWT used for authorization. Falls back to the
                ``PINAK_JWT_TOKEN`` environment variable when omitted.
            tenant: Optional tenant identifier. Falls back to
                ``PINAK_TENANT`` when omitted.
            project_id: Optional project identifier. Falls back to
                ``PINAK_PROJECT`` when omitted.
            client: Optional pre-configured :class:`httpx.Client` instance to
                use for requests.
        """
        self.base_url = f"{service_base_url}/api/v1/memory"
        self.client = client or httpx.Client()
        self.token = token or os.getenv("PINAK_JWT_TOKEN")
        self.tenant = tenant or os.getenv("PINAK_TENANT")
        self.project_id = project_id or os.getenv("PINAK_PROJECT")
        logger.debug("MemoryManager client initialized for base URL %s", self.base_url)

    def add_memory(self, content: str, tags: list = None) -> Dict[str, Any]:
        """Sends a request to the Memory Service to add a new memory."""
        try:
            response = self.client.post(
                f"{self.base_url}/add",
                json={"content": content, "tags": tags or []},
                headers=self._build_headers(),
                params=self._build_query_params(),
                timeout=10.0
            )
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            return response.json()
        except httpx.HTTPError as e:
            logger.warning("Error during add_memory request to %s: %s", e.request.url, e)
            return None

    def search_memory(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Sends a request to the Memory Service to search for memories."""
        try:
            response = self.client.get(
                f"{self.base_url}/search",
                params=self._build_query_params({"query": query, "k": k}),
                headers=self._build_headers(),
                timeout=10.0
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.warning("Error during search_memory request to %s: %s", e.request.url, e)
            return []

    def _build_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if self.tenant:
            headers["X-Tenant-ID"] = self.tenant
        if self.project_id:
            headers["X-Project-ID"] = self.project_id
        return headers

    def _build_query_params(self, base: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params = dict(base or {})
        if self.tenant:
            params.setdefault("tenant", self.tenant)
        if self.project_id:
            params.setdefault("project_id", self.project_id)
        return params
