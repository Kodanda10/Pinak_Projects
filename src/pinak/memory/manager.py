from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx


class MemoryManagerError(Exception):
    """Raised when the Memory service returns an error."""


class MemoryManager:
    """An API client for interacting with the Pinak Memory Service."""

    def __init__(
        self,
        service_base_url: str = "http://localhost:8001",
        *,
        token: Optional[str] = None,
        tenant_id: Optional[str] = None,
        project_id: Optional[str] = None,
        client: Optional[httpx.Client] = None,
        timeout: float = 10.0,
    ) -> None:
        """Initializes the client with optional authentication context."""

        self.timeout = timeout
        self.client = client or httpx.Client()
        self.service_base_url = service_base_url.rstrip("/")
        self.base_url = f"{self.service_base_url}/api/v1/memory"

        self.token = token
        self.tenant_id = tenant_id
        self.project_id = project_id
        self._apply_headers()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _apply_headers(self) -> None:
        """Apply authentication headers to the HTTP client."""

        # Remove existing headers before applying new ones
        for header in ("Authorization", "X-Pinak-Tenant", "X-Pinak-Project"):
            if header in self.client.headers:
                del self.client.headers[header]

        if self.token:
            self.client.headers["Authorization"] = f"Bearer {self.token}"
        if self.tenant_id:
            self.client.headers["X-Pinak-Tenant"] = self.tenant_id
        if self.project_id:
            self.client.headers["X-Pinak-Project"] = self.project_id

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        url = f"{self.base_url}{path}"
        try:
            response = self.client.request(method, url, timeout=self.timeout, **kwargs)
            response.raise_for_status()
            return response
        except httpx.RequestError as exc:
            raise MemoryManagerError(
                f"Unable to reach {exc.request.url!s}: {exc!s}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            detail = self._extract_error_detail(exc.response)
            raise MemoryManagerError(
                f"Request to {exc.request.url!s} failed with status "
                f"{exc.response.status_code}: {detail}"
            ) from exc

    @staticmethod
    def _extract_error_detail(response: httpx.Response) -> str:
        """Return a readable description of an error response."""

        try:
            payload = response.json()
        except ValueError:
            return response.text or "Unknown error"

        if isinstance(payload, dict):
            detail = payload.get("detail")
            if isinstance(detail, list):
                return "; ".join(str(item) for item in detail)
            if detail:
                return str(detail)
            return ", ".join(f"{key}: {value}" for key, value in payload.items()) or "Unknown error"

        if isinstance(payload, list):
            return "; ".join(str(item) for item in payload) or "Unknown error"

        return str(payload) or "Unknown error"

    def configure(
        self,
        *,
        service_base_url: Optional[str] = None,
        token: Optional[str] = None,
        tenant_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> None:
        """Update runtime configuration for subsequent requests."""

        if service_base_url:
            self.service_base_url = service_base_url.rstrip("/")
            self.base_url = f"{self.service_base_url}/api/v1/memory"
        if token is not None:
            self.token = token
        if tenant_id is not None:
            self.tenant_id = tenant_id
        if project_id is not None:
            self.project_id = project_id
        self._apply_headers()

    # ------------------------------------------------------------------
    # Authentication helpers
    # ------------------------------------------------------------------
    def login(
        self,
        *,
        token: str,
        tenant_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> Dict[str, Optional[str]]:
        """Store authentication context for subsequent requests."""

        self.configure(token=token, tenant_id=tenant_id, project_id=project_id)
        return {
            "token": self.token,
            "tenant_id": self.tenant_id,
            "project_id": self.project_id,
        }

    def add_memory(self, content: str, tags: list = None) -> Dict[str, Any]:
        """Sends a request to the Memory Service to add a new memory."""
        response = self._request(
            "POST",
            "/add",
            json={"content": content, "tags": tags or []},
        )
        return response.json()

    def search_memory(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Sends a request to the Memory Service to search for memories."""
        response = self._request(
            "GET",
            "/search",
            params={"query": query, "k": k},
        )
        return response.json()

    def list_events(
        self,
        *,
        query: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Retrieve events from the memory service."""

        params = {
            "q": query,
            "since": since,
            "until": until,
            "limit": limit,
            "offset": offset,
        }
        params = {key: value for key, value in params.items() if value is not None}
        response = self._request("GET", "/events", params=params)
        return response.json()

    def list_session(
        self,
        session_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
        since: Optional[str] = None,
        until: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve session data for a given session identifier."""

        params = {
            "session_id": session_id,
            "limit": limit,
            "offset": offset,
            "since": since,
            "until": until,
        }
        params = {key: value for key, value in params.items() if value is not None}
        response = self._request("GET", "/session/list", params=params)
        return response.json()
