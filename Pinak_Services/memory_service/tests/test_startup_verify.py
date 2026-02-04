import os
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient

from app.main import app

def test_skip_verify_on_startup(monkeypatch):
    monkeypatch.setenv("PINAK_SKIP_VERIFY_ON_STARTUP", "1")
    with patch("app.main.get_memory_service") as get_service:
        # svc must support await initialize()
        svc = MagicMock()
        svc.initialize = AsyncMock()
        svc.verify_and_recover = AsyncMock()

        get_service.return_value = svc
        with TestClient(app) as client:
            resp = client.get("/")
            assert resp.status_code == 200

    svc.initialize.assert_called_once()
    svc.verify_and_recover.assert_not_called()
