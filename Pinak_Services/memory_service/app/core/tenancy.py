from fastapi import Request


def resolve_tenant(request: Request, payload: dict) -> str:
    # Priority: Header -> payload.tenant -> default
    h = request.headers.get("X-Pinak-Tenant")
    if h:
        return h.strip()
    t = payload.get("tenant")
    if isinstance(t, str) and t:
        return t
    return "default"

