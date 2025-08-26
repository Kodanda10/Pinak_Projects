from typing import Optional


def ttl_for_layer(layer: str, default_ttl: Optional[int] = None) -> Optional[int]:
    # Simple defaults; future: read from config or policy service
    table = {
        "working": 300,
        "session": 172800,
        "event": 259200,
        "changelog": 0,
        "episodic": 0,
        "semantic": 0,
        "procedural": 0,
        "rag": 1800,
    }
    if default_ttl is not None:
        return default_ttl
    return table.get(layer, 0)

