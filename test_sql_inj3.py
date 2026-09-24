from typing import Dict, Any

def get_updates() -> Dict[str, Any]:
    return {"name": "new_name", "tenant = 'hacked', project_id": "val2"}

def update_memory(layer: str, memory_id: str, updates: Dict[str, Any], tenant: str, project_id: str) -> bool:
        # Security: Prevent updating system fields
        forbidden_keys = {"id", "tenant", "project_id", "created_at", "embedding_id"}
        safe_updates = {k: v for k, v in updates.items() if k not in forbidden_keys}
        print("Safe updates:", safe_updates)

update_memory("semantic", "1", get_updates(), "t1", "p1")
