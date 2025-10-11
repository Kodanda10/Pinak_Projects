"""Role-based authorization module for enforcing access controls."""

from enum import Enum
from typing import List, Optional
from fastapi import HTTPException, status


class Role(str, Enum):
    """Predefined roles for the memory service."""
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"
    SERVICE = "service"


class Permission(str, Enum):
    """Granular permissions for operations."""
    READ_MEMORY = "read:memory"
    WRITE_MEMORY = "write:memory"
    DELETE_MEMORY = "delete:memory"
    READ_EVENTS = "read:events"
    WRITE_EVENTS = "write:events"
    READ_AUDIT = "read:audit"
    ADMIN_ALL = "admin:all"


# Role-to-permission mapping
ROLE_PERMISSIONS = {
    Role.ADMIN: [
        Permission.READ_MEMORY,
        Permission.WRITE_MEMORY,
        Permission.DELETE_MEMORY,
        Permission.READ_EVENTS,
        Permission.WRITE_EVENTS,
        Permission.READ_AUDIT,
        Permission.ADMIN_ALL,
    ],
    Role.USER: [
        Permission.READ_MEMORY,
        Permission.WRITE_MEMORY,
        Permission.READ_EVENTS,
        Permission.WRITE_EVENTS,
    ],
    Role.GUEST: [
        Permission.READ_MEMORY,
        Permission.READ_EVENTS,
    ],
    Role.SERVICE: [
        Permission.READ_MEMORY,
        Permission.WRITE_MEMORY,
        Permission.READ_EVENTS,
        Permission.WRITE_EVENTS,
    ],
}


def get_permissions_for_roles(roles: List[str]) -> List[Permission]:
    """Get all permissions for given roles."""
    permissions = set()
    for role_str in roles:
        try:
            role = Role(role_str.lower())
            permissions.update(ROLE_PERMISSIONS.get(role, []))
        except ValueError:
            # Unknown role, skip
            continue
    return list(permissions)


def require_permission(required_permission: Permission) -> None:
    """Dependency to check if user has required permission."""
    def permission_checker(user_roles: List[str]) -> None:
        permissions = get_permissions_for_roles(user_roles)
        if required_permission not in permissions and Permission.ADMIN_ALL not in permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required permission: {required_permission.value}"
            )
    return permission_checker


def check_permission(roles: List[str], required_permission: Permission) -> bool:
    """Check if roles have required permission."""
    permissions = get_permissions_for_roles(roles)
    return required_permission in permissions or Permission.ADMIN_ALL in permissions
