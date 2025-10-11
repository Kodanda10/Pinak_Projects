"""Tests for role-based authorization."""

import pytest
from fastapi import HTTPException

from app.core.authorization import (
    Role,
    Permission,
    ROLE_PERMISSIONS,
    get_permissions_for_roles,
    check_permission,
    require_permission,
)


def test_role_enum_values():
    """Test that all roles are defined."""
    assert Role.ADMIN == "admin"
    assert Role.USER == "user"
    assert Role.GUEST == "guest"
    assert Role.SERVICE == "service"


def test_permission_enum_values():
    """Test that all permissions are defined."""
    assert Permission.READ_MEMORY == "read:memory"
    assert Permission.WRITE_MEMORY == "write:memory"
    assert Permission.DELETE_MEMORY == "delete:memory"
    assert Permission.READ_EVENTS == "read:events"
    assert Permission.WRITE_EVENTS == "write:events"
    assert Permission.READ_AUDIT == "read:audit"
    assert Permission.ADMIN_ALL == "admin:all"


def test_admin_has_all_permissions():
    """Test that admin role has all permissions."""
    admin_permissions = ROLE_PERMISSIONS[Role.ADMIN]
    assert Permission.READ_MEMORY in admin_permissions
    assert Permission.WRITE_MEMORY in admin_permissions
    assert Permission.DELETE_MEMORY in admin_permissions
    assert Permission.READ_EVENTS in admin_permissions
    assert Permission.WRITE_EVENTS in admin_permissions
    assert Permission.READ_AUDIT in admin_permissions
    assert Permission.ADMIN_ALL in admin_permissions


def test_user_has_read_write_permissions():
    """Test that user role has read and write permissions."""
    user_permissions = ROLE_PERMISSIONS[Role.USER]
    assert Permission.READ_MEMORY in user_permissions
    assert Permission.WRITE_MEMORY in user_permissions
    assert Permission.READ_EVENTS in user_permissions
    assert Permission.WRITE_EVENTS in user_permissions
    # Users should not have delete or admin permissions
    assert Permission.DELETE_MEMORY not in user_permissions
    assert Permission.READ_AUDIT not in user_permissions
    assert Permission.ADMIN_ALL not in user_permissions


def test_guest_has_only_read_permissions():
    """Test that guest role has only read permissions."""
    guest_permissions = ROLE_PERMISSIONS[Role.GUEST]
    assert Permission.READ_MEMORY in guest_permissions
    assert Permission.READ_EVENTS in guest_permissions
    # Guests should not have write, delete, or admin permissions
    assert Permission.WRITE_MEMORY not in guest_permissions
    assert Permission.DELETE_MEMORY not in guest_permissions
    assert Permission.WRITE_EVENTS not in guest_permissions
    assert Permission.READ_AUDIT not in guest_permissions
    assert Permission.ADMIN_ALL not in guest_permissions


def test_service_has_service_permissions():
    """Test that service role has appropriate permissions."""
    service_permissions = ROLE_PERMISSIONS[Role.SERVICE]
    assert Permission.READ_MEMORY in service_permissions
    assert Permission.WRITE_MEMORY in service_permissions
    assert Permission.READ_EVENTS in service_permissions
    assert Permission.WRITE_EVENTS in service_permissions
    # Service should not have delete or admin permissions
    assert Permission.DELETE_MEMORY not in service_permissions
    assert Permission.READ_AUDIT not in service_permissions
    assert Permission.ADMIN_ALL not in service_permissions


def test_get_permissions_for_single_role():
    """Test getting permissions for a single role."""
    permissions = get_permissions_for_roles(["admin"])
    assert Permission.ADMIN_ALL in permissions
    assert len(permissions) == len(ROLE_PERMISSIONS[Role.ADMIN])


def test_get_permissions_for_multiple_roles():
    """Test getting permissions for multiple roles (union)."""
    permissions = get_permissions_for_roles(["user", "guest"])
    # Should have all user permissions (which includes guest permissions)
    assert Permission.READ_MEMORY in permissions
    assert Permission.WRITE_MEMORY in permissions
    assert Permission.READ_EVENTS in permissions
    assert Permission.WRITE_EVENTS in permissions


def test_get_permissions_for_unknown_role():
    """Test that unknown roles are ignored."""
    permissions = get_permissions_for_roles(["unknown_role", "user"])
    # Should only have user permissions
    assert Permission.READ_MEMORY in permissions
    assert Permission.WRITE_MEMORY in permissions
    # Should not fail, just ignore unknown role
    assert Permission.ADMIN_ALL not in permissions


def test_get_permissions_for_empty_roles():
    """Test getting permissions for empty roles list."""
    permissions = get_permissions_for_roles([])
    assert len(permissions) == 0


def test_check_permission_admin_can_read():
    """Test that admin can read memory."""
    assert check_permission(["admin"], Permission.READ_MEMORY) is True


def test_check_permission_admin_can_write():
    """Test that admin can write memory."""
    assert check_permission(["admin"], Permission.WRITE_MEMORY) is True


def test_check_permission_admin_can_delete():
    """Test that admin can delete memory."""
    assert check_permission(["admin"], Permission.DELETE_MEMORY) is True


def test_check_permission_user_can_read():
    """Test that user can read memory."""
    assert check_permission(["user"], Permission.READ_MEMORY) is True


def test_check_permission_user_can_write():
    """Test that user can write memory."""
    assert check_permission(["user"], Permission.WRITE_MEMORY) is True


def test_check_permission_user_cannot_delete():
    """Test that user cannot delete memory."""
    assert check_permission(["user"], Permission.DELETE_MEMORY) is False


def test_check_permission_user_cannot_read_audit():
    """Test that user cannot read audit logs."""
    assert check_permission(["user"], Permission.READ_AUDIT) is False


def test_check_permission_guest_can_read():
    """Test that guest can read memory."""
    assert check_permission(["guest"], Permission.READ_MEMORY) is True


def test_check_permission_guest_cannot_write():
    """Test that guest cannot write memory."""
    assert check_permission(["guest"], Permission.WRITE_MEMORY) is False


def test_check_permission_guest_cannot_delete():
    """Test that guest cannot delete memory."""
    assert check_permission(["guest"], Permission.DELETE_MEMORY) is False


def test_check_permission_with_multiple_roles():
    """Test permission check with multiple roles."""
    # User + guest should still only have user permissions
    assert check_permission(["user", "guest"], Permission.WRITE_MEMORY) is True
    assert check_permission(["user", "guest"], Permission.DELETE_MEMORY) is False


def test_check_permission_admin_all_grants_any_permission():
    """Test that ADMIN_ALL permission grants any permission."""
    # Admin should be able to do anything because of ADMIN_ALL
    assert check_permission(["admin"], Permission.READ_MEMORY) is True
    assert check_permission(["admin"], Permission.WRITE_MEMORY) is True
    assert check_permission(["admin"], Permission.DELETE_MEMORY) is True
    assert check_permission(["admin"], Permission.READ_AUDIT) is True


def test_require_permission_decorator_allows_authorized():
    """Test that require_permission allows authorized users."""
    checker = require_permission(Permission.READ_MEMORY)
    # Should not raise exception
    try:
        checker(["user"])
    except HTTPException:
        pytest.fail("Should not raise exception for authorized user")


def test_require_permission_decorator_denies_unauthorized():
    """Test that require_permission denies unauthorized users."""
    checker = require_permission(Permission.DELETE_MEMORY)
    # Should raise 403 for user role
    with pytest.raises(HTTPException) as exc_info:
        checker(["user"])
    assert exc_info.value.status_code == 403
    assert "Missing required permission" in exc_info.value.detail


def test_require_permission_decorator_allows_admin_all():
    """Test that ADMIN_ALL permission allows any operation."""
    checker = require_permission(Permission.DELETE_MEMORY)
    # Should not raise exception for admin
    try:
        checker(["admin"])
    except HTTPException:
        pytest.fail("Should not raise exception for admin")


def test_require_permission_guest_cannot_write():
    """Test that guest cannot write memory."""
    checker = require_permission(Permission.WRITE_MEMORY)
    with pytest.raises(HTTPException) as exc_info:
        checker(["guest"])
    assert exc_info.value.status_code == 403


def test_case_insensitive_role_matching():
    """Test that role matching is case-insensitive."""
    assert check_permission(["ADMIN"], Permission.READ_MEMORY) is True
    assert check_permission(["User"], Permission.WRITE_MEMORY) is True
    assert check_permission(["GUest"], Permission.READ_MEMORY) is True
