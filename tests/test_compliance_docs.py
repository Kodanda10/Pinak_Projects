"""Tests to validate presence and content of compliance documentation files."""

import os
import pytest
from pathlib import Path


# Path to repository root
REPO_ROOT = Path(__file__).parent.parent


def test_security_md_exists():
    """Test that SECURITY.md file exists."""
    security_file = REPO_ROOT / "SECURITY.md"
    assert security_file.exists(), "SECURITY.md file must exist in repository root"


def test_contributing_md_exists():
    """Test that CONTRIBUTING.md file exists."""
    contributing_file = REPO_ROOT / "CONTRIBUTING.md"
    assert contributing_file.exists(), "CONTRIBUTING.md file must exist in repository root"


def test_license_exists():
    """Test that LICENSE file exists."""
    license_file = REPO_ROOT / "LICENSE"
    assert license_file.exists(), "LICENSE file must exist in repository root"


def test_readme_exists():
    """Test that README.md file exists."""
    readme_file = REPO_ROOT / "README.md"
    assert readme_file.exists(), "README.md file must exist in repository root"


def test_security_md_has_content():
    """Test that SECURITY.md has substantial content."""
    security_file = REPO_ROOT / "SECURITY.md"
    content = security_file.read_text()
    assert len(content) > 500, "SECURITY.md should have substantial content"
    assert "Security Policy" in content or "security" in content.lower()


def test_security_md_has_reporting_section():
    """Test that SECURITY.md includes vulnerability reporting information."""
    security_file = REPO_ROOT / "SECURITY.md"
    content = security_file.read_text()
    # Check for vulnerability reporting keywords
    assert any(keyword in content.lower() for keyword in [
        "reporting", "vulnerability", "report a vulnerability", "disclose"
    ]), "SECURITY.md should include vulnerability reporting information"


def test_security_md_has_contact_info():
    """Test that SECURITY.md includes contact information."""
    security_file = REPO_ROOT / "SECURITY.md"
    content = security_file.read_text()
    # Should have email or contact method
    assert "@" in content or "contact" in content.lower(), \
        "SECURITY.md should include contact information"


def test_contributing_md_has_content():
    """Test that CONTRIBUTING.md has substantial content."""
    contributing_file = REPO_ROOT / "CONTRIBUTING.md"
    content = contributing_file.read_text()
    assert len(content) > 500, "CONTRIBUTING.md should have substantial content"


def test_contributing_md_has_setup_instructions():
    """Test that CONTRIBUTING.md includes setup instructions."""
    contributing_file = REPO_ROOT / "CONTRIBUTING.md"
    content = contributing_file.read_text()
    # Check for setup-related keywords
    assert any(keyword in content.lower() for keyword in [
        "getting started", "setup", "install", "prerequisites"
    ]), "CONTRIBUTING.md should include setup instructions"


def test_contributing_md_has_testing_guidelines():
    """Test that CONTRIBUTING.md includes testing guidelines."""
    contributing_file = REPO_ROOT / "CONTRIBUTING.md"
    content = contributing_file.read_text()
    assert any(keyword in content.lower() for keyword in [
        "test", "testing", "tdd", "coverage"
    ]), "CONTRIBUTING.md should include testing guidelines"


def test_contributing_md_has_pr_process():
    """Test that CONTRIBUTING.md includes PR process."""
    contributing_file = REPO_ROOT / "CONTRIBUTING.md"
    content = contributing_file.read_text()
    assert any(keyword in content.lower() for keyword in [
        "pull request", "pr", "merge", "review"
    ]), "CONTRIBUTING.md should include PR process information"


def test_license_is_mit():
    """Test that LICENSE file is MIT license."""
    license_file = REPO_ROOT / "LICENSE"
    content = license_file.read_text()
    assert "MIT License" in content, "LICENSE should be MIT license"
    assert "Permission is hereby granted" in content, "LICENSE should contain MIT license text"


def test_license_has_copyright():
    """Test that LICENSE file includes copyright information."""
    license_file = REPO_ROOT / "LICENSE"
    content = license_file.read_text()
    assert "Copyright" in content, "LICENSE should include copyright notice"
    assert "2025" in content or "202" in content, "LICENSE should include current year"


def test_readme_references_security():
    """Test that README references SECURITY.md."""
    readme_file = REPO_ROOT / "README.md"
    content = readme_file.read_text()
    assert "SECURITY.md" in content or "security" in content.lower(), \
        "README should reference security documentation"


def test_readme_references_contributing():
    """Test that README references CONTRIBUTING.md."""
    readme_file = REPO_ROOT / "README.md"
    content = readme_file.read_text()
    assert "CONTRIBUTING.md" in content or "contributing" in content.lower(), \
        "README should reference contributing guidelines"


def test_readme_references_license():
    """Test that README references LICENSE."""
    readme_file = REPO_ROOT / "README.md"
    content = readme_file.read_text()
    assert "LICENSE" in content or ("MIT" in content and "License" in content), \
        "README should reference license"


def test_all_files_are_readable():
    """Test that all compliance files are readable."""
    files_to_check = [
        REPO_ROOT / "SECURITY.md",
        REPO_ROOT / "CONTRIBUTING.md",
        REPO_ROOT / "LICENSE",
        REPO_ROOT / "README.md",
    ]
    
    for file_path in files_to_check:
        assert file_path.exists(), f"{file_path.name} should exist"
        assert file_path.is_file(), f"{file_path.name} should be a file"
        # Try to read to ensure it's readable
        content = file_path.read_text()
        assert len(content) > 0, f"{file_path.name} should not be empty"


def test_security_md_structure():
    """Test that SECURITY.md has proper structure."""
    security_file = REPO_ROOT / "SECURITY.md"
    content = security_file.read_text()
    
    # Check for important sections
    sections = [
        "Security",
        "Reporting",
        "Supported",
    ]
    
    found_sections = sum(1 for section in sections if section in content)
    assert found_sections >= 2, "SECURITY.md should have multiple important sections"


def test_contributing_md_structure():
    """Test that CONTRIBUTING.md has proper structure."""
    contributing_file = REPO_ROOT / "CONTRIBUTING.md"
    content = contributing_file.read_text()
    
    # Check for important sections
    sections = [
        "Contributing",
        "Code",
        "Test",
        "Pull Request",
    ]
    
    found_sections = sum(1 for section in sections if section in content)
    assert found_sections >= 2, "CONTRIBUTING.md should have multiple important sections"


def test_files_use_markdown_format():
    """Test that documentation files use markdown format."""
    md_files = [
        REPO_ROOT / "SECURITY.md",
        REPO_ROOT / "CONTRIBUTING.md",
        REPO_ROOT / "README.md",
    ]
    
    for md_file in md_files:
        content = md_file.read_text()
        # Check for markdown indicators
        has_headers = "#" in content
        has_lists = any(marker in content for marker in ["- ", "* ", "1. "])
        assert has_headers or has_lists, f"{md_file.name} should use markdown formatting"
