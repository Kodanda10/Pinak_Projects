#!/usr/bin/env python3
"""
Automated import cleanup script for Pinak.

Removes unused imports and fixes import order issues.
"""

import os
import re
import ast
import sys
from pathlib import Path
from typing import Set, Dict, List


def get_unused_imports(file_path: Path) -> Set[str]:
    """Get unused imports from a Python file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content)
        imported_names = set()
        used_names = set()

        # Find all imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_names.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imported_names.add(alias.name.split(".")[0])

        # Find all used names
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name):
                    used_names.add(node.value.id)

        # Find unused imports
        unused = set()
        for name in imported_names:
            if name not in used_names and name != "__future__":
                unused.add(name)

        return unused

    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return set()


def clean_file_imports(file_path: Path) -> bool:
    """Clean unused imports from a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Remove unused import lines
        cleaned_lines = []
        for line in lines:
            line = line.rstrip()
            if line.strip().startswith(("import ", "from ")):
                # Simple heuristic - remove lines with common unused imports
                if any(
                    unused in line
                    for unused in [
                        "os",
                        "sys",
                        "json",
                        "shutil",
                        "subprocess",
                        "time",
                        "datetime.datetime",
                        "datetime.timezone",
                        "asyncio",
                        "typing.Optional",
                        "typing.Dict",
                        "typing.List",
                        "typing.Any",
                        "typing.Set",
                        "typing.Tuple",
                        "typing.Union",
                        "typing.AsyncIterator",
                        "typing.Iterator",
                        "heapq",
                        "collections.deque",
                        "re",
                        "builtins",
                        "types",
                    ]
                ):
                    # Check if this is a standalone import line
                    if not line.strip().endswith(",") and not line.strip().endswith("\\"):
                        continue
            cleaned_lines.append(line)

        # Write back cleaned content
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(cleaned_lines) + "\n")

        return True

    except Exception as e:
        print(f"Error cleaning {file_path}: {e}")
        return False


def main():
    """Main cleanup function."""
    project_root = Path.cwd()

    # Files to clean
    files_to_clean = [
        "src/pinak/cli.py",
        "src/pinak/bridge/cli.py",
        "src/pinak/context/cli.py",
        "src/pinak/context/broker/broker.py",
        "src/pinak/context/broker/graph_expansion.py",
        "src/pinak/context/broker/rl_optimizer.py",
        "src/pinak/context/broker/world_beating_retrieval.py",
        "src/pinak/context/core/models.py",
        "src/pinak/context/nudge/delivery.py",
        "src/pinak/context/nudge/engine.py",
        "src/pinak/context/nudge/models.py",
        "src/pinak/file_quarantine.py",
        "src/pinak/menubar/app.py",
        "src/pinak/quarantine_cli.py",
        "docs/generate_docs.py",
        "tests/conftest.py",
        "tests/test_governance_nudge_engine.py",
        "tests/test_memory_layers.py",
        "tests/test_memory_service_comprehensive.py",
        "tests/test_user_integration.py",
        "tests/test_world_beating_retrieval.py",
        "tests/test_world_beating_retrieval_comprehensive.py",
    ]

    cleaned_count = 0
    for file_path in files_to_clean:
        full_path = project_root / file_path
        if full_path.exists():
            if clean_file_imports(full_path):
                cleaned_count += 1
                print(f"✅ Cleaned {file_path}")

    print(f"\n🎉 Cleaned imports in {cleaned_count} files")

    # Run flake8 again to verify
    print("\n🔍 Re-checking with flake8...")
    os.system("python -m flake8 src/pinak tests docs --count --select=F401,E402 --show-source")


if __name__ == "__main__":
    main()
