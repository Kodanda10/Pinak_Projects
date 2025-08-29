#!/usr/bin/env python3
"""Validate requirement version strings to ensure they conform to PEP 440.
Usage: validate_requirements.py <requirements-file>
Exits with code 1 if any invalid version strings are found.
"""
import sys
from packaging.version import Version, InvalidVersion
import re


def extract_version(req_line: str):
    # Very small parser: handles 'pkg==1.2.3', 'pkg>=1.0', 'pkg[extra]==1.2.3', 'pkg==1.2.3; python_version...'
    line = req_line.split('#', 1)[0].strip()
    if not line or line.startswith(('-e', 'git+', 'http', '--')):
        return None
    # remove environment markers
    if ';' in line:
        line = line.split(';', 1)[0].strip()
    # match package spec with operator
    m = re.search(r"(?:==|~=|!=|<=|>=|<|>)([^\s,;]+)", line)
    if m:
        return m.group(1).strip()
    return None


def main():
    if len(sys.argv) < 2:
        print("usage: validate_requirements.py <requirements-file>")
        return 2
    path = sys.argv[1]
    invalid = []
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            for i, line in enumerate(fh, start=1):
                ver = extract_version(line)
                if ver:
                    try:
                        Version(ver)
                    except InvalidVersion:
                        invalid.append((i, line.strip(), ver))
    except FileNotFoundError:
        print(f"Requirements file not found: {path}")
        return 2

    if invalid:
        print(f"Found {len(invalid)} invalid requirement version(s) in {path}:")
        for ln, raw, ver in invalid:
            print(f"  line {ln}: '{raw}' -> version token '{ver}' is not PEP 440 compliant")
        return 1

    print(f"All requirement versions in {path} appear PEP 440-compliant.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
