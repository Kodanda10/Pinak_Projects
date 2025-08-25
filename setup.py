
from setuptools import setup, find_namespace_packages
import os

# Function to read requirements from a file
def read_requirements(path):
    if not os.path.exists(path):
        return []
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read dependencies from both projects
memory_reqs = read_requirements('memory-baseline/requirements.txt')
security_reqs = read_requirements('security-baseline/requirements.txt')
all_reqs = list(set(memory_reqs + security_reqs))

setup(
    name="Pinak_Project",
    version="0.2.0", # Version bump for the new structure
    author="Abhijita/Gemini",
    description="A unified local-first package for AI memory and security auditing.",
    
    # Find all packages in the 'src' directories of the sub-projects
    package_dir={
        'pinak.memory': 'memory-baseline/src/pinak/memory',
        'pinak.security': 'security-baseline/src/pinak/security',
    },
    packages=['pinak.memory', 'pinak.security'],
    
    # Install all dependencies
    install_requires=all_reqs,
    
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
    ],
)
