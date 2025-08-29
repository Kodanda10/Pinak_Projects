
from setuptools import setup, find_namespace_packages
import sys

# Read dependencies from the master requirements.txt file
with open('requirements.txt', 'r') as f:
    install_requires = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Add macOS-specific dependencies
if sys.platform == 'darwin':  # macOS
    install_requires.append('pyobjc')

setup(
    name="Pinak",
    version="1.1.1",
    author="Abhijita/Gemini",
    description="A unified local-first package for AI memory and security auditing.",
    
    # Define the single source root
    package_dir={"": "src"},
    
    # Find all packages under the single 'src' directory
    packages=find_namespace_packages(where="src"),
    
    # Install all dependencies
    install_requires=install_requires,

    # Make config files available to the package
    include_package_data=False,

    entry_points={
        'console_scripts': [
            'pinak=pinak.cli:main',
            'pinak-bridge=pinak.bridge.cli:main',
            'pinak-memory=pinak.memory.cli:main',
            'pinak-menubar=pinak.menubar.app:main',
        ]
    },

    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
    ],
)
