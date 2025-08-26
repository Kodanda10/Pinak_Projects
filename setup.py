
from setuptools import setup, find_namespace_packages

# Read dependencies from the master requirements.txt file
with open('requirements.txt', 'r') as f:
    install_requires = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="Pinak_Project",
    version="1.0.1", # Patch: auth support & tests
    author="Abhijita/Gemini",
    description="A unified local-first package for AI memory and security auditing.",
    
    # Define the single source root
    package_dir={"": "src"},
    
    # Find all packages under the single 'src' directory
    packages=find_namespace_packages(where="src"),
    
    # Install all dependencies
    install_requires=install_requires,

    # Make config files available to the package
    include_package_data=False, # In a real scenario we'd handle this better

    entry_points={
        'console_scripts': [
            'pinak-memory=pinak.memory.cli:main',
        ]
    },

    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
    ],
)
