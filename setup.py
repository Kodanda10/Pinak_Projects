from pathlib import Path

from setuptools import setup, find_namespace_packages


def _read_requirements(filename: str) -> list[str]:
    requirements_path = Path(filename)
    if not requirements_path.exists():
        return []
    return [
        line.strip()
        for line in requirements_path.read_text().splitlines()
        if line.strip() and not line.startswith('#')
    ]


install_requires = _read_requirements("requirements.txt")

setup(
    name="Pinak_Project",
    version="1.0.0",  # Major version bump for new stable structure
    author="Abhijita/Gemini",
    description="A unified local-first package for AI memory and security auditing.",

    # Define the single source root
    package_dir={"": "src"},

    # Find all packages under the single 'src' directory
    packages=find_namespace_packages(where="src"),

    # Install all dependencies
    install_requires=install_requires,
    extras_require={
        "gemini": ["google-generativeai>=0.5"],
    },

    # Make config files available to the package
    include_package_data=False,  # In a real scenario we'd handle this better

    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
    ],
    entry_points={
        "console_scripts": [
            "pinak=pinak.cli:main",
        ]
    },
)
