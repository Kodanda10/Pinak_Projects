
from setuptools import setup, find_packages

setup(
    name="vayu-local-memory",
    version="0.1.0",
    packages=find_packages(),
    description="A local-first memory management system for AI agents.",
    author="Abhijita/Gemini",
    install_requires=[
        "sentence-transformers",
        "faiss-cpu",
        "redis",
        "numpy",
    ],
)
