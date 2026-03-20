"""Pytest root conftest — adds the project root to sys.path so that
`vector_os` is importable without installing the package.
"""
import sys
import os

# Insert the repo root so `import vector_os` works in all test files
sys.path.insert(0, os.path.dirname(__file__))
