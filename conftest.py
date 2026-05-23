# conftest.py – ensure the project root is on sys.path for pytest
import os, sys
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
