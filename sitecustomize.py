import os, sys, pathlib

# Ensure the project root (directory containing this file) is on sys.path
project_root = pathlib.Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
