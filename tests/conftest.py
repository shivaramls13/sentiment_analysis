# tests/conftest.py
import sys
from pathlib import Path

# Get the absolute path of the project's root directory
project_root = Path(__file__).parent.parent

# Add the 'src' directory to the Python path
src_path = project_root / "src"
sys.path.insert(0, str(src_path))
