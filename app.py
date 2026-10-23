"""Repository entry point for the SceneSift Streamlit demo."""

from pathlib import Path
import runpy
import sys

app_dir = Path(__file__).parent / "scene_sift"
sys.path.insert(0, str(app_dir))

runpy.run_path(str(app_dir / "app.py"), run_name="__main__")
