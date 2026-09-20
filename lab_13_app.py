"""Session 13-compatible Streamlit entry point.

The maintained UI is ``app.py``. Running this file executes that same UI, so
the course-style filename and the normal project command cannot drift apart.
"""

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name("app.py")), run_name="__main__")
