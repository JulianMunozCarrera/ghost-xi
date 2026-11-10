import json
import os
import shutil
import subprocess

import pytest

HERE = os.path.dirname(__file__)


def test_website_engine_matches_python():
    """The JavaScript in the website must give the same ghosts and danger numbers as Python."""
    if shutil.which("node") is None:
        pytest.skip("Node.js is not installed")
    out = subprocess.run(["node", os.path.join(HERE, "web_engine_check.js")], capture_output=True, text=True, check=True)
    result = json.loads(out.stdout)
    # Biggest difference over every ghost position (meters) and every danger number.
    assert result["worst_difference"] < 1e-3


def test_built_page_contains_no_placeholders():
    page = open(os.path.join(HERE, "..", "docs", "index.html")).read()
    assert "/*__" not in page
    assert "GhostEngine" in page and "GHOST_DATA" in page
