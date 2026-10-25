# Tests are small programs that check the real code behaves as expected. pytest finds every
# file named test_*.py in this folder and runs it, here and on GitHub for every change.
# Each module gets its own test file. This first one is a smoke test: if it passes, the
# package installs and imports correctly, which every later test depends on.
import re

import localtune


def test_version_looks_like_a_version_number():
    assert re.fullmatch(r"\d+\.\d+\.\d+", localtune.__version__)
