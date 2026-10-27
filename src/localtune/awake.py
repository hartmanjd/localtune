"""Keep the computer from going to sleep while localtune is benchmarking."""

# "localtune tune" and "localtune sweep" can run for hours, often overnight, and Windows puts
# an idle computer to sleep after a while. Windows counts only mouse and keyboard use as
# activity, so it would fall asleep in the middle of a benchmark and stop it. While a long
# command runs (cli.py), localtune tells Windows the computer is in use, the way video players
# do, and gives that back the moment the command ends, so sleep works as usual afterwards. The
# screen may still turn off; only sleep is held back. Other systems are left alone.

import sys
from collections.abc import Iterator
from contextlib import contextmanager

ES_CONTINUOUS = 0x80000000  # the request lasts until it is changed again
ES_SYSTEM_REQUIRED = 0x00000001  # the computer is in use, so don't sleep


@contextmanager
def keep_awake() -> Iterator[None]:
    if sys.platform != "win32":
        yield
        return
    import ctypes

    kernel32 = ctypes.windll.kernel32
    kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    try:
        yield
    finally:
        kernel32.SetThreadExecutionState(ES_CONTINUOUS)
