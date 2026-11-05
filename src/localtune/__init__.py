"""localtune: benchmark local LLMs on your own hardware."""

# The heart of the package. Every module localtune gains later lives in this folder,
# and this file is what runs when anything does "import localtune".
# The version number comes from the release tag the package was built from (pyproject.toml)
# and is read back from the installed package, and saved benchmark results record it, so each
# result shows which version measured it.
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("localtune")
except PackageNotFoundError:
    __version__ = "0.0.0"
