"""localtune: benchmark local LLMs on your own hardware."""

# The heart of the package. Every module localtune gains later lives in this folder,
# and this file is what runs when anything does "import localtune".
# The version number below is the single source of truth: packaging (pyproject.toml) reads it,
# and saved benchmark results will record it, so each result shows which version measured it.
__version__ = "0.3.7"
