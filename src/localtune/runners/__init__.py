"""Runners: the code that runs a model in one engine and reports how fast it went."""

# Each engine (llama.cpp now, local AI servers later) gets its own file here. They all hand
# their answers to base.py, which watches the GPU, works out energy and cost, and builds the
# saved record, so every engine's runs end up in the same shape.
