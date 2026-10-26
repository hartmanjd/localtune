"""Runners: the code that runs a model in one engine and reports how fast it went."""

# Each way of running a model gets its own file here: llamacpp.py for llama.cpp's llama-bench,
# web_api.py for local AI servers like Ollama. They all hand their answers to base.py, which
# watches the GPU, works out energy and cost, and builds the saved record, so every engine's
# runs end up in the same shape.
