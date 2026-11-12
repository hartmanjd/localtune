# localtune

Benchmark local LLMs on your own hardware, and find out which model, quantization and KV cache
setting runs best on your machine.

Each run measures:

| Metric | Meaning |
| --- | --- |
| Prefill tok/s | How fast the model reads your prompt |
| Decode tok/s | How fast it writes the answer |
| TTFT ms | Time to first token: the wait before the answer starts |
| Peak VRAM GB | GPU memory used at the busiest moment (added up when a model is spread over several GPUs) |
| Avg / peak W | GPU power draw |
| Energy Wh | Electricity the GPU used during the run |

## Install

localtune needs Python 3.10 or newer and, for benchmarks, an NVIDIA graphics card. It runs on
Windows and Linux.

```
pip install localtune
```

or, with [uv](https://docs.astral.sh/uv/):

```
uv tool install localtune
```

## Quick start

Check that your computer is ready. localtune offers to install llama.cpp if it's missing:

```
localtune doctor
```

See whether a model fits on your GPU and how fast it should write, before downloading it:

```
localtune predict qwen3:14b
```

`predict` also says the biggest context (the longest conversation) that keeps the whole model on
the GPU. That number is worked out from sizes written in the model's file. To prove it on your
computer, with whatever else is using the GPU, load the model at that size and read the GPU's
memory:

```
localtune context qwen3:14b
```

See whether speculative decoding makes a model faster on your computer. A small "draft" model of
the same family guesses a few tokens ahead and the big model checks them all at once, so the
answer is the same and arrives sooner when the guesses are right. With only a model name,
`speculate` picks small models you already have as drafts, adds llama.cpp's lookup type that
needs no second model, and measures each on an answer that copies from the prompt and one
written fresh. Each row shows the writing speed, the speed-up over no speculation, and the share
of guesses the model kept. It takes a few minutes; `--thorough` also tries a 16K-token prompt
and a temperature of 0.8, with three answers each, and takes about half an hour:

```
localtune speculate qwen3:14b
```

Every list option takes several values, and every combination runs. `--llama-args` passes
anything else on to llama-server as it is, and `--prompt-file` measures your own prompts:

```
localtune speculate qwen3:14b --draft qwen3:0.6b --draft qwen3:1.7b --n-max 2,4,8,16 --p-min 0,0.5,0.8 --repeats 5
```

The rows are saved to `speculate.jsonl` in localtune's folder, one JSON object per line, apart
from the benchmark runs.

See the models and GPUs `predict` knows (the 100 most downloaded models on Ollama, and NVIDIA's
RTX 20 to 50 cards plus common data center GPUs). Add a word to show only matching names:

```
localtune models qwen3
```

```
localtune gpus 4090
```

Benchmark any model from [Ollama's library](https://ollama.com/library), or a `.gguf` file.
A model you don't have yet is downloaded first; Ollama itself doesn't need to be installed:

```
localtune measure qwen3:14b
```

Test it the way you'll use it with `--use`. Each name sets fixed prompt and answer lengths, the
same for every model:

| `--use` | Prompt the model reads | Answer it writes |
|---|---|---|
| `chat` | 512 tokens | 256 tokens |
| `code` | 8,000 tokens | 512 tokens |
| `documents` | 16,000 tokens | 256 tokens |

A number like `--use 30` instead makes the conversation (prompt and answer together) fill 30% of
the biggest context the model can have on your computer, so small and big models get a similar
load. The answer is 256 tokens and the prompt is the rest:

```
localtune measure qwen3:14b --use 30
```

Find the fastest llama.cpp settings for it on your computer:

```
localtune tune qwen3:14b
```

For a model downloaded with Ollama, with Ollama running, tune then runs the same chat test in
Ollama and in `llama-server` at the tuned settings, one after the other, and shows both. To
repeat that comparison later without tuning again:

```
localtune compare qwen3:14b
```

Benchmark every model you have in Ollama, one after another:

```
localtune sweep
```

See your saved runs, or save them all to a CSV file:

```
localtune results
localtune results --csv runs.csv
```

Run `localtune` on its own to list every command, and add `--help` to any command for its
options. Runs are saved in `~/.localtune` (set `LOCALTUNE_HOME` to use another folder) and
stay on your computer.

## Your own llama.cpp or AI server

localtune looks for llama.cpp's programs (`llama-bench`, `llama-server`) on your PATH, then in
the copy it installs itself. If you built llama.cpp yourself, `localtune doctor` looks for the
build in the usual places (like `~/llama.cpp/build/bin`) before offering to install another
copy, and asks for its folder if it can't find it. The folder is remembered for every command.

To choose the folder yourself instead, point localtune at the folder that holds those programs.

Linux, macOS or WSL:

```
export LOCALTUNE_LLAMA_DIR=~/llama.cpp/build/bin
```

Windows (PowerShell):

```
$env:LOCALTUNE_LLAMA_DIR = "C:\llama.cpp\build\bin\Release"
```

That folder is used before anything else, by `doctor`, `measure` and `tune` alike, and before a
folder doctor saved. Add the line to your shell's startup file (such as `~/.bashrc`) to keep it
set in new terminals.

`localtune doctor` looks for a running AI server (Ollama, llama-server, LM Studio, vLLM) on
its usual port on this computer. A server on another port, or on another computer, works too:
give its address with `--url`.

```
localtune measure -b api --url http://127.0.0.1:8081
```

## Data sources

The KV cache quality check reads the test set of [WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext)
(Wikipedia articles, Merity et al., 2016, CC BY-SA 3.0), the same text llama.cpp's own quality
comparisons use. It isn't included in localtune: it is downloaded the first time it's needed.

## License

[Apache 2.0](LICENSE)
