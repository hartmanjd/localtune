# localtune

Benchmark local LLMs on your own hardware, and find out which model, quantization and KV cache
setting runs best on your machine.

Each run measures:

| Metric | Meaning |
| --- | --- |
| Prefill tok/s | How fast the model reads your prompt |
| Decode tok/s | How fast it writes the answer |
| TTFT ms | Time to first token: the wait before the answer starts |
| Peak VRAM GB | GPU memory used at the busiest moment |
| Avg / peak W | GPU power draw |
| Energy Wh | Electricity the GPU used during the run |
| Cost | That energy times an electricity price |

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

Benchmark a model you have in [Ollama](https://ollama.com) (or a `.gguf` file):

```
localtune run qwen3:14b
```

Find the fastest llama.cpp settings for it on your computer, and get a ready-to-use
`llama-server` command:

```
localtune tune qwen3:14b
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

## Data sources

Electricity cost uses the most local price localtune has: your zip code, then your state or
country, then the US average. A price per kWh that you give yourself always replaces these.
Each cost records which price it was based on.

| Price | Source | Covers | License |
| --- | --- | --- | --- |
| US zip codes | [NREL, U.S. Electric Utility Companies and Rates (2023)](https://catalog.data.gov/dataset/u-s-electric-utility-companies-and-rates-look-up-by-zipcode-2023), scaled by each state's change since 2023 (EIA) | Zip codes where one company both sells and delivers power | CC BY 4.0 |
| US states | [EIA-861M monthly sales and revenue by state](https://www.eia.gov/electricity/data/state/), last 12 months | All 50 states and DC | Public domain |
| Countries | [Eurostat, electricity prices for household consumers (nrg_pc_204)](https://ec.europa.eu/eurostat/databrowser/view/nrg_pc_204/default/table), all taxes included, in euros | 41 European countries | CC BY 4.0 |
| US average | [EIA Short-Term Energy Outlook](https://www.eia.gov/outlooks/steo/report/elec_coal_renew.php), October 2026 | Whole US | Public domain |

The tables in `src/localtune/data/` are built from these sources by
`scripts/build_price_tables.py`.

The KV cache quality check reads the test set of [WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext)
(Wikipedia articles, Merity et al., 2016, CC BY-SA 3.0), the same text llama.cpp's own quality
comparisons use. It isn't included in localtune: it is downloaded the first time it's needed.

## License

[Apache 2.0](LICENSE)
