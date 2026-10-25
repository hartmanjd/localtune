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

## License

[Apache 2.0](LICENSE)
