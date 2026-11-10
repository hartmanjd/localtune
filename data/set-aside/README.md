# Set-aside runs

Runs kept for the record but left out of training and of every number on the data page.
scripts/combine_runs.py only reads data/runs, so a machine's folder moved here is skipped.

- **rtxa2000**: the test on this rented RTX A2000 went wrong; only one of its 31 runs was usable.
- **teslap100x2**: 29 runs shared from a server with two Tesla P100 cards, made with localtune
  0.2.2. The model was spread over both cards, but that version only watched the first one, so
  the memory, power and energy in these runs are one card's share.
