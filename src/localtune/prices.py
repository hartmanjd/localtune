"""Look up a local electricity price by zip code, state or country."""

# Electricity prices vary a lot from place to place, so a national average can be far off.
# This module finds the closest price localtune knows for where you are, from small tables
# shipped inside the package (built from public data by scripts/build_price_tables.py):
# your zip code first, then your state or country, then the US average. A price you type in
# yourself always wins. Whatever it picks comes back as an ElectricityPrice (cost.py) that
# says where it came from, so every cost can show what it's based on.

import csv
from importlib.resources import files

from localtune.cost import DEFAULT_PRICE, ElectricityPrice

# The tables live in src/localtune/data/:
#   prices_us_zips.csv       zip, state, per_kwh  (NREL utility rates, brought up to date with EIA)
#   prices_us_states.csv     state, per_kwh, period  (EIA monthly state averages)
#   prices_countries.csv     country, per_kwh, currency, period  (Eurostat household prices)

DATA_DIR = files("localtune") / "data"


def read_table(name: str, key: str) -> dict[str, dict[str, str]]:
    with (DATA_DIR / name).open(newline="") as f:
        return {row[key].upper(): row for row in csv.DictReader(f)}


def state_price(state: str) -> ElectricityPrice | None:
    row = read_table("prices_us_states.csv", "state").get(state.upper())
    if row is None:
        return None
    return ElectricityPrice(
        per_kwh=float(row["per_kwh"]),
        source=f"{row['state']} average, EIA {row['period']}",
    )


def zip_price(zip_code: str) -> ElectricityPrice | None:
    row = read_table("prices_us_zips.csv", "zip").get(zip_code.strip().zfill(5))
    if row is None:
        return None
    states = read_table("prices_us_states.csv", "state")
    period = states[row["state"]]["period"] if row["state"] in states else "2023"
    return ElectricityPrice(
        per_kwh=float(row["per_kwh"]),
        source=f"zip {row['zip']} ({row['state']}), NREL utility rates adjusted to EIA {period}",
    )


def country_price(country: str) -> ElectricityPrice | None:
    row = read_table("prices_countries.csv", "country").get(country.upper())
    if row is None:
        return None
    return ElectricityPrice(
        per_kwh=float(row["per_kwh"]),
        currency=row["currency"],
        source=f"{row['country']} average, Eurostat {row['period']}",
    )


# The order to try. Anything given but not found in the tables falls through to the next
# option, and the source says so, rather than stopping with an error.


def find_price(
    zip_code: str | None = None,
    state: str | None = None,
    country: str | None = None,
    your_price: float | None = None,
) -> ElectricityPrice:
    if your_price is not None:
        return ElectricityPrice(per_kwh=your_price, source="your price")

    not_found = []
    for given, lookup in ((zip_code, zip_price), (state, state_price), (country, country_price)):
        if given:
            price = lookup(given)
            if price is not None:
                return price
            not_found.append(given)

    if not_found:
        return DEFAULT_PRICE.model_copy(
            update={"source": f"{DEFAULT_PRICE.source} ({', '.join(not_found)} not found)"}
        )
    return DEFAULT_PRICE
