"""Build localtune's electricity price tables from public data.

Run from the project folder:
    uv run python scripts/build_price_tables.py
"""

# localtune ships three small price tables (src/localtune/data/) so that nobody using it needs
# an internet connection or an API key to get a local electricity price. This script makes those
# tables from the original public data, and is re-run every few months to bring them up to date.
# Only people working on localtune run it; users only ever read the finished tables.
#
# Steps: download the raw files (or reuse ones already in raw-price-data/), boil each down to a
# price per kWh per place, and write the three tables.

import argparse
import csv
import urllib.request
from collections import defaultdict
from pathlib import Path

import openpyxl

SOURCES = {
    # NREL: each US zip code's electric utilities and their average rates (2023), CC BY 4.0.
    "iou_zipcodes_2023.csv": "https://data.openei.org/files/6225/iou_zipcodes_2023.csv",
    "non_iou_zipcodes_2023.csv": "https://data.openei.org/files/6225/non_iou_zipcodes_2023.csv",
    # EIA: monthly revenue and electricity sold, by state and sector (2010 on), public domain.
    "HS861M 2010-.xlsx": "https://www.eia.gov/electricity/data/state/xls/861m/HS861M%202010-.xlsx",
    # Eurostat: household electricity prices twice a year by country, CC BY 4.0.
    "estat_nrg_pc_204_en.csv": "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/nrg_pc_204/?format=SDMX-CSV",
}

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "raw-price-data"
OUT_DIR = ROOT / "src" / "localtune" / "data"
NREL_YEAR = 2023


def download(raw_dir: Path) -> None:
    raw_dir.mkdir(exist_ok=True)
    for name, url in SOURCES.items():
        if not (raw_dir / name).exists():
            print(f"Downloading {name}")
            urllib.request.urlretrieve(url, raw_dir / name)


# US states (EIA). Home prices swing with the seasons, so each state's price is the last 12
# months of revenue divided by the last 12 months of electricity sold. The same sum for NREL's
# year is kept too, to bring the older zip code rates up to date below.


def state_prices(xlsx: Path) -> tuple[dict[str, float], dict[str, float], str]:
    sheet = openpyxl.load_workbook(xlsx, read_only=True)["Monthly-States"]
    revenue: dict[tuple[str, int, int], float] = {}
    sales: dict[tuple[str, int, int], float] = {}
    for row in sheet.iter_rows(min_row=4, values_only=True):
        year, month, state = row[0], row[1], row[2]
        if not isinstance(year, int) or not state:
            continue
        revenue[(state, year, month)] = row[4] or 0
        sales[(state, year, month)] = row[5] or 0

    latest = max((year, month) for _, year, month in revenue)
    last_12 = set()
    year, month = latest
    for _ in range(12):
        last_12.add((year, month))
        year, month = (year, month - 1) if month > 1 else (year - 1, 12)

    now, then = defaultdict(lambda: [0.0, 0.0]), defaultdict(lambda: [0.0, 0.0])
    for (state, year, month), dollars in revenue.items():
        mwh = sales[(state, year, month)]
        if (year, month) in last_12:
            now[state][0] += dollars
            now[state][1] += mwh
        if year == NREL_YEAR:
            then[state][0] += dollars
            then[state][1] += mwh

    def per_kwh(totals: dict[str, list[float]]) -> dict[str, float]:
        # Thousand dollars / megawatt-hours = dollars per kWh.
        return {state: d / mwh for state, (d, mwh) in totals.items() if mwh}

    return per_kwh(now), per_kwh(then), f"{latest[0]}-{latest[1]:02d}"


# US zip codes (NREL). A zip code served by several utilities gets the average of their home
# rates. Only zip codes where every utility both sells and delivers the power ("bundled") are
# kept. Where people shop for their power separately (much of Texas, Ohio and the Northeast),
# NREL only has the delivery part of the bill, so those zip codes fall back to their state
# average instead of looking far too cheap. Each rate is then scaled by how much its state's
# average has changed since NREL's year.


def zip_prices(
    csv_files: list[Path], now: dict[str, float], then: dict[str, float]
) -> list[tuple[str, str, float]]:
    rates: dict[str, list[float]] = defaultdict(list)
    states: dict[str, str] = {}
    not_bundled: set[str] = set()
    for path in csv_files:
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                zip_code = row["zip"].zfill(5)
                rate = float(row["res_rate"] or 0)
                if row["service_type"] != "Bundled":
                    not_bundled.add(zip_code)
                elif rate > 0:
                    rates[zip_code].append(rate)
                    states.setdefault(zip_code, row["state"])

    table = []
    for zip_code, values in sorted(rates.items()):
        if zip_code in not_bundled:
            continue
        state = states[zip_code]
        change = now[state] / then[state] if state in now and state in then else 1.0
        table.append((zip_code, state, sum(values) / len(values) * change))
    return table


# Countries (Eurostat). The price for a typical household (2,500-5,000 kWh a year), with all
# taxes, in euros, from the latest half-year each country has reported. Countries that stopped
# reporting more than a year before the newest data (the UK after 2020) are left out.


def country_prices(csv_file: Path) -> list[tuple[str, float, str]]:
    latest: dict[str, tuple[str, float]] = {}
    with csv_file.open(newline="") as f:
        for row in csv.DictReader(f):
            if (
                row["nrg_cons"] == "KWH2500-4999"
                and row["tax"] == "I_TAX"
                and row["currency"] == "EUR"
                and row["unit"] == "KWH"
                and row["OBS_VALUE"]
            ):
                geo, period = row["geo"], row["TIME_PERIOD"]
                if geo not in latest or period > latest[geo][0]:
                    latest[geo] = (period, float(row["OBS_VALUE"]))
    newest_year = int(max(period for period, _ in latest.values())[:4])
    return [
        (geo, value, period)
        for geo, (period, value) in sorted(latest.items())
        if int(period[:4]) >= newest_year - 1
    ]


def write(name: str, header: list[str], rows: list) -> None:
    with (OUT_DIR / name).open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Wrote {name}: {len(rows)} rows")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="Where the raw files go.")
    raw_dir = parser.parse_args().raw_dir

    download(raw_dir)
    now, then, period = state_prices(raw_dir / "HS861M 2010-.xlsx")
    zips = zip_prices(
        [raw_dir / "iou_zipcodes_2023.csv", raw_dir / "non_iou_zipcodes_2023.csv"], now, then
    )
    countries = country_prices(raw_dir / "estat_nrg_pc_204_en.csv")

    OUT_DIR.mkdir(exist_ok=True)
    write(
        "prices_us_states.csv",
        ["state", "per_kwh", "period"],
        [
            (state, round(price, 4), f"12 months to {period}")
            for state, price in sorted(now.items())
        ],
    )
    write(
        "prices_us_zips.csv", ["zip", "state", "per_kwh"], [(z, s, round(p, 4)) for z, s, p in zips]
    )
    write(
        "prices_countries.csv",
        ["country", "per_kwh", "currency", "period"],
        [(geo, round(value, 4), "EUR", p) for geo, value, p in countries],
    )


if __name__ == "__main__":
    main()
