"""Turn the energy a run used into money."""

# Running a model locally costs electricity. gpu_monitor.py measures the energy a run used,
# which is the same wherever you live; this module multiplies it by an electricity price to get
# what it cost you. Energy and price are kept separate on purpose: saved results keep the energy,
# and the cost can be worked out again for any price, which is how local runs will be compared
# with paid cloud APIs.

from pydantic import BaseModel

# A price always travels with where it came from, so every cost can say what it is based on.


class ElectricityPrice(BaseModel):
    per_kwh: float
    currency: str = "USD"
    source: str


# Used when nobody has given a price: the US average residential price from the US Energy
# Information Administration's Short-Term Energy Outlook (October 2026, forecast for 2026):
# https://www.eia.gov/outlooks/steo/report/elec_coal_renew.php
# EIA data is published by the US government and is free to use.
DEFAULT_PRICE = ElectricityPrice(per_kwh=0.182, source="US average, EIA 2026")


def energy_cost(energy_wh: float, price: ElectricityPrice) -> float:
    return energy_wh / 1000 * price.per_kwh


def hourly_cost(power_w: float, price: ElectricityPrice) -> float:
    return energy_cost(power_w, price)
