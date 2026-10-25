# Tests for turning energy into money, using round numbers whose answers are easy to check.

from localtune.cost import DEFAULT_PRICE, ElectricityPrice, energy_cost, hourly_cost

PRICE = ElectricityPrice(per_kwh=0.20, source="test")


def test_energy_cost():
    assert energy_cost(500, PRICE) == 0.10


def test_hourly_cost_is_one_hour_of_energy():
    assert hourly_cost(250, PRICE) == energy_cost(250, PRICE) == 0.05


def test_default_price_names_its_source():
    assert DEFAULT_PRICE.per_kwh > 0
    assert DEFAULT_PRICE.source
