# Tests for the price lookup. They swap the real tables for tiny ones written here, so they can
# check the order localtune tries (zip, then state or country, then the US average, with your
# own price always winning) without depending on what the real data says this year.

import pytest

from localtune import prices
from localtune.cost import DEFAULT_PRICE


@pytest.fixture(autouse=True)
def small_tables(tmp_path, monkeypatch):
    (tmp_path / "prices_us_zips.csv").write_text("zip,state,per_kwh\n43210,OH,0.171\n")
    (tmp_path / "prices_us_states.csv").write_text(
        "state,per_kwh,period\nOH,0.165,2026-08\nCA,0.322,2026-08\n"
    )
    (tmp_path / "prices_countries.csv").write_text(
        "country,per_kwh,currency,period\nDE,0.395,EUR,2026-S1\n"
    )
    monkeypatch.setattr(prices, "DATA_DIR", tmp_path)


def test_zip_comes_first():
    price = prices.find_price(zip_code="43210", state="CA")
    assert price.per_kwh == 0.171
    assert "zip 43210" in price.source


def test_state():
    price = prices.find_price(state="ca")
    assert price.per_kwh == 0.322
    assert price.source == "CA average, EIA 2026-08"


def test_country_keeps_its_currency():
    price = prices.find_price(country="de")
    assert price.per_kwh == 0.395
    assert price.currency == "EUR"


def test_your_price_always_wins():
    price = prices.find_price(zip_code="43210", your_price=0.10)
    assert price.per_kwh == 0.10
    assert price.source == "your price"


def test_unknown_zip_falls_through_to_state():
    assert prices.find_price(zip_code="99999", state="OH").per_kwh == 0.165


def test_nothing_given_uses_the_us_average():
    assert prices.find_price() == DEFAULT_PRICE


def test_nothing_found_uses_the_us_average_and_says_so():
    price = prices.find_price(zip_code="99999")
    assert price.per_kwh == DEFAULT_PRICE.per_kwh
    assert "99999 not found" in price.source


def test_shipped_tables_are_readable(monkeypatch):
    # The real tables inside the package, not the small test ones: each loads and looks sane.
    monkeypatch.undo()
    for price in (
        prices.find_price(zip_code="90210"),
        prices.find_price(state="TX"),
        prices.find_price(country="FR"),
    ):
        assert "not found" not in price.source
        assert 0.01 < price.per_kwh < 1.0
