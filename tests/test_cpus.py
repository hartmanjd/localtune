# Tests for cpus.py: matching the CPU names Windows and Linux report to the CPU table, and the
# RAM speed a run gets from its share of the CPU's cores.

import pytest

from localtune.cpus import find_cpu_spec, load_cpu_table, ram_bandwidth_gbs


def test_every_row_says_where_it_came_from():
    assert all(spec.source for spec in load_cpu_table())


@pytest.mark.parametrize(
    "reported, name",
    [
        ("12th Gen Intel(R) Core(TM) i9-12900K", "Core i9-12900K"),
        ("Intel(R) Xeon(R) CPU E5-2696 v3 @ 2.30GHz", "Xeon E5-2696 v3"),
        ("AMD EPYC 7402 24-Core Processor", "EPYC 7402"),
        ("AMD Ryzen Threadripper PRO 7975WX 32-Cores", "Ryzen Threadripper PRO 7975WX"),
    ],
)
def test_reported_names_match_the_table(reported, name):
    assert find_cpu_spec(reported).name == name


def test_an_unknown_cpu_has_no_ram_speed():
    assert find_cpu_spec("Intel64 Family 6 Model 151 Stepping 2, GenuineIntel") is None
    assert ram_bandwidth_gbs(None, 8) is None


# A rented machine with 9 threads of an 18-core Xeon E5-2696 v3 (68 GB/s) gets half of it; a
# desktop using all its cores gets all of it, and more threads than cores don't add any.


def test_ram_speed_scales_with_the_share_of_cores():
    assert ram_bandwidth_gbs("Intel(R) Xeon(R) CPU E5-2696 v3 @ 2.30GHz", 9) == pytest.approx(34)
    assert ram_bandwidth_gbs("12th Gen Intel(R) Core(TM) i9-12900K", 24) == pytest.approx(76.8)
    assert ram_bandwidth_gbs("12th Gen Intel(R) Core(TM) i9-12900K", None) == pytest.approx(76.8)
