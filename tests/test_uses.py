# Tests for the prompt and answer lengths each kind of use stands for, a percent of the context,
# and that lengths given by hand win over them.

import pytest

from localtune.uses import (
    DEFAULT_LENGTHS,
    USES,
    UnknownUseError,
    percent_lengths,
    token_lengths,
    use_label,
    use_percent,
)


def test_no_use_keeps_the_usual_lengths():
    assert token_lengths(None, None, None) == DEFAULT_LENGTHS


def test_a_use_sets_both_lengths():
    assert token_lengths("documents", None, None) == USES["documents"]


def test_lengths_given_by_hand_win_over_the_use():
    assert token_lengths("code", 2000, None) == (2000, USES["code"][1])


def test_an_unknown_use_is_refused():
    with pytest.raises(UnknownUseError, match="chat, code, documents"):
        token_lengths("poems", None, None)


def test_a_percent_fills_that_much_of_the_context():
    assert token_lengths("30", None, None, context=40960) == (12288 - 256, 256)


def test_a_percent_sign_is_allowed_and_saved_the_same():
    assert token_lengths("30%", None, None, context=40960) == token_lengths("30", None, None, 40960)
    assert use_label("30") == use_label("30%") == "30%"
    assert use_label("chat") == "chat"


def test_a_small_percent_keeps_half_for_the_answer():
    assert percent_lengths(1, 2048) == (10, 10)


def test_the_whole_context_fits_exactly():
    assert sum(percent_lengths(100, 32768)) == 32768


@pytest.mark.parametrize("use", ["0", "101", "-5", "half", "30.5"])
def test_a_percent_outside_1_to_100_is_refused(use):
    with pytest.raises(UnknownUseError, match="from 1 to 100"):
        use_percent(use)


def test_a_percent_without_a_context_is_refused():
    with pytest.raises(UnknownUseError, match="needs to know the model's context"):
        token_lengths("30", None, None)
