# Tests for the prompt and answer lengths each kind of use stands for, and that lengths given by
# hand win over them.

import pytest

from localtune.uses import DEFAULT_LENGTHS, USES, UnknownUseError, token_lengths


def test_no_use_keeps_the_usual_lengths():
    assert token_lengths(None, None, None) == DEFAULT_LENGTHS


def test_a_use_sets_both_lengths():
    assert token_lengths("documents", None, None) == USES["documents"]


def test_lengths_given_by_hand_win_over_the_use():
    assert token_lengths("code", 2000, None) == (2000, USES["code"][1])


def test_an_unknown_use_is_refused():
    with pytest.raises(UnknownUseError, match="chat, code, documents"):
        token_lengths("poems", None, None)
