import pytest

from ding0.core.powerflow import q_sign


def test_q_sign():
    """
    Checks that the correct
    sign of the reactive power values are obtained
    in the correct sign convention given active
    power, cosine phi and the mode of cosine phi i.e.
    "inductive" or "capacitive".
    """
    assert q_sign('inductive', 'generator') == -1
    assert q_sign('capacitive', 'generator') == 1
    assert q_sign('inductive', 'load') == 1
    assert q_sign('capacitive', 'load') == -1
