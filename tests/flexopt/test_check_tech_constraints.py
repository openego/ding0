import pytest

from numpy import sqrt
from ding0.flexopt.check_tech_constraints import (
    voltage_delta_vde
)


def test_voltage_delta_vde():
    r"""
    Checks the calculation of the change in the
    voltage according to the [#VDE]_

    The formulation is:

        * for change in voltage across an inductive generator:
            .. math::
                \\Delta u = \\frac{S_{Amax} \cdot ( R_{kV} \cdot cos(\phi) + X_{kV} \cdot sin(\phi) )}{U_{nom}}

        * for change in voltage across a capacitive generator:
            .. math::
                \\Delta u = \\frac{S_{Amax} \cdot ( R_{kV} \cdot cos(\phi) - X_{kV} \cdot sin(\phi) )}{U_{nom}}

    =================  =============================
    Symbol             Description
    =================  =============================
    :math:`\Delta u`   Voltage drop/increase at node
    :math:`S_{Amax}`   Apparent power
    :math:`R_{kV}`     Resistance across points
    :math:`X_{kV}`     Reactance across points
    :math:`cos(\phi)`  Power factor
    :math:`U_{nom}`    Nominal voltage
    =================  =============================

    References
    ----------
    .. [#VDE] VDE Anwenderrichtlinie: Erzeugungsanlagen am Niederspannungsnetz –
        Technische Mindestanforderungen für Anschluss und Parallelbetrieb von
        Erzeugungsanlagen am Niederspannungsnetz, 2011
    """

    v_nom = 1000  # V
    s_max = 1  # kVA
    r = 1  # Ohms
    x = 1  # Ohms
    cos_phi = 1/sqrt(2)  # no unit i.e. 1/ sqrt(2) => angle is 45 degrees
    x_sign_capacitive = 1
    x_sign_inductive = -1
    voltage_delta_inductive = voltage_delta_vde(v_nom,
                                                s_max,
                                                r,
                                                x_sign_inductive * x,
                                                cos_phi)
    voltage_delta_inductive_expected = sqrt(2)/1000
    voltage_delta_capacitive = voltage_delta_vde(v_nom,
                                                 s_max,
                                                 r,
                                                 x_sign_capacitive * x,
                                                 cos_phi)
    voltage_delta_capacitive_expected = 0

    assert voltage_delta_inductive == pytest.approx(voltage_delta_inductive_expected,
                                                    abs=0.000001)
    assert voltage_delta_capacitive == pytest.approx(voltage_delta_capacitive_expected,
                                                     abs=0.000001)
