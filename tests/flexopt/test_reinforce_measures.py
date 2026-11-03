from types import SimpleNamespace

import pandas as pd
import pytest

from ding0.core.network import BranchDing0
from ding0.flexopt.reinforce_measures import (
    reinforce_branches_current,
    reinforce_branches_voltage,
)


class Branch(BranchDing0):
    """Branch without a grid; the real repr needs the grid's graph."""

    def __repr__(self):
        return "Branch"


@pytest.fixture
def grid():
    """MV grid stub with the 20 kV cable types of ding0's equipment data."""
    cables = pd.DataFrame(
        {
            "U_n": [20, 20, 20],
            "I_max_th": [319, 417, 476],
        },
        index=[
            "NA2XS2Y 3x1x150 RE/25",
            "NA2XS2Y 3x1x240",
            "NA2XS(FL)2Y 3x1x300 RM/25",
        ],
    )
    return SimpleNamespace(
        network=SimpleNamespace(static_data={"MV_cables": cables}),
        v_level=20,
    )


@pytest.fixture
def overhead_line():
    line_type = pd.Series({"U_n": 20, "I_max_th": 210}, name="48-AL1/8-ST1A")
    return Branch(kind="line", type=line_type)


def test_reinforce_branches_current_turns_overhead_line_into_cable(grid, overhead_line):
    reinforce_branches_current(grid, {overhead_line: 1.2})

    assert overhead_line.type.name == "NA2XS2Y 3x1x150 RE/25"
    assert overhead_line.kind == "cable"


def test_reinforce_branches_voltage_turns_overhead_line_into_cable(grid, overhead_line):
    reinforce_branches_voltage(grid, [overhead_line])

    assert overhead_line.type.name == "NA2XS2Y 3x1x150 RE/25"
    assert overhead_line.kind == "cable"


def test_branch_without_suitable_cable_keeps_its_kind(grid):
    line_type = pd.Series({"U_n": 20, "I_max_th": 410}, name="122-AL1/20-ST1A")
    line = Branch(kind="line", type=line_type)

    reinforce_branches_current(grid, {line: 2.0})

    assert line.type.name == "122-AL1/20-ST1A"
    assert line.kind == "line"
