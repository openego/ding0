"""End-to-end run of ding0 on a synthetic grid district without a database.

The expected grid in `test_data/synthetic_district_expected.json` was produced
by upstream ding0 (4d113a0) on its original stack (Python 3.8, pandas 1.2,
networkx 2.5, shapely 1.8, pypsa 0.17). The comparison ignores component
names, which depend on iteration order, and compares the grid by geometry:
buses by coordinates, lines by end points, length, type and parallel count.

Set DING0_UPDATE_EXPECTED=1 to rewrite the expected file from the current run.
"""

import json
import os

import pytest

from ding0.core import NetworkDing0
from tests import synthetic_district

EXPECTED = os.path.join(os.path.dirname(__file__), "test_data",
                        "synthetic_district_expected.json")


def canonical_grid(components):
    """Name-independent description of an exported grid."""
    buses = components["Bus"]
    xy = {name: [round(row.x, 3), round(row.y, 3), float(row.v_nom)]
          for name, row in buses.iterrows()}
    lines = components["Line"]
    loads = components["Load"]
    generators = components["Generator"]
    transformers = components["Transformer"]
    return {
        "buses": sorted(xy.values()),
        "lines": sorted(
            [sorted([xy[row.bus0], xy[row.bus1]]), round(row.length, 6),
             row.type_info, int(row.num_parallel)]
            for _, row in lines.iterrows()),
        "loads": sorted(
            [xy[row.bus], round(row.p_set, 9), row.sector]
            for _, row in loads.iterrows()),
        "generators": sorted(
            [xy[row.bus], round(row.p_nom, 9), row.type]
            for _, row in generators.iterrows()),
        "transformers": sorted(
            [xy[row.bus0], xy[row.bus1], round(float(row.s_nom), 9)]
            for _, row in transformers.iterrows()),
    }


@pytest.fixture(scope="module")
def synthetic_run():
    monkeypatch = pytest.MonkeyPatch()
    synthetic_district.patch_database(monkeypatch)
    try:
        network = NetworkDing0(name="synthetic")
        # The patched getters ignore the ORM; an empty one keeps NetworkDing0
        # from reflecting the database tables.
        network._orm = {}
        messages = network.run_ding0(session=None,
                                     mv_grid_districts_no=[synthetic_district.MV_GRID_ID])
        components = network.to_dataframe()
    finally:
        monkeypatch.undo()
    return network, messages, components


def test_run_completes_without_invalid_districts(synthetic_run):
    _, messages, _ = synthetic_run
    assert messages == [[]]


def test_lv_grids_per_load_area(synthetic_run):
    network, _, _ = synthetic_run
    (mv_grid_district,) = list(network.mv_grid_districts())
    lv_grids = {la.id_db: len(list(la.lv_grid_districts()))
                for la in mv_grid_district.lv_load_areas()}
    assert lv_grids == {101: 3, 102: 1, 103: 1}


def test_grid_matches_upstream_reference(synthetic_run):
    _, _, components = synthetic_run
    grid = canonical_grid(components)
    if os.environ.get("DING0_UPDATE_EXPECTED"):
        with open(EXPECTED, "w") as f:
            json.dump(grid, f, indent=1)
    with open(EXPECTED) as f:
        expected = json.load(f)
    for key in expected:
        assert grid[key] == expected[key], key
