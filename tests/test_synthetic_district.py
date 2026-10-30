"""End-to-end runs of ding0 on synthetic grid districts without a database.

Three runs are compared with reference grids: grid district 1 alone, grid
district 2 alone (urban MV routing with stubs, satellite, industrial loads,
generators of voltage levels 4 to 7) and both together in one run, which is
how ding0 is used for more than one grid district. See
`tests/synthetic_district.py` for the data.

The reference grids in `test_data/` were produced by upstream ding0
(4d113a0) on its original stack (Python 3.8, pandas 1.2, networkx 2.5,
shapely 1.8, pypsa 0.17), so they record upstream behaviour, including its
bugs: in the run of both districts, only district 1's generators are
imported. The comparison ignores component names, which depend on iteration
order, and compares the grid by geometry: buses by coordinates, lines by end
points, length, type and parallel count.

Set DING0_UPDATE_EXPECTED=1 to rewrite the expected files from the current
runs.
"""

import json
import os

import pytest

from ding0.core import NetworkDing0
from tests import synthetic_district

TEST_DATA = os.path.join(os.path.dirname(__file__), "test_data")

# run name: MV grid districts of the run
RUNS = {
    "district_1": [1],
    "district_2": [2],
    "districts_1_2": [1, 2],
}

# per MV grid district: aggregated and satellite load areas and the number
# of LV grid districts per load area
STRUCTURE = {
    1: {"aggregated": [], "satellite": [], "lv_grids": {101: 3, 102: 1, 103: 1}},
    2: {"aggregated": [201], "satellite": [203], "lv_grids": {201: 14, 202: 1, 203: 1}},
}


def expected_file(run):
    return os.path.join(TEST_DATA, f"synthetic_{run}_expected.json")


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


@pytest.fixture(scope="module", params=list(RUNS))
def synthetic_run(request):
    monkeypatch = pytest.MonkeyPatch()
    synthetic_district.patch_database(monkeypatch)
    try:
        network = NetworkDing0(name="synthetic")
        # The patched getters ignore the ORM; an empty one keeps NetworkDing0
        # from reflecting the database tables.
        network._orm = {}
        messages = network.run_ding0(session=None,
                                     mv_grid_districts_no=RUNS[request.param])
        components = network.to_dataframe()
    finally:
        monkeypatch.undo()
    return request.param, network, messages, components


def test_run_completes_without_invalid_districts(synthetic_run):
    _, _, messages, _ = synthetic_run
    assert messages == [[]]


def test_load_area_structure(synthetic_run):
    _, network, _, _ = synthetic_run
    structure = {
        mvgd.id_db: {
            "aggregated": sorted(la.id_db for la in mvgd.lv_load_areas() if la.is_aggregated),
            "satellite": sorted(la.id_db for la in mvgd.lv_load_areas() if la.is_satellite),
            "lv_grids": {la.id_db: len(list(la.lv_grid_districts()))
                         for la in mvgd.lv_load_areas()},
        }
        for mvgd in network.mv_grid_districts()
    }
    assert structure == {mvgd_id: STRUCTURE[mvgd_id] for mvgd_id in structure}


def test_grid_matches_upstream_reference(synthetic_run):
    run, _, _, components = synthetic_run
    grid = canonical_grid(components)
    if os.environ.get("DING0_UPDATE_EXPECTED"):
        with open(expected_file(run), "w") as f:
            json.dump(grid, f, indent=1)
    with open(expected_file(run)) as f:
        expected = json.load(f)
    for key in expected:
        assert grid[key] == expected[key], key
