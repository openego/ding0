"""Nothing from the input data gets lost silently.

Targeted cases on the synthetic grid districts (no database): input data
ding0 cannot use must lead to an error or a message, never to a grid that
quietly lacks something.
"""

import logging

import pandas as pd
import pytest
from shapely.geometry import Point

import ding0.core
from ding0.config.config_lv_grids_osm import get_config_osm
from ding0.core import NetworkDing0
from ding0.grid.lv_grid.routing import identify_street_loads
from tests import synthetic_district
from tests.invariants import assert_no_violations, completeness


def build(monkeypatch, mvgd_ids, components=False):
    synthetic_district.patch_database(monkeypatch, components=components)
    network = NetworkDing0(name="synthetic")
    network._orm = {}
    messages = network.run_ding0(session=None, mv_grid_districts_no=mvgd_ids)
    return network, messages


@pytest.mark.xfail(strict=True, reason=(
    "B2: import_generators creates ValueError('False voltage level') without "
    "raising it, so the generator is dropped silently"))
def test_generator_of_unknown_voltage_level_is_not_dropped_silently(monkeypatch, caplog):
    """A generator with a voltage level outside 4..7 is either in the grid or
    reported (error or log message naming it)."""
    original = synthetic_district.res_generators

    def res_generators(mvgd_id=1):
        frame = original(mvgd_id)
        extra = frame.iloc[[0]].assign(gens_id="pv_level_3", voltage_level=3)
        return pd.concat([frame, extra], ignore_index=True)

    monkeypatch.setattr(synthetic_district, "res_generators", res_generators)
    with caplog.at_level(logging.WARNING):
        try:
            network, _ = build(monkeypatch, [1])
        except ValueError as error:
            assert "voltage level" in str(error).lower()
            return
    in_grid = {g.gens_id for g in completeness.grid_generators(list(network.mv_grid_districts())[0])}
    assert "pv_level_3" in in_grid or "pv_level_3" in caplog.text


@pytest.mark.xfail(strict=True, reason=(
    "B8: when the clustering of a load area fails, "
    "import_lv_load_areas_and_build_new_lv_districts returns, so all "
    "following load areas of the MV grid district are missing; the "
    "returned message is discarded"))
def test_failed_clustering_skips_only_its_load_area(monkeypatch):
    """If the LV clustering of one load area fails, the other load areas of
    the district are built and the failure is reported."""
    original = ding0.core.distance_restricted_clustering

    def clustering(simp_graph, n_cluster, street_loads_df, mv_grid_district, id_db):
        result = original(simp_graph, n_cluster, street_loads_df, mv_grid_district, id_db)
        if id_db == 101:
            return (False,) + tuple(result[1:])
        return result

    monkeypatch.setattr(ding0.core, "distance_restricted_clustering", clustering)
    synthetic_district.patch_database(monkeypatch)
    network = NetworkDing0(name="synthetic")
    network._orm = {}
    # only step 1 of run_ding0: without load areas the MV grid has no lines
    # and connecting generators loops forever (B23)
    network.import_mv_grid_districts(None, [1])
    load_areas = {la.id_db for la in list(network.mv_grid_districts())[0].lv_load_areas()}
    assert load_areas == {102, 103}
    assert any("101" in str(m) for m in network.message)


@pytest.mark.xfail(strict=True, reason=(
    "B23: calc_geo_branches_in_buffer widens its search radius without limit "
    "('while not branches'), so connecting a generator to an MV grid without "
    "lines never ends"))
def test_connecting_to_a_grid_without_lines_ends():
    """Searching lines near a node of an MV grid that has none ends with an
    error instead of looping forever."""
    import signal

    from ding0.core.network.grids import MVGridDing0
    from ding0.core.network.stations import MVStationDing0
    from ding0.tools.geo import calc_geo_branches_in_buffer

    grid = MVGridDing0(id_db=0, station=MVStationDing0(id_db=0, geo_data=Point(0, 0)))
    node = type("Node", (), {"geo_data": Point(100, 100)})()

    def timeout(signum, frame):
        raise TimeoutError("still searching after 5 s")

    previous = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(5)
    try:
        with pytest.raises(Exception) as error:
            calc_geo_branches_in_buffer(node, grid, 2000, 1000, proj=None)
        assert not isinstance(error.value, TimeoutError)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


@pytest.mark.xfail(strict=True, reason=(
    "B10: identify_street_loads drops loads above mv_lv_threshold_capacity "
    "(>), while loads from it (>=) are MV loads, so a building of exactly "
    "the threshold is counted twice"))
def test_building_at_the_mv_threshold_is_no_street_load():
    threshold = get_config_osm("mv_lv_threshold_capacity")
    buildings = pd.DataFrame({"capacity": [50.0, threshold], "nn": [1, 2],
                              "number_households": [1, 0]})
    street_loads, _ = identify_street_loads(buildings, graph=None)
    assert street_loads.capacity.sum() == pytest.approx(50.0)

