"""Components of newer egon-data versions on synthetic grid district 2.

Heat pumps, charging points, bus depots, rail substations and battery storage
units have to be exported as eDisGo expects them: heat pumps and charging
points as loads of type 'heat_pump' and 'charging_point', rail substations as
conventional loads of sector 'rail', batteries as storage units. See
`tests/synthetic_district.py` for the data.
"""

import pandas as pd
import pypsa
import pytest
from shapely.geometry import Point

import ding0.tools.egon_data_integration as db_io
from ding0.core import NetworkDing0
from tests import synthetic_district

# columns of eDisGo's topology (edisgo.network.topology.COLUMNS)
EDISGO_LOADS_COLUMNS = ["bus", "p_set", "building_id", "type",
                        "annual_consumption", "sector", "number_households"]
EDISGO_STORAGE_UNITS_COLUMNS = ["bus", "control", "p_nom", "max_hours"]
EDISGO_CHARGING_POINT_SECTORS = {"home", "work", "public", "hpc"}


@pytest.fixture(scope="module")
def components_run():
    monkeypatch = pytest.MonkeyPatch()
    synthetic_district.patch_database(monkeypatch, components=True)
    try:
        network = NetworkDing0(name="synthetic")
        network._orm = {}
        messages = network.run_ding0(session=None, mv_grid_districts_no=[2])
        components = network.to_dataframe()
    finally:
        monkeypatch.undo()
    return network, messages, components


def test_run_completes(components_run):
    _, messages, _ = components_run
    assert messages == [[]]


def test_loads_by_type_and_sector(components_run):
    _, _, components = components_run
    loads = components["Load"]
    p_set = loads.groupby(["type", "sector"]).p_set.sum().round(6).to_dict()
    assert p_set[("heat_pump", "individual_heating")] == 0.15
    # two home charging points in buildings, one at a location of its own
    assert p_set[("charging_point", "home")] == 0.033
    assert p_set[("charging_point", "public")] == 0.022
    assert p_set[("charging_point", "work")] == 0.15
    # urban fast, highway fast and bus depot
    assert p_set[("charging_point", "hpc")] == 3.1
    assert p_set[("conventional_load", "rail")] == 1.2
    charging_points = loads.loc[loads.type == "charging_point"]
    assert set(charging_points.sector) <= EDISGO_CHARGING_POINT_SECTORS


def test_components_have_edisgo_columns(components_run):
    _, _, components = components_run
    assert set(EDISGO_LOADS_COLUMNS) <= set(components["Load"].columns)
    assert set(EDISGO_STORAGE_UNITS_COLUMNS) <= set(components["StorageUnit"].columns)
    for df in (components["Load"], components["StorageUnit"]):
        assert df.bus.isin(components["Bus"].index).all()


def test_building_components_at_building_bus(components_run):
    _, _, components = components_run
    loads = components["Load"]
    building_loads = loads.loc[loads.type == "conventional_load"].dropna(
        subset=["building_id"])
    bus_of_building = building_loads.groupby("building_id").bus.first()
    in_buildings = loads.loc[loads.type.isin(["heat_pump", "charging_point"])
                             & loads.building_id.notna()]
    assert len(in_buildings) == 6
    assert (in_buildings.bus == in_buildings.building_id.map(bus_of_building)).all()
    home_storage = components["StorageUnit"].loc[
        components["StorageUnit"].type == "home_storage"]
    assert len(home_storage) == 2
    assert (home_storage.bus == home_storage.building_id.map(bus_of_building)).all()


def test_point_components_by_voltage_level(components_run):
    _, _, components = components_run
    v_nom = components["Bus"].v_nom
    loads = components["Load"]
    use_case_v_nom = loads.dropna(subset=["use_case"]).set_index("use_case").bus.map(v_nom)
    # voltage levels 4 and 5 in the MV grid, 6 and 7 in LV grids
    assert use_case_v_nom.to_dict() == {
        "home_detached": 0.4, "urban_fast": 10.0, "highway_fast": 10.0,
        "street": 0.4, "work": 0.4, "bus_depot": 10.0, "rail_transit_dc": 10.0}
    # voltage level 4 has a line of its own to the HV/MV station
    highway = loads.loc[loads.use_case == "highway_fast", "bus"].iloc[0]
    lines = components["Line"]
    to_station = lines.loc[(lines.bus0 == highway) | (lines.bus1 == highway)]
    assert len(to_station) == 1
    assert {to_station.bus0.iloc[0], to_station.bus1.iloc[0]} == {highway, "Busbar_mvgd_2_MV"}
    storage = components["StorageUnit"]
    large = storage.loc[storage.type == "large_storage"].iloc[0]
    assert (v_nom[large.bus], large.p_nom, large.max_hours, large.control) == (10.0, 0.5, 1.0, "PQ")


def test_building_components_in_grid_dimensioning(components_run):
    network, _, _ = components_run
    (mv_grid_district,) = list(network.mv_grid_districts())
    load_area = next(la for la in mv_grid_district.lv_load_areas() if la.id_db == 202)
    buildings = synthetic_district.buildings(202)
    lv_buildings = buildings.loc[buildings.capacity < 200]
    # heat pumps and home charging points in load area 202 (factors 1.0)
    assert load_area.peak_load == pytest.approx(lv_buildings.capacity.sum() + 30.0 + 22.0)


def test_csv_export_imports_into_pypsa(components_run, tmp_path):
    network, _, components = components_run
    network.to_csv(str(tmp_path))
    grid = pypsa.Network()
    grid.import_from_csv_folder(str(tmp_path / "2"))
    assert len(grid.storage_units) == 3
    assert (grid.storage_units.control == "PQ").all()
    assert set(grid.loads.type) == {"conventional_load", "heat_pump", "charging_point"}
    assert len(grid.loads) == len(components["Load"])


def test_assign_to_nearest_buildings():
    NetworkDing0()  # loads the config
    locations = pd.DataFrame({"location_id": [1, 2, 3], "p_set": [11.0, 22.0, 11.0],
                              "geometry": [Point(0, 0), Point(100, 0), Point(5, 5)]})
    buildings = pd.DataFrame({"geometry": [Point(1, 0), Point(5, 6)]},
                             index=pd.Index([10, 20], name="building_id"))
    assigned = db_io.assign_to_nearest_buildings(locations, buildings, 50.0)
    assert assigned.set_index("location_id").building_id.to_dict() == {1: 10, 3: 20}


def test_pypsa_network_runs_power_flow(components_run, tmp_path):
    network, _, components = components_run
    network.to_netcdf(str(tmp_path))
    grid = pypsa.Network(str(tmp_path / "2.nc"))
    assert len(grid.loads) == len(components["Load"])
    assert len(grid.storage_units) == 3
    assert grid.meta["mv_grid_district_id"] == 2
    assert grid.meta["ding0"]["scenario"] == "status2024"
    assert len(grid.meta["transformers_hvmv"]) == 2
    # slack at the MV busbar of the HV/MV station, as in eDisGo
    assert grid.generators.loc[grid.generators.control == "Slack", "bus"].tolist() == [
        "Busbar_mvgd_2_MV"]
    grid.determine_network_topology()
    assert len(grid.sub_networks) == 1
    result = grid.pf()
    assert result["converged"].all().all()


def test_csv_export_has_pypsa_version(components_run, tmp_path):
    network, _, _ = components_run
    network.to_csv(str(tmp_path))
    network_csv = pd.read_csv(tmp_path / "2" / "network.csv")
    assert network_csv.pypsa_version.iloc[0] == pypsa.__version__


def test_mv_point_loads_are_routed_in_rings(components_run):
    network, _, _ = components_run
    (mv_grid_district,) = list(network.mv_grid_districts())
    mv_grid = mv_grid_district.mv_grid
    ring_nodes = {node for ring in mv_grid.rings_nodes(include_root_node=False)
                  for node in ring}
    routed = {component.use_case for component in mv_grid.components()
              if component in ring_nodes}
    assert routed == {"urban_fast", "bus_depot", "rail_transit_dc"}


def test_point_loads_in_mv_grid_district_peak_load(components_run):
    network, _, _ = components_run
    (mv_grid_district,) = list(network.mv_grid_districts())
    load_areas = sum(la.peak_load for la in mv_grid_district.lv_load_areas())
    mv_loads = sum(load.peak_load for load in mv_grid_district.mv_grid.loads())
    # all point loads of district 2, incl. the home charging point without a
    # building nearby (factor 1.0)
    point_loads = 300.0 + 2_000.0 + 22.0 + 150.0 + 800.0 + 1_200.0 + 11.0
    assert mv_grid_district.peak_load == pytest.approx(load_areas + mv_loads + point_loads)
