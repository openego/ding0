import pytest
from egoio.tools import db
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def oedb_session():
    """
    Returns an ego.io oedb session and closes it on finishing the test
    """
    engine = db.connection(readonly=True)
    session = sessionmaker(bind=engine)()
    yield session
    print("closing session")
    session.close()


# Tests that need an egon-data database. Deselect with -m "not db".
NEEDS_DATABASE = [
    "tests/tools/test_database.py",
    "tests/tools/test_egon_data_integration.py",
    "tests/core/network/test_grids.py::TestMVGridDing0::test_routing",
    "tests/core/network/test_network.py::TestGridDing0::test_grid_stats",
]

# Tests that also fail on upstream ding0 4d113a0 with its original stack
# (Python 3.8, pandas 1.2, networkx 2.5). They test behaviour ding0 no longer
# has. strict=True reports them as soon as one of them passes again.
_OLD_LV_MODEL = ("the minimal test grid builds LV grid districts of the old "
                 "model-grid approach without buildings, which the OSM-based "
                 "LV builder needs")
KNOWN_STALE = {
    "tests/core/network/test_grids.py::TestMVGridDing0::test_local_routing": _OLD_LV_MODEL,
    "tests/core/network/test_grids.py::TestMVGridDing0::test_construct": _OLD_LV_MODEL,
    "tests/core/test_core.py::TestNetworkDing0::test_to_csv": _OLD_LV_MODEL,
    "tests/core/test_core.py::TestNetworkDing0::test_run_powerflow": _OLD_LV_MODEL,
    "tests/grid/test_grid.py::test_mv_connect_generators": _OLD_LV_MODEL,
    "tests/tools/test_results.py::TestCalculateStats::test_calculate_stats_connect_generators": _OLD_LV_MODEL,
    "tests/core/network/test_grids.py::TestLVGridDing0": (
        "the fixture passes a Series as peak_load and builds LV grids with the "
        "old model-grid approach"),
    "tests/core/network/test_network.py::TestGridDing0::test_loads_list": (
        "expects LoadDing0 to increment id_db, which it no longer does"),
    "tests/core/network/test_network.py::TestGridDing0::test_graph_add_node_load": (
        "expects graph_add_node to ignore loads, which are graph nodes since "
        "the OSM-based LV grids"),
}


def pytest_collection_modifyitems(config, items):
    for item in items:
        if any(item.nodeid.startswith(prefix) for prefix in NEEDS_DATABASE):
            item.add_marker(pytest.mark.db)
        for prefix, reason in KNOWN_STALE.items():
            if item.nodeid.startswith(prefix):
                item.add_marker(pytest.mark.xfail(reason=reason, strict=True))
