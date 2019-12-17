import pytest
from tests.core.network.test_grids import TestMVGridDing0
from ding0.tools.results import calculate_mvgd_stats
import pandas as pd
from pandas.util.testing import assert_frame_equal
import os


TEST_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "test_data"))


class TestCalculateStats(object):

    @pytest.fixture
    def connect_generators(self):
        network, _, _ = TestMVGridDing0().minimal_unrouted_testgrid()
        network.mv_routing()
        network.connect_generators()
        return network

    @pytest.fixture
    def set_branch_ids(self, connect_generators):
        connect_generators.set_branch_ids()
        return connect_generators

    @pytest.fixture
    def set_circuit_breakers(self, set_branch_ids):
        set_branch_ids.set_circuit_breakers(debug=False)
        return set_branch_ids

    @pytest.fixture
    def run_powerflow(self, set_circuit_breakers):
        set_circuit_breakers.run_powerflow(
            "session", method='onthefly', export_pypsa=False, debug=False)
        return set_circuit_breakers

    @pytest.fixture
    def reinforce_grid(self, run_powerflow):
        run_powerflow.reinforce_grid()
        return run_powerflow

    @pytest.fixture
    def control_circuit_breakers(self, reinforce_grid):
        reinforce_grid.control_circuit_breakers(mode='close')
        return reinforce_grid

    @pytest.mark.dependency()
    def test_calculate_stats_connect_generators(self, connect_generators):
        mvgd_stats = calculate_mvgd_stats(connect_generators).reset_index()
        mvgd_stats_expected = pd.read_csv(os.path.join(
            TEST_DATA_PATH,
            "mvgd_stats_testgrid_after_connect_generators_expected.csv"))
        assert_frame_equal(mvgd_stats, mvgd_stats_expected, check_dtype=False)

    @pytest.mark.dependency(depends=[
        "TestCalculateStats::test_calculate_stats_connect_generators"])
    def test_calculate_stats_set_branch_ids(self, set_branch_ids):
        mvgd_stats = calculate_mvgd_stats(set_branch_ids).reset_index()
        mvgd_stats_expected = pd.read_csv(os.path.join(
            TEST_DATA_PATH,
            "mvgd_stats_testgrid_after_set_branch_id_expected.csv"))
        assert_frame_equal(mvgd_stats, mvgd_stats_expected, check_dtype=False)

    @pytest.mark.dependency(depends=[
        "TestCalculateStats::test_calculate_stats_set_branch_ids"])
    def test_calculate_stats_set_circuit_breakers(self, set_circuit_breakers):
        mvgd_stats = calculate_mvgd_stats(set_circuit_breakers).reset_index()
        mvgd_stats_expected = pd.read_csv(os.path.join(
            TEST_DATA_PATH,
            "mvgd_stats_testgrid_after_set_circuit_breakers_expected.csv"))
        assert_frame_equal(mvgd_stats, mvgd_stats_expected, check_dtype=False)

    @pytest.mark.dependency(depends=[
        "TestCalculateStats::test_calculate_stats_set_circuit_breakers"])
    def test_calculate_stats_run_powerflow(self, run_powerflow):
        mvgd_stats = calculate_mvgd_stats(run_powerflow).reset_index()
        mvgd_stats_expected = pd.read_csv(os.path.join(
            TEST_DATA_PATH,
            "mvgd_stats_testgrid_after_run_powerflow_expected.csv"))
        assert_frame_equal(mvgd_stats, mvgd_stats_expected, check_dtype=False)

    @pytest.mark.dependency(depends=[
        "TestCalculateStats::test_calculate_stats_run_powerflow"])
    def test_calculate_stats_reinforce_grid(self, reinforce_grid):
        mvgd_stats = calculate_mvgd_stats(reinforce_grid).reset_index()
        mvgd_stats_expected = pd.read_csv(os.path.join(
            TEST_DATA_PATH,
            "mvgd_stats_testgrid_after_reinforce_grid_expected.csv"))
        assert_frame_equal(mvgd_stats, mvgd_stats_expected, check_dtype=False)

    @pytest.mark.dependency(depends=[
        "TestCalculateStats::test_calculate_stats_reinforce_grid"])
    def test_calculate_stats_control_circuit_breakers(self, control_circuit_breakers):
        mvgd_stats = calculate_mvgd_stats(control_circuit_breakers).reset_index()
        mvgd_stats_expected = pd.read_csv(os.path.join(
            TEST_DATA_PATH,
            "mvgd_stats_testgrid_after_control_circuit_breakers-close_expected.csv"))
        assert_frame_equal(mvgd_stats, mvgd_stats_expected, check_dtype=False)


def create_test_expected_files(savepath=None):
    nd, mv_grid, lv_stations = \
        TestMVGridDing0().minimal_unrouted_testgrid()
    nd.mv_routing()
    nd.connect_generators()
    mvgd_stats = calculate_mvgd_stats(nd)
    mvgd_stats.to_csv(os.path.join(
        savepath,
        "mvgd_stats_testgrid_after_connect_generators_expected.csv"))

    nd.set_branch_ids()
    mvgd_stats = calculate_mvgd_stats(nd)
    mvgd_stats.to_csv(os.path.join(
        savepath,
        "mvgd_stats_testgrid_after_set_branch_id_expected.csv"))

    nd.set_circuit_breakers(debug=False)
    mvgd_stats = calculate_mvgd_stats(nd)
    mvgd_stats.to_csv(os.path.join(
        savepath,
        "mvgd_stats_testgrid_after_set_circuit_breakers_expected.csv"))

    nd.run_powerflow("session", method='onthefly', export_pypsa=False, debug=False)
    mvgd_stats = calculate_mvgd_stats(nd)
    mvgd_stats.to_csv(os.path.join(
        savepath,
        "mvgd_stats_testgrid_after_run_powerflow_expected.csv"))

    nd.reinforce_grid()
    mvgd_stats = calculate_mvgd_stats(nd)
    mvgd_stats.to_csv(os.path.join(
        savepath,
        "mvgd_stats_testgrid_after_reinforce_grid_expected.csv"))

    nd.control_circuit_breakers(mode='close')
    mvgd_stats = calculate_mvgd_stats(nd)
    mvgd_stats.to_csv(os.path.join(
        savepath,
        "mvgd_stats_testgrid_after_control_circuit_breakers-close_expected.csv"))


if __name__ == "__main__":
    create_test_expected_files(savepath=TEST_DATA_PATH)