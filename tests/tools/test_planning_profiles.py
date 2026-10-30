"""Tests for the planning profiles, run without a database."""
from types import SimpleNamespace

import pandas as pd
import pytest

from ding0.config.config_lv_grids_osm import get_config_osm
from ding0.core import NetworkDing0
from ding0.tools import config as cfg_ding0
from ding0.tools import planning_profiles
from ding0.tools.egon_data_integration import filter_generators_by_capacity


@pytest.fixture(autouse=True)
def default_profile():
    """Loads the config and resets the global profile state after each test."""
    for filename in ["config_db_tables.cfg", "config_calc.cfg", "config_files.cfg",
                     "config_misc.cfg"]:
        cfg_ding0.load_config(filename)
    planning_profiles.apply_profile(planning_profiles.DEFAULT_PROFILE)
    yield
    planning_profiles.apply_profile(planning_profiles.DEFAULT_PROFILE)


def static_data(profile):
    return NetworkDing0.import_static_data(SimpleNamespace(_planning_profile=profile))


class TestPlanningProfiles:
    def test_available_profiles(self):
        profiles = planning_profiles.available_profiles()
        assert profiles[0] == "default"
        assert "snb" in profiles

    def test_default_values(self):
        assert cfg_ding0.get("mv_routing", "load_area_count_per_ring") == 20
        assert cfg_ding0.get("assumptions", "cos_phi_load") == 0.97
        assert cfg_ding0.get("assumptions", "cos_phi_load_mode") == "inductive"
        assert cfg_ding0.get("assumptions", "lv_standard_line") == "NAYY 4x1x150"
        assert planning_profiles.get_limit("assumptions", "max_generator_capacity") is None
        assert get_config_osm("hv_mv_threshold_capacity") == 5500
        assert isinstance(get_config_osm("hv_mv_threshold_capacity"), int)
        assert get_config_osm("additional_trafo_capacity") == 1.0
        # technical parameters are not part of the profile
        assert get_config_osm("srid") == 3035
        assert cfg_ding0.get("assumptions", "frequency") == 50

    def test_all_parameters_applied(self):
        for profile in planning_profiles.available_profiles():
            planning_profiles.apply_profile(profile)
            for (section, key), value in planning_profiles.get_parameters(profile).items():
                if section == planning_profiles.OSM_SECTION:
                    assert get_config_osm(key) == planning_profiles._parse_value(value)
                else:
                    assert cfg_ding0.cfg.get(section, key) == value

    def test_snb_overrides_and_inherits(self):
        planning_profiles.apply_profile("snb")
        assert planning_profiles.get_active_profile() == "snb"
        # set in snb
        assert cfg_ding0.get("mv_routing", "max_half_ring_length") == 9
        assert get_config_osm("hv_mv_threshold_capacity") == 13500
        assert planning_profiles.get_limit("assumptions", "max_generator_capacity") == 13500
        # inherited from default
        assert cfg_ding0.get("assumptions", "branch_detour_factor") == 1.3
        assert get_config_osm("ons_dist_threshold") == 1500

    def test_reapply_default_restores_values(self):
        planning_profiles.apply_profile("snb")
        planning_profiles.apply_profile("default")
        assert cfg_ding0.get("mv_routing", "max_half_ring_length") == 28
        assert get_config_osm("hv_mv_threshold_capacity") == 5500

    def test_reloading_config_keeps_active_profile(self):
        planning_profiles.apply_profile("snb")
        cfg_ding0.load_config("config_calc.cfg")
        assert cfg_ding0.get("mv_routing", "max_half_ring_length") == 9

    def test_unknown_profile(self):
        with pytest.raises(ValueError, match="Unknown planning profile"):
            planning_profiles.apply_profile("does_not_exist")


class TestEquipmentSelection:
    def test_default_equipment(self):
        data = static_data("default")
        assert list(data["MV_trafos"]["S_nom"]) == [25000, 40000, 63000]
        assert len(data["LV_cables"]) == 8
        assert list(data["LV_trafos"].index) == list(range(len(data["LV_trafos"])))

    def test_snb_equipment(self):
        data = static_data("snb")
        assert list(data["MV_trafos"]["S_nom"]) == [31500, 40000]
        assert list(data["LV_trafos"]["S_nom"]) == [400, 630, 800]
        assert "NAYY 4x1x300" not in data["LV_cables"].index
        assert "NAYY 4x1x150" in data["LV_cables"].index
        assert len(data["MV_cables"]) == len(static_data("default")["MV_cables"])

    def test_selection_must_match_catalog(self):
        data = static_data("default")
        data["LV_cables"] = data["LV_cables"].drop("NAYY 4x1x35")
        with pytest.raises(ValueError, match="does not match"):
            planning_profiles.select_equipment(data, "default")


class TestGeneratorCapacity:
    generators = pd.DataFrame({"electrical_capacity": [100.0, 13500.0, 13600.0, 50000.0]})

    def test_no_limit_in_default(self):
        result = filter_generators_by_capacity(self.generators, "1")
        assert len(result) == 4

    def test_limit_in_snb(self):
        planning_profiles.apply_profile("snb")
        result = filter_generators_by_capacity(self.generators, "1")
        assert list(result["electrical_capacity"]) == [100.0, 13500.0]
