import pytest

from ding0.core import NetworkDing0
from ding0.tools import config as cfg_ding0


@pytest.fixture(scope="session")
def static_data():
    """Equipment data of the default planning profile."""
    return NetworkDing0(name="static_data").static_data


@pytest.fixture(scope="session")
def lvgd_cfg(static_data):
    """Parameters of the LV grid generation as `build_branches_on_osm_ways`
    collects them."""
    return {
        "v_nom": cfg_ding0.get("assumptions", "lv_nominal_voltage") / 1e3,
        "v_diff_max": float(cfg_ding0.get("assumptions", "lv_max_v_level_lc_diff_normal")),
        "cos_phi_load": cfg_ding0.get("assumptions", "cos_phi_load"),
        "cos_phi_load_mode": cfg_ding0.get("assumptions", "cos_phi_load_mode"),
        "lv_cable_lf": cfg_ding0.get("assumptions", "load_factor_lv_cable_lc_normal"),
        "lv_cables_df": static_data["LV_cables"].sort_values("I_max_th"),
    }
