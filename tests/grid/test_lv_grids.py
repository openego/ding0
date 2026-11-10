from hypothesis import given, strategies as st

from ding0.grid.lv_grid.routing import get_lvgd_id


@given(st.integers(1, 224497), st.integers(0, 9999),
       st.integers(1, 224497), st.integers(0, 9999))
def test_lv_grid_districts_of_different_clusters_get_different_ids(la1, c1, la2, c2):
    if (la1, c1) != (la2, c2):
        assert get_lvgd_id(la1, c1) != get_lvgd_id(la2, c2)


def test_lv_grid_districts_of_load_areas_92_and_920_get_different_ids():
    assert get_lvgd_id(92, 0) != get_lvgd_id(920, 0)
