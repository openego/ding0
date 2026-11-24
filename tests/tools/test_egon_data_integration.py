import pytest
from types import SimpleNamespace

import shapely.wkt
from sqlalchemy import text

from ding0.tools import database
from ding0.tools import egon_data_integration as db_io
from ding0.core import NetworkDing0


class TestEgonDataIntegration:
    @pytest.fixture
    def session(self):
        with database.session_scope() as session:
            self.nd = NetworkDing0(name="network", session=session)
            print("Open session")
            yield session
            print("Close session")

    @pytest.fixture
    def grid_id(self, session):
        # the MV grid district with the fewest load areas (at least 3), so
        # that the tests run fast on any egon-data database
        bus_id = session.execute(text(
            "SELECT bus_id FROM demand.egon_loadarea GROUP BY bus_id "
            "HAVING count(*) >= 3 ORDER BY count(*), bus_id LIMIT 1"
        )).scalar()
        return [int(bus_id)]

    @pytest.fixture
    def mv_grid_district(self, session, grid_id):
        df = db_io.get_mv_data(self.nd.orm, session, grid_id)
        return SimpleNamespace(
            id_db=grid_id[0], geo_data=shapely.wkt.loads(df.poly_geom.iloc[0]))

    @pytest.fixture
    def lv_load_area(self, session, grid_id):
        df = db_io.get_lv_load_areas(self.nd.orm, session, grid_id[0])
        return df.loc[df.area.idxmax()]

    def test_import_mvgd(self, session, grid_id):
        df = db_io.get_mv_data(self.nd.orm, session, grid_id)

        assert list(df.index) == grid_id

    def test_import_load_area(self, session, grid_id):
        df = db_io.get_lv_load_areas(self.nd.orm, session, grid_id[0])

        assert len(df) >= 3

    def test_get_egon_ways(self, session, lv_load_area):
        df = db_io.get_egon_ways(self.nd.orm, session, lv_load_area.geo_area)

        assert not df.empty

    def test_get_egon_buildings(self, session, grid_id, lv_load_area):
        df = db_io.get_egon_buildings(self.nd.orm, session, grid_id[0], lv_load_area)

        assert not df.empty

    def test_get_res_generators(self, session, mv_grid_district):
        df = db_io.get_res_generators(self.nd.orm, session, mv_grid_district)

        assert (df.subst_id == mv_grid_district.id_db).all()

    def test_get_conv_generators(self, session, mv_grid_district):
        df = db_io.get_conv_generators(self.nd.orm, session, mv_grid_district)

        assert (df.subst_id == mv_grid_district.id_db).all()
