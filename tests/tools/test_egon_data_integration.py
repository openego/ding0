import pytest
import subprocess

from ding0.tools import database
from ding0.tools import egon_data_integration as db_io
from ding0.core import NetworkDing0
import pandas as pd

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
        import saio
        import geopandas as gpd

        engine = session.bind

        saio.register_schema("grid", engine)
        saio.register_schema("boundaries", engine)

        from saio.grid import egon_hvmv_substation as MVGridDistricts
        from saio.boundaries import vg250_lan

        table = MVGridDistricts
        cells_query = session.query(
            table.bus_id.label("subst_id"), table.point.label("geom")
        )
        df = gpd.read_postgis(
            sql=cells_query.statement, con=cells_query.session.bind, index_col=None
        )

        cells_query = session.query(
            vg250_lan.gen.label("name"), vg250_lan.geometry.label("geom")
        ).filter(vg250_lan.gf == 4)

        boundaries_gdf = gpd.read_postgis(
            sql=cells_query.statement,
            con=cells_query.session.bind,
            index_col="name",
        )

        df = df[
            df.geometry.intersects(boundaries_gdf.loc["Schleswig-Holstein", "geom"])
        ]["subst_id"]

        print("Got the DATA!")
        return [int(df[1])]

    @pytest.fixture
    def grid_id_2(self, session):

        sql_statement = """
        SELECT *
        FROM
        (
            SELECT
                generators.bus_id,
                count(*) AS count,
                count(*) FILTER (WHERE generators.type = 'wind') AS wind,
                count(*) FILTER (WHERE generators.type = 'pv_open_place') AS pv_open_place,
                count(*) FILTER (WHERE generators.type = 'pv_rooftop') AS pv_rooftop,
                count(*) FILTER (WHERE generators.type = 'biomass') AS biomass,
                count(*) FILTER (WHERE generators.type = 'hydro') AS hydro
            FROM
            (
                (
                    SELECT bus_id, 'wind' AS type
                    FROM supply.egon_power_plants_wind
                )
            UNION ALL
                (
                    SELECT bus_id, 'pv_open_place' AS type
                    FROM supply.egon_power_plants_pv
                )
            UNION ALL
                (
                    SELECT bus_id, 'pv_rooftop' AS type
                    FROM supply.egon_power_plants_pv_roof_building
                )
            UNION ALL
                (
                    SELECT bus_id, 'biomass' AS type
                    FROM supply.egon_power_plants_biomass
                )
            UNION ALL
                (
                    SELECT bus_id, 'hydro' AS type
                    FROM supply.egon_power_plants_hydro
                )
            )
            AS generators
            GROUP BY generators.bus_id
            ORDER BY count DESC
        )
        AS
        test
        WHERE test.wind > 1 AND test.pv_open_place > 1 AND test.pv_rooftop > 1 AND test.biomass > 1 AND test.hydro > 1
        """
        df = pd.read_sql_query(
            sql=sql_statement,
            con=session.bind,
            index_col=None,
        )

        return [int(df.bus_id[0])]

    @pytest.fixture
    def lv_load_area(self, session, grid_id):
        print(grid_id)
        df = db_io.get_lv_load_areas(self.nd.orm, session, grid_id[0])
        return df.loc[3479]  # df.loc[3479,"geo_area"]

    def test_import_mvgd(self, session, grid_id):
        print(grid_id)
        df = db_io.get_mv_data(self.nd.orm, session, grid_id)

        assert True

    def test_import_load_area(self, session, grid_id):
        df = db_io.get_lv_load_areas(self.nd.orm, session, grid_id[0])

        assert True

    def test_get_egon_ways(self, session, lv_load_area):
        df = db_io.get_egon_ways(self.nd.orm, session, lv_load_area.geo_area)

        assert True

    def test_get_egon_buildings(self, session, grid_id, lv_load_area):
        df = db_io.get_egon_buildings(self.nd.orm, session, grid_id[0], lv_load_area)

        assert True

    def test_get_res_generators(self, session, grid_id_2):
        df = db_io.get_res_generators(self.nd.orm, session, grid_id_2[0])

        assert True

    def test_get_conv_generators(self, session, grid_id):
        df = db_io.get_conv_generators(self.nd.orm, session, grid_id[0])

        assert True
