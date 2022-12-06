import pandas as pd
from geoalchemy2.shape import to_shape
from sqlalchemy import Integer, cast, func

from ding0.config.config_lv_grids_osm import get_config_osm
from ding0.tools import config as cfg_ding0


def get_mv_data(orm, session, mv_grid_districts_no):
    # build SQL query
    grid_districts = (
        session.query(
            orm["orm_mv_grid_districts"].bus_id,
            func.ST_AsText(orm["orm_mv_grid_districts"].geom).label("poly_geom"),
            func.ST_AsText(orm["orm_mv_stations"].point).label("subs_geom"),
        )
        .join(
            orm["orm_mv_stations"],
            orm["orm_mv_grid_districts"].bus_id == orm["orm_mv_stations"].bus_id,
        )
        .filter(orm["orm_mv_grid_districts"].bus_id.in_(mv_grid_districts_no))
        .filter(orm["version_condition_mvgd"])
        .filter(orm["version_condition_mv_stations"])
        .distinct()
    )

    # read MV data from db
    mv_data = pd.read_sql_query(
        grid_districts.statement, session.bind, index_col="bus_id"
    )

    return mv_data


def get_lv_load_areas(orm, session, mv_grid_district):
    # threshold: load area peak load, if peak load < threshold => disregard
    # load area
    lv_loads_threshold = cfg_ding0.get("mv_routing", "load_area_threshold")
    mw2kw = 10**3  # load in database is in MW -> scale to kW
    lv_nominal_voltage = cfg_ding0.get("assumptions", "lv_nominal_voltage")

    # build SQL query
    lv_load_areas_sqla = (
        session.query(
            orm["orm_lv_load_areas"].id.label("id_db"),
            orm["orm_lv_load_areas"].zensus_sum.label("population"),
            orm["orm_lv_load_areas"].zensus_count.label("zensus_cnt"),
            orm["orm_lv_load_areas"].area_ha.label("area"),
            orm["orm_lv_load_areas"].sector_area_agricultural,
            orm["orm_lv_load_areas"].sector_area_cts,
            orm["orm_lv_load_areas"].sector_area_industrial,
            orm["orm_lv_load_areas"].sector_area_residential,
            orm["orm_lv_load_areas"].sector_share_agricultural,
            orm["orm_lv_load_areas"].sector_share_cts,
            orm["orm_lv_load_areas"].sector_share_industrial,
            orm["orm_lv_load_areas"].sector_share_residential,
            orm["orm_lv_load_areas"].sector_count_agricultural,
            orm["orm_lv_load_areas"].sector_count_cts,
            orm["orm_lv_load_areas"].sector_count_industrial,
            orm["orm_lv_load_areas"].sector_count_residential,
            orm["orm_lv_load_areas"].nuts.label("nuts_code"),
            func.ST_AsText(orm["orm_lv_load_areas"].geom).label("geo_area"),
            func.ST_AsText(orm["orm_lv_load_areas"].geom_centre).label("geo_centre"),
            (orm["orm_lv_load_areas"].sector_peakload_cts * mw2kw).label(
                "peak_load_cts"
            ),
            (orm["orm_lv_load_areas"].sector_peakload_industrial * mw2kw).label(
                "peak_load_industrial"
            ),
            (orm["orm_lv_load_areas"].sector_peakload_residential * mw2kw).label(
                "peak_load_residential"
            ),
            (
                (
                    orm["orm_lv_load_areas"].sector_peakload_cts
                    + orm["orm_lv_load_areas"].sector_peakload_industrial
                    + orm["orm_lv_load_areas"].sector_peakload_residential
                )
                * mw2kw
            ).label("peak_load"),
        )
        .filter(
            orm["orm_lv_load_areas"].bus_id == mv_grid_district.mv_grid._station.id_db
        )
        .filter(
            (
                (
                    orm["orm_lv_load_areas"].sector_peakload_cts
                    + orm["orm_lv_load_areas"].sector_peakload_industrial
                    + orm["orm_lv_load_areas"].sector_peakload_residential
                )
                * mw2kw
            )
            > lv_loads_threshold
        )
        .filter(orm["version_condition_la"])
    )

    # read data from db
    lv_load_areas = pd.read_sql_query(
        lv_load_areas_sqla.statement, session.bind, index_col="id_db"
    )

    return lv_load_areas


def get_egon_ways(orm, session, geo_area):
    """
    retrieve ways
    """
    query = session.query(
        orm["osm_ways_with_segments"].osm_id,
        orm["osm_ways_with_segments"].nodes,
        orm["osm_ways_with_segments"].geom.label("geometry"),
        orm["osm_ways_with_segments"].highway,
        orm["osm_ways_with_segments"].length_segments,
    ).filter(
        func.st_intersects(
            func.ST_GeomFromText(geo_area, get_config_osm("srid")),
            orm["osm_ways_with_segments"].geom,
        )
    )
    df = pd.read_sql(sql=query.statement, con=session.bind, index_col=None)
    return df


def get_egon_residential_buildings(orm, session, geo_area, scenario="eGon2035"):
    """
    retrieve residential buildings
    """

    cells_query = session.query(
        orm["osm_buildings_residential"].id,
        orm["osm_buildings_residential"].osm_id,
        orm["osm_buildings_residential"].geom_building.label("geometry"),
        orm["osm_buildings_residential"].geom_point.label("raccordement_building"),
        orm["osm_buildings_residential"].building.label("category"),
        orm["osm_buildings_residential"].area,
        func.ST_X(orm["osm_buildings_residential"].geom_point).label("x"),
        func.ST_Y(orm["osm_buildings_residential"].geom_point).label("y"),
    ).filter(
        func.st_intersects(
            func.ST_GeomFromText(geo_area, get_config_osm("srid")),
            orm["osm_buildings_residential"].geom_point,
        )
    )

    df_osm_buildings_residential = pd.read_sql(
        sql=cells_query.statement, con=cells_query.session.bind, index_col=None
    )

    # convert wkb-geom to shapely
    df_osm_buildings_residential["geometry"] = df_osm_buildings_residential[
        "geometry"
    ].apply(to_shape)
    df_osm_buildings_residential[
        "raccordement_building"
    ] = df_osm_buildings_residential["raccordement_building"].apply(to_shape)

    df_osm_buildings_residential["category"].replace("yes", "residential", inplace=True)

    # retrieve synthetic buildings
    cells_query = (
        session.query(
            orm["osm_buildings_synthetic"].id,
            orm["osm_buildings_synthetic"].geom_building.label("geometry"),
            orm["osm_buildings_synthetic"].geom_point.label("raccordement_building"),
            orm["osm_buildings_synthetic"].building.label("category"),
            orm["osm_buildings_synthetic"].area,
            func.ST_X(orm["osm_buildings_synthetic"].geom_point).label("x"),
            func.ST_Y(orm["osm_buildings_synthetic"].geom_point).label("y"),
        )
        .filter(orm["osm_buildings_synthetic"].building == "residential")
        .filter(
            func.st_intersects(
                func.ST_GeomFromText(geo_area, get_config_osm("srid")),
                orm["osm_buildings_synthetic"].geom_point,
            )
        )
    )

    df_osm_buildings_synthetic = pd.read_sql(
        sql=cells_query.statement, con=cells_query.session.bind, index_col=None
    )

    # convert wkb-geom to shapely
    df_osm_buildings_synthetic["geometry"] = df_osm_buildings_synthetic[
        "geometry"
    ].apply(to_shape)
    df_osm_buildings_synthetic["raccordement_building"] = df_osm_buildings_synthetic[
        "raccordement_building"
    ].apply(to_shape)

    # TODO: id of synthetic buildings comes in str format -> convert to int in database, workaround:
    df_osm_buildings_synthetic["id"] = df_osm_buildings_synthetic["id"].astype(int)

    # append residential and synthetic buildings
    df_all_buildings_residential = pd.concat(
        [df_osm_buildings_residential, df_osm_buildings_synthetic], ignore_index=True
    )
    if df_all_buildings_residential["id"].duplicated(keep=False).any():
        raise ValueError(
            "There are duplicated building_ids, for residential non-synthetic and synthetic buildings."
        )

    # retrieve number of households in buildings for selected buildings
    cells_query = (
        session.query(
            orm["household_electricity_profile"].building_id.label("id"),
            func.count(orm["household_electricity_profile"].profile_id).label(
                "number_households"
            ),
            # TODO: no status quo peak loads are available, grid are build based on 2035 scenario
            (orm["building_peak_loads"].peak_load_in_w / 1000).label("capacity"),
        )
        .filter(
            orm["household_electricity_profile"].building_id.in_(
                df_all_buildings_residential["id"].values
            )
        )
        # TODO: check if .join() is slower than to create a third separate df with building peak loads
        .join(
            orm["building_peak_loads"],
            orm["household_electricity_profile"].building_id ==
            # TODO: id of building peak loads comes in str format -> convert to int in database, workaround:
            cast(orm["building_peak_loads"].building_id, Integer),
        )
        .filter(orm["building_peak_loads"].sector == "residential")
        .filter(
            # TODO: which scenario should be taken?
            orm["building_peak_loads"].scenario
            == scenario
        )
        .group_by(
            orm["household_electricity_profile"].building_id,
            orm["building_peak_loads"].peak_load_in_w,
        )
    )

    df_apartments_per_building = pd.read_sql(
        sql=cells_query.statement, con=cells_query.session.bind, index_col=None
    )

    df_all_buildings_residential = pd.merge(
        left=df_all_buildings_residential,
        right=df_apartments_per_building,
        left_on="id",
        right_on="id",
        how="left",
    )

    # TODO: do not import buildings with no_households or peak load == NaN, workaround:
    df_all_buildings_residential.dropna(axis=0, inplace=True)
    # set residential category and egon_id to df index
    df_all_buildings_residential["category"] = "residential"

    return df_all_buildings_residential


def get_egon_cts_buildings(orm, session, geo_area, scenario="eGon2035"):

    # retrieve residential buildings
    cells_query = session.query(
        orm["osm_buildings_filtered"].id,
        orm["osm_buildings_filtered"].osm_id,
        orm["osm_buildings_filtered"].geom_building.label("geometry"),
        orm["osm_buildings_filtered"].geom_point.label("raccordement_building"),
        orm["osm_buildings_filtered"].building.label("category"),
        orm["osm_buildings_filtered"].area,
        func.ST_X(orm["osm_buildings_filtered"].geom_point).label("x"),
        func.ST_Y(orm["osm_buildings_filtered"].geom_point).label("y"),
    ).filter(
        func.st_intersects(
            func.ST_GeomFromText(geo_area, get_config_osm("srid")),
            orm["osm_buildings_filtered"].geom_point,
        )
    )

    df_osm_buildings_cts = pd.read_sql(
        sql=cells_query.statement, con=cells_query.session.bind, index_col=None
    )

    # convert wkb-geom to shapely
    df_osm_buildings_cts["geometry"] = df_osm_buildings_cts["geometry"].apply(to_shape)
    df_osm_buildings_cts["raccordement_building"] = df_osm_buildings_cts[
        "raccordement_building"
    ].apply(to_shape)

    df_osm_buildings_cts["category"].replace("yes", "residential", inplace=True)

    # retrieve synthetic buildings
    cells_query = (
        session.query(
            orm["osm_buildings_synthetic"].id,
            orm["osm_buildings_synthetic"].geom_building.label("geometry"),
            orm["osm_buildings_synthetic"].geom_point.label("raccordement_building"),
            orm["osm_buildings_synthetic"].building.label("category"),
            orm["osm_buildings_synthetic"].area,
            func.ST_X(orm["osm_buildings_synthetic"].geom_point).label("x"),
            func.ST_Y(orm["osm_buildings_synthetic"].geom_point).label("y"),
        )
        .filter(orm["osm_buildings_synthetic"].building == "cts")
        .filter(
            func.st_intersects(
                func.ST_GeomFromText(geo_area, get_config_osm("srid")),
                orm["osm_buildings_synthetic"].geom_point,
            )
        )
    )

    df_osm_buildings_synthetic = pd.read_sql(
        sql=cells_query.statement, con=cells_query.session.bind, index_col=None
    )

    # convert wkb-geom to shapely
    df_osm_buildings_synthetic["geometry"] = df_osm_buildings_synthetic[
        "geometry"
    ].apply(to_shape)
    df_osm_buildings_synthetic["raccordement_building"] = df_osm_buildings_synthetic[
        "raccordement_building"
    ].apply(to_shape)

    # TODO: id of synthetic buildings comes in str format -> convert to int in database, workaround:
    df_osm_buildings_synthetic["id"] = df_osm_buildings_synthetic["id"].astype(int)

    # append residential and synthetic buildings
    df_all_buildings = pd.concat(
        [df_osm_buildings_cts, df_osm_buildings_synthetic], ignore_index=True
    )
    if df_all_buildings["id"].duplicated(keep=False).any():
        raise ValueError(
            "There are duplicated building_ids, for cts non-synthetic and synthetic buildings."
        )

    # retrieve peak load of cts
    cells_query = (
        session.query(
            orm["building_peak_loads"].building_id.label("id"),
            (orm["building_peak_loads"].peak_load_in_w / 1000).label("capacity"),
        )
        .filter(orm["building_peak_loads"].scenario == scenario)
        .filter(orm["building_peak_loads"].sector == "cts")
    )

    df_cts_peak_load = pd.read_sql(
        sql=cells_query.statement, con=cells_query.session.bind, index_col=None
    )

    df_all_buildings_cts = pd.merge(
        left=df_all_buildings,
        right=df_cts_peak_load,
        left_on="id",
        right_on="id",
        how="left",
    )

    # TODO: do not import buildings with no_households or peak load == NaN, workaround:
    df_all_buildings_cts.dropna(axis=0, inplace=True)
    # set residential category and egon_id to df index
    df_all_buildings_cts["category"] = "cts"
    df_all_buildings_cts["number_households"] = 0

    return df_all_buildings_cts


def get_res_generators(orm, session, mv_grid_districts_dict):
    srid = 3035  # new to calc distance matrix in step 6
    # build query
    generators_sqla = (
        session.query(
            orm["orm_re_generators"].columns.id,
            orm["orm_re_generators"].columns.subst_id,
            orm["orm_re_generators"].columns.la_id,
            orm["orm_re_generators"].columns.mvlv_subst_id,
            orm["orm_re_generators"].columns.electrical_capacity,
            orm["orm_re_generators"].columns.generation_type,
            orm["orm_re_generators"].columns.generation_subtype,
            orm["orm_re_generators"].columns.voltage_level,
            orm["orm_re_generators"].columns.w_id,
            func.ST_AsText(
                func.ST_Transform(orm["orm_re_generators"].columns.rea_geom_new, srid)
            ).label("geom_new"),
            func.ST_AsText(
                func.ST_Transform(orm["orm_re_generators"].columns.geom, srid)
            ).label("geom"),
        )
        .filter(
            orm["orm_re_generators"].columns.subst_id.in_(list(mv_grid_districts_dict))
        )
        .filter(orm["orm_re_generators"].columns.voltage_level.in_([4, 5, 6, 7]))
        .filter(orm["version_condition_re"])
    )

    # read data from db
    generators = pd.read_sql_query(
        generators_sqla.statement, session.bind, index_col="id"
    )
    return generators


def get_conv_generators(orm, session, mv_grid_districts_dict):
    srid = 3035
    # build query
    generators_sqla = (
        session.query(
            orm["orm_conv_generators"].columns.id,
            orm["orm_conv_generators"].columns.subst_id,
            orm["orm_conv_generators"].columns.name,
            orm["orm_conv_generators"].columns.capacity,
            orm["orm_conv_generators"].columns.fuel,
            orm["orm_conv_generators"].columns.voltage_level,
            func.ST_AsText(
                func.ST_Transform(orm["orm_conv_generators"].columns.geom, srid)
            ).label("geom"),
        )
        .filter(
            orm["orm_conv_generators"].columns.subst_id.in_(
                list(mv_grid_districts_dict)
            )
        )
        .filter(orm["orm_conv_generators"].columns.voltage_level.in_([4, 5, 6]))
        .filter(orm["version_condition_conv"])
    )

    # read data from db
    generators = pd.read_sql_query(
        generators_sqla.statement, session.bind, index_col="id"
    )

    return generators
