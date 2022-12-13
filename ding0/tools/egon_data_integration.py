import logging

logger = logging.getLogger(__name__)

import pandas as pd
import shapely.wkb
from ding0.config.config_lv_grids_osm import get_config_osm
from ding0.tools import config as cfg_ding0
from geoalchemy2.shape import to_shape
from sqlalchemy import Integer, cast, func


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


def get_lv_load_areas(orm, session, mv_grid_id):
    # threshold: load area peak load, if peak load < threshold => disregard
    # load area
    lv_loads_threshold = cfg_ding0.get("mv_routing", "load_area_threshold")
    mw2kw = 10**3  # load in database is in MW -> scale to kW
    lv_nominal_voltage = cfg_ding0.get("assumptions", "lv_nominal_voltage")

    # build SQL query
    lv_load_areas_sqla = session.query(
        orm["orm_lv_load_areas"].id.label("id_db"),
        orm["orm_lv_load_areas"].zensus_sum.label("population"),
        orm["orm_lv_load_areas"].zensus_count.label("zensus_cnt"),
        orm["orm_lv_load_areas"].area_ha.label("area"),
        # orm["orm_lv_load_areas"].sector_area_agricultural,
        # orm["orm_lv_load_areas"].sector_area_cts,
        # orm["orm_lv_load_areas"].sector_area_industrial,
        # orm["orm_lv_load_areas"].sector_area_residential,
        # orm["orm_lv_load_areas"].sector_share_agricultural,
        # orm["orm_lv_load_areas"].sector_share_cts,
        # orm["orm_lv_load_areas"].sector_share_industrial,
        # orm["orm_lv_load_areas"].sector_share_residential,
        # orm["orm_lv_load_areas"].sector_count_agricultural,
        # orm["orm_lv_load_areas"].sector_count_cts,
        # orm["orm_lv_load_areas"].sector_count_industrial,
        # orm["orm_lv_load_areas"].sector_count_residential,
        # orm["orm_lv_load_areas"].nuts.label("nuts_code"),
        func.ST_AsText(orm["orm_lv_load_areas"].geom).label("geo_area"),
        func.ST_AsText(orm["orm_lv_load_areas"].geom_centre).label("geo_centre"),
        orm["orm_lv_load_areas"].sector_peakload_residential.label(
            "peak_load_residential"
        ),
        orm["orm_lv_load_areas"].sector_peakload_cts.label("peak_load_cts"),
        orm["orm_lv_load_areas"].sector_peakload_industrial.label(
            "peak_load_industrial"
        ),
    ).filter(
        orm["orm_lv_load_areas"].bus_id == mv_grid_id,
        orm["version_condition_la"],
    )

    # read data from db
    lv_load_areas = pd.read_sql_query(
        lv_load_areas_sqla.statement, session.bind, index_col="id_db"
    )
    peak_load_columns = [
        "peak_load_residential",
        "peak_load_cts",
        "peak_load_industrial",
    ]
    lv_load_areas[peak_load_columns] = lv_load_areas[peak_load_columns].fillna(
        value=0.0
    )
    lv_load_areas[peak_load_columns] = lv_load_areas[peak_load_columns] * mw2kw
    lv_load_areas["peak_load"] = lv_load_areas[peak_load_columns].sum(axis="columns")
    lv_load_areas = lv_load_areas[lv_load_areas["peak_load"] > lv_loads_threshold]
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


def get_egon_residential_buildings(orm, session, subst_id, load_area):
    """
    Get the residential bildings from egon-data. Buildings of orm["egon_map_zensus_mvgd_buildings"],
    filtered by:
        - sector == 'residential',
        - electricity == true,
        - bus_id == subst_id.
    Two queries one for the osm, the other for the synthetic buildings.
    Get the geometries from:
        - osm -> orm["osm_buildings_residential"]
        - synthetic -> orm["osm_buildings_synthetic"]
    Get the capacity/peak_load of the buildings from a join of orm["household_electricity_profile"] and
    orm["building_peak_loads"].
    """
    logger.debug("Get residential buildings by 'subst_id' from database.")

    # TODO: which scenario should be taken?
    scenario = "eGon2035"
    sector = "residential"

    query = (
        session.query(
            # orm["egon_map_zensus_mvgd_buildings"],
            orm["egon_map_zensus_mvgd_buildings"].building_id,
            orm["egon_map_zensus_mvgd_buildings"].sector,
            # orm["osm_buildings_residential"]
            orm["osm_buildings_residential"].geom_point.label("geometry"),
        )
        .join(
            orm["osm_buildings_residential"],
            orm["egon_map_zensus_mvgd_buildings"].building_id
            == orm["osm_buildings_residential"].id,
        )
        .filter(
            orm["egon_map_zensus_mvgd_buildings"].sector == sector,
            orm["egon_map_zensus_mvgd_buildings"].bus_id == subst_id,
            orm["egon_map_zensus_mvgd_buildings"].electricity == True,
            orm["egon_map_zensus_mvgd_buildings"].osm == True,
            func.st_intersects(
                func.ST_GeomFromText(load_area.geo_area, get_config_osm("srid")),
                orm["osm_buildings_residential"].geom_point,
            ),
        )
    )
    residential_buildings_osm_df = pd.read_sql(
        sql=query.statement, con=query.session.bind, index_col=None
    )

    query = (
        session.query(
            # orm["egon_map_zensus_mvgd_buildings"],
            orm["egon_map_zensus_mvgd_buildings"].building_id,
            orm["egon_map_zensus_mvgd_buildings"].sector,
            # orm["osm_buildings_synthetic"]
            orm["osm_buildings_synthetic"].geom_point.label("geometry"),
        )
        .join(
            orm["osm_buildings_synthetic"],
            orm["egon_map_zensus_mvgd_buildings"].building_id
            == orm["osm_buildings_synthetic"].id.cast(Integer),
        )
        .filter(
            orm["egon_map_zensus_mvgd_buildings"].sector == sector,
            orm["egon_map_zensus_mvgd_buildings"].bus_id == subst_id,
            orm["egon_map_zensus_mvgd_buildings"].electricity == True,
            orm["egon_map_zensus_mvgd_buildings"].osm == False,
            func.st_intersects(
                func.ST_GeomFromText(load_area.geo_area, get_config_osm("srid")),
                orm["osm_buildings_synthetic"].geom_point,
            ),
        )
    )
    residential_buildings_synthetic_df = pd.read_sql(
        sql=query.statement, con=query.session.bind, index_col=None
    )

    residential_buildings_df = pd.concat(
        [residential_buildings_osm_df, residential_buildings_synthetic_df],
        ignore_index=True,
    )
    residential_buildings_df["geometry"] = residential_buildings_df["geometry"].apply(
        to_shape
    )

    if residential_buildings_df["building_id"].duplicated(keep=False).any():
        raise ValueError(
            "There are duplicated building_ids, for residential osm and synthetic buildings."
        )

    # Retrieve number of households in buildings for selected buildings.
    query = (
        session.query(
            orm["household_electricity_profile"].building_id,
            # func.count(orm["household_electricity_profile"].profile_id).label(
            #     "number_of_households"
            # ),
            func.sum(orm["building_peak_loads"].peak_load_in_w / 1000).label(
                "capacity"
            ),
        )
        .join(
            orm["building_peak_loads"],
            orm["household_electricity_profile"].building_id
            == orm["building_peak_loads"].building_id.cast(Integer),
        )
        .filter(
            orm["building_peak_loads"].sector == "residential",
            orm["building_peak_loads"].scenario == scenario,
            orm["household_electricity_profile"].building_id.in_(
                residential_buildings_df["building_id"].values
            ),
        )
        .group_by(
            orm["household_electricity_profile"].building_id,
        )
    )

    capacity_per_building_df = pd.read_sql(
        sql=query.statement, con=query.session.bind, index_col=None
    )

    residential_buildings_df = pd.merge(
        left=residential_buildings_df,
        right=capacity_per_building_df,
        left_on="building_id",
        right_on="building_id",
        how="left",
    )

    if not load_area.peak_load_residential == residential_buildings_df.capacity.sum():
        logger.error(
            f"{load_area.peak_load_residential=} != {residential_buildings_df.capacity.sum()=}"
        )

    return residential_buildings_df


def get_egon_cts_buildings(orm, session, subst_id, load_area):
    logger.debug("Get cts buildings by 'subst_id' from database.")

    # TODO: which scenario should be taken?
    scenario = "eGon2035"
    sector = "cts"

    query = (
        session.query(
            # orm["egon_map_zensus_mvgd_buildings"],
            orm["egon_map_zensus_mvgd_buildings"].building_id,
            orm["egon_map_zensus_mvgd_buildings"].sector,
            # orm["osm_buildings_residential"]
            orm["osm_buildings_filtered"].geom_point.label("geometry"),
        )
        .join(
            orm["osm_buildings_filtered"],
            orm["egon_map_zensus_mvgd_buildings"].building_id
            == orm["osm_buildings_filtered"].id,
        )
        .filter(
            orm["egon_map_zensus_mvgd_buildings"].sector == sector,
            orm["egon_map_zensus_mvgd_buildings"].bus_id == subst_id,
            orm["egon_map_zensus_mvgd_buildings"].electricity == True,
            orm["egon_map_zensus_mvgd_buildings"].osm == True,
            func.st_intersects(
                func.ST_GeomFromText(load_area.geo_area, get_config_osm("srid")),
                orm["osm_buildings_filtered"].geom_point,
            ),
        )
    )
    cts_buildings_osm_df = pd.read_sql(
        sql=query.statement, con=query.session.bind, index_col=None
    )

    query = (
        session.query(
            # orm["egon_map_zensus_mvgd_buildings"],
            orm["egon_map_zensus_mvgd_buildings"].building_id,
            orm["egon_map_zensus_mvgd_buildings"].sector,
            # orm["osm_buildings_synthetic"]
            orm["osm_buildings_synthetic"].geom_point.label("geometry"),
        )
        .join(
            orm["osm_buildings_synthetic"],
            orm["egon_map_zensus_mvgd_buildings"].building_id
            == orm["osm_buildings_synthetic"].id.cast(Integer),
        )
        .filter(
            orm["egon_map_zensus_mvgd_buildings"].sector == sector,
            orm["egon_map_zensus_mvgd_buildings"].bus_id == subst_id,
            orm["egon_map_zensus_mvgd_buildings"].electricity == True,
            orm["egon_map_zensus_mvgd_buildings"].osm == False,
            func.st_intersects(
                func.ST_GeomFromText(load_area.geo_area, get_config_osm("srid")),
                orm["osm_buildings_synthetic"].geom_point,
            ),
        )
    )
    cts_buildings_synthetic_df = pd.read_sql(
        sql=query.statement, con=query.session.bind, index_col=None
    )

    cts_buildings_df = pd.concat(
        [cts_buildings_osm_df, cts_buildings_synthetic_df],
        ignore_index=True,
    )
    cts_buildings_df["geometry"] = cts_buildings_df["geometry"].apply(to_shape)

    if cts_buildings_df["building_id"].duplicated(keep=False).any():
        raise ValueError(
            "There are duplicated building_ids, for residential osm and synthetic buildings."
        )

    # retrieve peak load of cts
    cells_query = (
        session.query(
            orm["building_peak_loads"].building_id,
            (orm["building_peak_loads"].peak_load_in_w / 1000).label("capacity"),
        )
        .filter(orm["building_peak_loads"].scenario == scenario)
        .filter(orm["building_peak_loads"].sector == "cts")
    )
    capacity_per_building_df = pd.read_sql(
        sql=cells_query.statement, con=cells_query.session.bind, index_col=None
    )

    cts_buildings_df = pd.merge(
        left=cts_buildings_df,
        right=capacity_per_building_df,
        left_on="building_id",
        right_on="building_id",
        how="left",
    )

    if not load_area.peak_load_cts == cts_buildings_df.capacity.sum():
        logger.error(
            f"{load_area.peak_load_cts=} != {cts_buildings_df.capacity.sum()=}"
        )

    return cts_buildings_df


def get_egon_industrial_buildings(orm, session, subst_id, load_area):
    logger.debug("Get industrial buildings by 'subst_id' from database.")
    # Industrial loads 1
    # demand.egon_sites_ind_load_curves_individual, geom from demand.egon_industrial_sites
    # Filter: voltage level, scenario, subst_id
    scn_name = "eGon2035"
    mw2kw = 10**3
    query = (
        session.query(
            (orm["egon_sites_ind_load_curves_individual"].peak_load * mw2kw).label(
                "capacity"
            ),
            func.ST_AsText(
                func.ST_Transform(orm["egon_industrial_sites"].geom, 3035)
            ).label("geometry"),
            orm["egon_sites_ind_load_curves_individual"].site_id,
        )
        .join(
            orm["egon_sites_ind_load_curves_individual"],
            orm["egon_sites_ind_load_curves_individual"].site_id
            == orm["egon_industrial_sites"].id,
        )
        .filter(
            orm["egon_sites_ind_load_curves_individual"].bus_id == subst_id,
            orm["egon_sites_ind_load_curves_individual"].scn_name == scn_name,
            orm["egon_sites_ind_load_curves_individual"].voltage_level.in_(
                [4, 5, 6, 7]
            ),
            func.st_intersects(
                func.ST_GeomFromText(load_area.geo_area, get_config_osm("srid")),
                func.ST_Transform(orm["egon_industrial_sites"].geom, 3035),
            ),
        )
    )
    industrial_loads_1_df = pd.read_sql(
        sql=query.statement, con=session.bind, index_col=None
    )
    industrial_loads_1_df["geometry"] = industrial_loads_1_df["geometry"].apply(
        shapely.wkt.loads
    )

    # Industrial loads 2
    # demand.egon_osm_ind_load_curves_individual, geom from openstreetmap.osm_landuse
    # Filter: voltage level, scenario, subst_id

    query = (
        session.query(
            orm["egon_osm_ind_load_curves_individual"].peak_load.label("capacity"),
            func.ST_AsText(func.ST_PointOnSurface(orm["osm_landuse"].geom)).label(
                "geometry"
            ),
            orm["egon_osm_ind_load_curves_individual"].osm_id,
        )
        .join(
            orm["osm_landuse"],
            orm["egon_osm_ind_load_curves_individual"].osm_id == orm["osm_landuse"].id,
        )
        .filter(
            orm["egon_osm_ind_load_curves_individual"].bus_id == subst_id,
            orm["egon_osm_ind_load_curves_individual"].scn_name == scn_name,
            orm["egon_osm_ind_load_curves_individual"].voltage_level.in_([4, 5, 6, 7]),
            func.st_intersects(
                func.ST_GeomFromText(load_area.geo_area, get_config_osm("srid")),
                orm["osm_landuse"].geom,
            ),
        )
    )
    industrial_loads_2_df = pd.read_sql(
        sql=query.statement, con=session.bind, index_col=None
    )
    industrial_loads_2_df["geometry"] = industrial_loads_2_df["geometry"].apply(
        shapely.wkt.loads
    )

    industrial_buildings_df = pd.concat(
        [industrial_loads_1_df, industrial_loads_2_df], ignore_index=True
    )
    industrial_buildings_df = industrial_buildings_df.rename(
        columns={"site_id": "industrial_site_id", "osm_id": "industrial_osm_id"}
    )

    if not load_area.peak_load_industrial == industrial_buildings_df.capacity.sum():
        logger.error(
            f"{load_area.peak_load_industrial=} != {industrial_buildings_df.capacity.sum()=}"
        )
    return industrial_buildings_df



def get_egon_buildings(orm, session, subst_id, load_area):
    logger.info("Get buildings by 'subst_id' from database.")
    peak_load = load_area.peak_load
    residential_buildings_df = get_egon_residential_buildings(
        orm, session, subst_id, load_area
    )
    cts_buildings_df = get_egon_cts_buildings(orm, session, subst_id, load_area)

    buildings_w_loads_df = pd.concat(
        [residential_buildings_df, cts_buildings_df], ignore_index=True
    )
    # sum capacity of buildings with the same id
    if buildings_w_loads_df["building_id"].duplicated(keep=False).any():

        def sum_capacity(x):
            y = x.head(1)
            if x.shape[0] > 1:
                y["sector"] = "mixed_residential_cts"
            y["capacity"] = x["capacity"].sum()
            return y

        buildings_w_loads_df = buildings_w_loads_df.groupby(
            ["building_id"], axis="rows", as_index=False
        ).apply(sum_capacity)
        logger.warning("There is a building_id which have cts and residential.")
    if buildings_w_loads_df["building_id"].duplicated(keep=False).any():
        raise ValueError(
            "There are duplicated building_ids, " "for residential and cts buildings"
        )

    industrial_loads_df = get_egon_industrial_buildings(
        orm, session, subst_id, load_area
    )
    industrial_loads_df["sector"] = "industrial"
    buildings_w_loads_df = pd.concat(
        [buildings_w_loads_df, industrial_loads_df], ignore_index=True
    )
    if buildings_w_loads_df.empty:
        raise ValueError("There are no buildings in the LoadArea.")
    if not load_area.peak_load == buildings_w_loads_df.capacity.sum():
        logger.error(
            f"{load_area.peak_load=} != {buildings_w_loads_df.capacity.sum()=}"
        )
    return buildings_w_loads_df


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
