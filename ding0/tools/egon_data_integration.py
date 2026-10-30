import logging

logger = logging.getLogger(__name__)

import pandas as pd
import shapely.wkb
from ding0.config.config_lv_grids_osm import get_config_osm
from ding0.tools import config as cfg_ding0
from geoalchemy2.shape import to_shape
from sqlalchemy import Integer, cast, func


def get_scenario(scenario=None):
    """Scenario of the egon-data tables to read; defaults to the config value
    `[input_data_source] scenario`."""
    if scenario is None:
        scenario = cfg_ding0.get("input_data_source", "scenario")
    return scenario


def check_load_area_peak_load(load_area_peak_load, buildings_peak_load, label):
    """Log if the peak load of the buildings differs from the load area's.

    egon-data does not always provide load area peak loads (they are NULL in
    some versions), so missing or zero values are not compared.
    """
    if pd.isna(load_area_peak_load) or load_area_peak_load == 0:
        return
    if not round(load_area_peak_load, -1) == round(buildings_peak_load, -1):
        logger.warning(
            f"Peak load {label} of load area ({load_area_peak_load:.1f} kW) differs "
            f"from the sum of its buildings ({buildings_peak_load:.1f} kW)."
        )


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
    # Load areas are not filtered by their peak load here, as egon-data does
    # not always provide it; the threshold `load_area_threshold` is applied to
    # the peak load of the buildings instead.
    mw2kw = 10**3  # load in database is in MW -> scale to kW
    lv_nominal_voltage = cfg_ding0.get("assumptions", "lv_nominal_voltage")

    # build SQL query
    lv_load_areas_sqla = session.query(
        orm["orm_lv_load_areas"].id.label("id_db"),
        orm["orm_lv_load_areas"].zensus_sum.label("population"),
        # orm["orm_lv_load_areas"].zensus_count.label("zensus_cnt"),
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
        func.ST_Intersects(
            func.ST_GeomFromText(geo_area, get_config_osm("srid")),
            orm["osm_ways_with_segments"].geom,
        )
    )
    df = pd.read_sql(sql=query.statement, con=session.bind, index_col=None)
    return df


def get_egon_residential_buildings(orm, session, subst_id, load_area, scenario=None):
    """
    Get the residential bildings from egon-data. Buildings of orm["egon_map_zensus_mvgd_buildings"],
    filtered by:
        - sector == 'residential',
        - electricity == true,
        - bus_id == subst_id.
    Two queries one for the osm, the other for the synthetic buildings.
    Get the geometries from:
        - osm -> orm["osm_buildings_filtered"]
        - synthetic -> orm["osm_buildings_synthetic"]
    Get the capacity/peak_load from orm["building_peak_loads"].
    """
    logger.debug(
        "Get residential buildings by 'subst_id' and 'load_area' from database."
    )

    scenario = get_scenario(scenario)
    sector = "residential"

    query = (
        session.query(
            # orm["egon_map_zensus_mvgd_buildings"],
            orm["egon_map_zensus_mvgd_buildings"].building_id,
            orm["egon_map_zensus_mvgd_buildings"].sector,
            # orm["osm_buildings_filtered"]
            orm["osm_buildings_filtered"].geom_point.label("geometry"),
            orm["osm_buildings_filtered"].geom_building.label("footprint"),
        )
        .join(
            orm["osm_buildings_filtered"],
            orm["egon_map_zensus_mvgd_buildings"].building_id
            == orm["osm_buildings_filtered"].id,
        )
        .filter(
            orm["egon_map_zensus_mvgd_buildings"].sector == sector,
            # orm["egon_map_zensus_mvgd_buildings"].bus_id == subst_id,
            orm["egon_map_zensus_mvgd_buildings"].electricity == True,
            orm["egon_map_zensus_mvgd_buildings"].osm == True,
            func.ST_Intersects(
                func.ST_GeomFromText(load_area.geo_area, get_config_osm("srid")),
                orm["osm_buildings_filtered"].geom_point,
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
            orm["osm_buildings_synthetic"].geom_building.label("footprint"),
        )
        .join(
            orm["osm_buildings_synthetic"],
            orm["egon_map_zensus_mvgd_buildings"].building_id
            == orm["osm_buildings_synthetic"].id.cast(Integer),
        )
        .filter(
            orm["egon_map_zensus_mvgd_buildings"].sector == sector,
            # orm["egon_map_zensus_mvgd_buildings"].bus_id == subst_id,
            orm["egon_map_zensus_mvgd_buildings"].electricity == True,
            orm["egon_map_zensus_mvgd_buildings"].osm == False,
            func.ST_Intersects(
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
    residential_buildings_df["footprint"] = residential_buildings_df["footprint"].apply(
        to_shape
    )

    if residential_buildings_df["building_id"].duplicated(keep=False).any():
        raise ValueError(
            "There are duplicated building_ids, for residential osm and synthetic buildings."
        )

    query = session.query(
        orm["building_peak_loads"].building_id,
        (orm["building_peak_loads"].peak_load_in_w / 1000).label("capacity"),
    ).filter(
        orm["building_peak_loads"].sector == "residential",
        orm["building_peak_loads"].scenario == scenario,
        orm["building_peak_loads"].building_id.in_(
            residential_buildings_df["building_id"].values
        ),
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
    check_load_area_peak_load(
        load_area.peak_load_residential,
        residential_buildings_df.capacity.sum(),
        "residential",
    )

    query = (
        session.query(
            orm["household_electricity_profile"].building_id.label("building_id"),
            func.count(orm["household_electricity_profile"].building_id).label(
                "number_households"
            ),
        )
        .filter(
            orm["household_electricity_profile"].building_id.in_(
                residential_buildings_df["building_id"].values
            )
        )
        .group_by(
            orm["household_electricity_profile"].building_id,
        )
    )

    apartments_per_building_df = pd.read_sql(
        sql=query.statement, con=query.session.bind, index_col=None
    )

    residential_buildings_df = pd.merge(
        left=residential_buildings_df,
        right=apartments_per_building_df,
        left_on="building_id",
        right_on="building_id",
        how="left",
    )

    return residential_buildings_df


def get_egon_cts_buildings(orm, session, subst_id, load_area, scenario=None):
    logger.debug("Get cts buildings by 'subst_id' and 'load_area' from database.")

    scenario = get_scenario(scenario)
    sector = "cts"

    query = (
        session.query(
            # orm["egon_map_zensus_mvgd_buildings"],
            orm["egon_map_zensus_mvgd_buildings"].building_id,
            orm["egon_map_zensus_mvgd_buildings"].sector,
            # orm["osm_buildings_filtered"]
            orm["osm_buildings_filtered"].geom_point.label("geometry"),
            orm["osm_buildings_filtered"].geom_building.label("footprint"),
        )
        .join(
            orm["osm_buildings_filtered"],
            orm["egon_map_zensus_mvgd_buildings"].building_id
            == orm["osm_buildings_filtered"].id,
        )
        .filter(
            orm["egon_map_zensus_mvgd_buildings"].sector == sector,
            # orm["egon_map_zensus_mvgd_buildings"].bus_id == subst_id,
            orm["egon_map_zensus_mvgd_buildings"].electricity == True,
            orm["egon_map_zensus_mvgd_buildings"].osm == True,
            func.ST_Intersects(
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
            orm["osm_buildings_synthetic"].geom_building.label("footprint"),
        )
        .join(
            orm["osm_buildings_synthetic"],
            orm["egon_map_zensus_mvgd_buildings"].building_id
            == orm["osm_buildings_synthetic"].id.cast(Integer),
        )
        .filter(
            orm["egon_map_zensus_mvgd_buildings"].sector == sector,
            # orm["egon_map_zensus_mvgd_buildings"].bus_id == subst_id,
            orm["egon_map_zensus_mvgd_buildings"].electricity == True,
            orm["egon_map_zensus_mvgd_buildings"].osm == False,
            func.ST_Intersects(
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
    cts_buildings_df["footprint"] = cts_buildings_df["footprint"].apply(to_shape)

    if cts_buildings_df["building_id"].duplicated(keep=False).any():
        raise ValueError(
            "There are duplicated building_ids, for residential osm and synthetic buildings."
        )

    # retrieve peak load of cts
    cells_query = session.query(
        orm["building_peak_loads"].building_id,
        (orm["building_peak_loads"].peak_load_in_w / 1000).label("capacity"),
    ).filter(
        orm["building_peak_loads"].scenario == scenario,
        orm["building_peak_loads"].sector == "cts",
        orm["building_peak_loads"].building_id.in_(
            cts_buildings_df["building_id"].values
        ),
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
    check_load_area_peak_load(
        load_area.peak_load_cts, cts_buildings_df.capacity.sum(), "cts"
    )

    return cts_buildings_df


def get_egon_industrial_buildings(orm, session, subst_id, load_area, scenario=None):
    logger.debug(
        "Get industrial buildings by 'subst_id' and 'load_area' from database."
    )
    # Industrial loads 1
    # demand.egon_sites_ind_load_curves_individual, geom from demand.egon_industrial_sites
    # Filter: voltage level, scenario, subst_id
    scn_name = get_scenario(scenario)
    mw2kw = 10**3
    query = (
        session.query(
            # MW, converted to kW for both industrial sources below
            orm["egon_sites_ind_load_curves_individual"].peak_load.label("capacity"),
            func.ST_AsText(
                func.ST_Transform(orm["egon_industrial_sites"].geom, 3035)
            ).label("geometry"),
            func.ST_AsText(
                func.ST_Transform(orm["egon_industrial_sites"].geom, 3035)
            ).label("footprint"),
            orm["egon_sites_ind_load_curves_individual"].site_id,
        )
        .join(
            orm["egon_sites_ind_load_curves_individual"],
            orm["egon_sites_ind_load_curves_individual"].site_id
            == orm["egon_industrial_sites"].id,
        )
        .filter(
            # orm["egon_sites_ind_load_curves_individual"].bus_id == subst_id,
            orm["egon_sites_ind_load_curves_individual"].scn_name == scn_name,
            orm["egon_sites_ind_load_curves_individual"].voltage_level.in_(
                [4, 5, 6, 7]
            ),
            func.ST_Intersects(
                func.ST_GeomFromText(load_area.geo_area, get_config_osm("srid")),
                func.ST_Transform(orm["egon_industrial_sites"].geom, 3035),
            ),
        )
    )
    industrial_loads_1_df = pd.read_sql(
        sql=query.statement, con=session.bind, index_col=None
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
            func.ST_AsText(orm["osm_landuse"].geom).label("footprint"),
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
            func.ST_Intersects(
                func.ST_GeomFromText(load_area.geo_area, get_config_osm("srid")),
                orm["osm_landuse"].geom,
            ),
        )
    )
    industrial_loads_2_df = pd.read_sql(
        sql=query.statement, con=session.bind, index_col=None
    )

    industrial_loads_1_df["site_id"] = industrial_loads_1_df["site_id"].astype(int)
    industrial_loads_1_df["building_id"] = industrial_loads_1_df["site_id"]
    industrial_loads_2_df["osm_id"] = industrial_loads_2_df["osm_id"].astype(int)
    industrial_loads_2_df["building_id"] = industrial_loads_2_df["osm_id"]

    industrial_buildings_df = pd.concat(
        [industrial_loads_1_df, industrial_loads_2_df], ignore_index=True
    )
    industrial_buildings_df = industrial_buildings_df.rename(
        columns={"site_id": "industrial_site_id", "osm_id": "industrial_osm_id"}
    )
    industrial_buildings_df["capacity"] = industrial_buildings_df["capacity"] * mw2kw
    industrial_buildings_df["geometry"] = industrial_buildings_df["geometry"].apply(
        shapely.wkt.loads
    )
    industrial_buildings_df["footprint"] = industrial_buildings_df["footprint"].apply(shapely.wkt.loads)

    industrial_buildings_df["sector"] = "industrial"
    check_load_area_peak_load(
        load_area.peak_load_industrial,
        industrial_buildings_df.capacity.sum(),
        "industrial",
    )

    return industrial_buildings_df


def get_egon_buildings(orm, session, subst_id, load_area, scenario=None):
    logger.info("Get buildings by 'subst_id' and 'load_area' from database.")

    residential_buildings_df = get_egon_residential_buildings(
        orm, session, subst_id, load_area, scenario=scenario
    )
    cts_buildings_df = get_egon_cts_buildings(
        orm, session, subst_id, load_area, scenario=scenario
    )
    industrial_buildings_df = get_egon_industrial_buildings(
        orm, session, subst_id, load_area, scenario=scenario
    )

    residential_buildings_df.drop(columns="sector", inplace=True)
    residential_buildings_df.rename(
        columns={
            "capacity": "residential_capacity",
            "geometry": "residential_geometry",
            "footprint": "residential_footprint",
        },
        inplace=True,
    )
    cts_buildings_df.drop(columns="sector", inplace=True)
    cts_buildings_df.rename(
        columns={
            "capacity": "cts_capacity",
            "geometry": "cts_geometry",
            "footprint": "cts_footprint",
        },
        inplace=True,
    )
    industrial_buildings_df.drop(
        columns=["sector", "industrial_site_id", "industrial_osm_id"], inplace=True
    )
    industrial_buildings_df.rename(
        columns={
            "capacity": "industrial_capacity",
            "geometry": "industrial_geometry",
            "footprint": "industrial_footprint",
        },
        inplace=True,
    )

    buildings_df = pd.merge(
        left=residential_buildings_df,
        right=cts_buildings_df,
        left_on="building_id",
        right_on="building_id",
        how="outer",
    )

    buildings_df = pd.merge(
        left=buildings_df,
        right=industrial_buildings_df,
        left_on="building_id",
        right_on="building_id",
        how="outer",
    )

    if buildings_df.empty:
        logger.error(f"No buildings in LoadArea {load_area.name}")
        return buildings_df

    buildings_df.fillna(
        {
            "number_households": 0,
            "residential_capacity": 0,
            "cts_capacity": 0,
            "industrial_capacity": 0,
        },
        inplace=True,
    )
    buildings_df = buildings_df.astype({"building_id": int, "number_households": int})

    columns_to_sum = ["residential_capacity", "cts_capacity", "industrial_capacity"]
    buildings_df["capacity"] = buildings_df[columns_to_sum].sum(axis=1)

    def reduce_geometry_columns(x):
        # Check if all coordinates are the same
        coordinates = x[x.notna()]
        for i in range(1, coordinates.size):
            if not coordinates.iat[i - 1].equals_exact(coordinates.iat[i], 0.5 * 10**-6):
                logger.error("Coordinates of buildings with the same id are not equal!")

        return coordinates.iat[0]

    geometry_columns = ["residential_geometry", "cts_geometry", "industrial_geometry"]
    buildings_df["geometry"] = buildings_df[geometry_columns].apply(
        reduce_geometry_columns, axis="columns"
    )
    buildings_df.drop(columns=geometry_columns, inplace=True)

    footprint_columns = [
        "residential_footprint",
        "cts_footprint",
        "industrial_footprint",
    ]
    buildings_df["footprint"] = buildings_df[footprint_columns].apply(
        reduce_geometry_columns, axis="columns"
    )
    buildings_df.drop(columns=footprint_columns, inplace=True)

    check_load_area_peak_load(load_area.peak_load, buildings_df.capacity.sum(), "total")
    buildings_df.set_index("building_id", inplace=True)

    return buildings_df


def func_within(geom_a, geom_b, srid=3035):
    return func.ST_Within(
        func.ST_Transform(
            geom_a,
            srid,
        ),
        func.ST_Transform(
            geom_b,
            srid,
        ),
    )


def func_intersect(geom_a, geom_b, srid=3035):
    return func.ST_Intersects(
        func.ST_Transform(
            geom_a,
            srid,
        ),
        func.ST_Transform(
            geom_b,
            srid,
        ),
    )


def get_res_generators(orm, session, mv_grid_district, scenario=None):
    srid = 3035  # new to calc distance matrix in step 6
    subst_id = str(mv_grid_district.id_db)
    geo_area = mv_grid_district.geo_data

    # Get PV open space join weather cell id
    query = (
        session.query(
            orm["generators_pv"].bus_id,
            orm["generators_pv"].gens_id,
            (orm["generators_pv"].capacity * 1000).label("electrical_capacity"),
            orm["generators_pv"].voltage_level,
            orm["weather_cells"].w_id,
            func.ST_AsText(func.ST_Transform(orm["generators_pv"].geom, srid)).label(
                "geom"
            ),
        )
        .join(
            orm["weather_cells"],
            func_within(orm["generators_pv"].geom, orm["weather_cells"].geom),
        )
        .filter(
            # orm["generators_pv"].bus_id == subst_id,
            orm["generators_pv"].site_type == "Freifläche",
            orm["generators_pv"].status == "InBetrieb",
            orm["generators_pv"].voltage_level.in_([4, 5, 6, 7]),
            func.ST_Intersects(
                func.ST_GeomFromText(geo_area.wkt, get_config_osm("srid")),
                func.ST_Transform(orm["generators_pv"].geom, srid),
            ),
        )
    )
    generators_pv_open_space_df = pd.read_sql(
        sql=query.statement, con=session.bind, index_col=None
    )
    generators_pv_open_space_df["generation_type"] = "solar"
    generators_pv_open_space_df["generation_subtype"] = "open_space"

    # Get pv rooftop join geoms
    osm_buildings_filtered = session.query(
        orm["osm_buildings_filtered"].id.cast(Integer).label("building_id"),
        func.ST_AsText(orm["osm_buildings_filtered"].geom_point).label("geom"),
    )
    osm_buildings_synthetic = session.query(
        orm["osm_buildings_synthetic"].id.cast(Integer).label("building_id"),
        func.ST_AsText(orm["osm_buildings_synthetic"].geom_point).label("geom"),
    )
    building_geoms = osm_buildings_filtered.union(osm_buildings_synthetic).subquery(
        name="building_geoms"
    )
    query = (
        session.query(
            orm["generators_pv_rooftop"].bus_id,
            orm["generators_pv_rooftop"].gens_id,
            orm["generators_pv_rooftop"].building_id,
            (orm["generators_pv_rooftop"].capacity * 1000).label("electrical_capacity"),
            orm["generators_pv_rooftop"].voltage_level,
            orm["generators_pv_rooftop"].weather_cell_id.label("w_id"),
            building_geoms.c.geom,
        )
        .join(
            building_geoms,
            orm["generators_pv_rooftop"].building_id == building_geoms.c.building_id,
        )
        .filter(
            # orm["generators_pv_rooftop"].bus_id == subst_id,
            orm["generators_pv_rooftop"].scenario == get_scenario(scenario),
            orm["generators_pv_rooftop"].voltage_level.in_([4, 5, 6, 7]),
            func.ST_Intersects(
                func.ST_GeomFromText(geo_area.wkt, get_config_osm("srid")),
                func.ST_GeomFromText(building_geoms.c.geom, get_config_osm("srid")),
            ),
        )
    )
    generators_pv_rooftop_df = pd.read_sql(
        sql=query.statement, con=session.bind, index_col=None
    )
    generators_pv_rooftop_df["generation_type"] = "solar"
    generators_pv_rooftop_df["generation_subtype"] = "pv_rooftop"

    # Get generators wind join weather cells
    query = (
        session.query(
            orm["generators_wind"].bus_id,
            orm["generators_wind"].gens_id,
            (orm["generators_wind"].capacity * 1000).label("electrical_capacity"),
            orm["generators_wind"].voltage_level,
            orm["weather_cells"].w_id,
            func.ST_AsText(func.ST_Transform(orm["generators_wind"].geom, srid)).label(
                "geom"
            ),
        )
        .join(
            orm["weather_cells"],
            func_within(orm["generators_wind"].geom, orm["weather_cells"].geom),
        )
        .filter(
            # orm["generators_wind"].bus_id == subst_id,
            orm["generators_wind"].site_type == "Windkraft an Land",
            orm["generators_wind"].status == "InBetrieb",
            orm["generators_wind"].voltage_level.in_([4, 5, 6, 7]),
            func.ST_Intersects(
                func.ST_GeomFromText(geo_area.wkt, get_config_osm("srid")),
                func.ST_Transform(orm["generators_wind"].geom, srid),
            ),
        )
    )
    generators_wind_df = pd.read_sql(
        sql=query.statement, con=session.bind, index_col=None
    )
    generators_wind_df["generation_type"] = "wind"

    # Generators biomass
    query = session.query(
        orm["generators_biomass"].bus_id,
        orm["generators_biomass"].gens_id,
        (orm["generators_biomass"].capacity * 1000).label("electrical_capacity"),
        orm["generators_biomass"].voltage_level,
        func.ST_AsText(func.ST_Transform(orm["generators_biomass"].geom, srid)).label(
            "geom"
        ),
    ).filter(
        # orm["generators_biomass"].bus_id == subst_id,
        orm["generators_biomass"].status == "InBetrieb",
        orm["generators_biomass"].voltage_level.in_([4, 5, 6, 7]),
        func.ST_Intersects(
            func.ST_GeomFromText(geo_area.wkt, get_config_osm("srid")),
            func.ST_Transform(orm["generators_biomass"].geom, srid),
        ),
    )
    generators_biomass_df = pd.read_sql(
        sql=query.statement, con=session.bind, index_col=None
    )
    generators_biomass_df["generation_type"] = "biomass"

    # Generators water
    query = session.query(
        orm["generators_water"].bus_id,
        orm["generators_water"].gens_id,
        (orm["generators_water"].capacity * 1000).label("electrical_capacity"),
        orm["generators_water"].voltage_level,
        func.ST_AsText(func.ST_Transform(orm["generators_water"].geom, srid)).label(
            "geom"
        ),
    ).filter(
        # orm["generators_water"].bus_id == subst_id,
        orm["generators_water"].status == "InBetrieb",
        orm["generators_water"].voltage_level.in_([4, 5, 6, 7]),
        func.ST_Intersects(
            func.ST_GeomFromText(geo_area.wkt, get_config_osm("srid")),
            func.ST_Transform(orm["generators_water"].geom, srid),
        ),
    )
    generators_water_df = pd.read_sql(
        sql=query.statement, con=session.bind, index_col=None
    )
    generators_water_df["generation_type"] = "water"

    renewable_generators_df = pd.concat(
        [
            generators_pv_open_space_df,
            generators_pv_rooftop_df,
            generators_wind_df,
            generators_biomass_df,
            generators_water_df,
        ],
        ignore_index=True,
    )
    renewable_generators_df.rename(columns={"bus_id": "subst_id"}, inplace=True)
    # define generators with unknown subtype as 'unknown'
    renewable_generators_df["generation_subtype"] = renewable_generators_df["generation_subtype"].fillna(value="unknown")
    # Overwrite subst_id of the data, to correct faulty data
    renewable_generators_df["subst_id"] = int(subst_id)

    return renewable_generators_df


def get_conv_generators(orm, session, mv_grid_district):
    srid = 3035  # new to calc distance matrix in step 6
    subst_id = str(mv_grid_district.id_db)
    geo_area = mv_grid_district.geo_data

    # Generators combustion
    query = session.query(
        orm["generators_combustion"].bus_id,
        orm["generators_combustion"].gens_id,
        (orm["generators_combustion"].capacity * 1000).label("electrical_capacity"),
        orm["generators_combustion"].voltage_level,
        func.ST_AsText(
            func.ST_Transform(orm["generators_combustion"].geom, srid)
        ).label("geom"),
    ).filter(
        # orm["generators_combustion"].bus_id == subst_id,
        orm["generators_combustion"].status == "InBetrieb",
        orm["generators_combustion"].voltage_level.in_([4, 5, 6, 7]),
        func.ST_Intersects(
            func.ST_GeomFromText(geo_area.wkt, get_config_osm("srid")),
            func.ST_Transform(orm["generators_combustion"].geom, srid),
        ),
    )
    generators_combustion_df = pd.read_sql(
        sql=query.statement, con=session.bind, index_col=None
    )
    generators_combustion_df["generation_type"] = "conventional"
    generators_combustion_df["generation_subtype"] = "combustion"

    # Generators gsgk (Grubengas, Klaerschlamm)
    query = session.query(
        orm["generators_gsgk"].bus_id,
        orm["generators_gsgk"].gens_id,
        (orm["generators_gsgk"].capacity * 1000).label("electrical_capacity"),
        orm["generators_gsgk"].voltage_level,
        func.ST_AsText(func.ST_Transform(orm["generators_gsgk"].geom, srid)).label(
            "geom"
        ),
    ).filter(
        # orm["generators_gsgk"].bus_id == subst_id,
        orm["generators_gsgk"].status == "InBetrieb",
        orm["generators_gsgk"].voltage_level.in_([4, 5, 6, 7]),
        func.ST_Intersects(
            func.ST_GeomFromText(geo_area.wkt, get_config_osm("srid")),
            func.ST_Transform(orm["generators_gsgk"].geom, srid),
        ),
    )
    generators_gsgk_df = pd.read_sql(
        sql=query.statement, con=session.bind, index_col=None
    )
    generators_gsgk_df["generation_type"] = "conventional"
    generators_gsgk_df["generation_subtype"] = "gsgk"

    conventional_generators_df = pd.concat(
        [
            generators_combustion_df,
            generators_gsgk_df,
        ],
        ignore_index=True,
    )
    conventional_generators_df.rename(columns={"bus_id": "subst_id"}, inplace=True)
    # Overwrite subst_id of the data, to correct faulty data
    conventional_generators_df["subst_id"] = int(subst_id)

    return conventional_generators_df


# Components of newer egon-data versions ##################################

# ORM keys of the tables, which are skipped if missing (see import_orm)
COMPONENT_TABLES = (
    "heat_pumps",
    "home_batteries",
    "charging_locations",
    "bus_depots",
    "rail_load_points",
    "etrago_loads",
    "etrago_load_timeseries",
    "storages",
)

# eDisGo sector of charging points by egon-data use case. eDisGo only accepts
# 'home', 'work', 'public' and 'hpc' (worst-case time series).
CHARGING_POINT_SECTORS = {
    "home_detached": "home",
    "home_apartment": "home",
    "work": "work",
    "depot": "work",
    "street": "public",
    "retail": "public",
    "urban_fast": "hpc",
    "highway_fast": "hpc",
    "bus_depot": "hpc",
}
HOME_CHARGING_USE_CASES = ("home_detached", "home_apartment")

BUILDING_COMPONENT_COLUMNS = [
    "heat_pump_p_set",
    "charging_point_p_set",
    "home_storage_p_nom",
    "home_storage_capacity",
]

POINT_COMPONENT_COLUMNS = [
    "component",
    "type",
    "sector",
    "use_case",
    "p_nom",
    "v_level",
    "geometry",
    "source_id",
    "max_hours",
]


def component_enabled(orm, component, tables):
    """Whether a component is switched on in `[components]` and its tables
    exist in the database."""
    return bool(cfg_ding0.get("components", component)) and all(
        table in orm for table in tables
    )


def voltage_level_from_power(p_kw):
    """Voltage level (4 to 7) of a component by its power in kW, with the
    limits of `[components] voltage_level_upper_limit_X` (as in eDisGo)."""
    for level in (7, 6, 5, 4):
        if p_kw <= cfg_ding0.get("components", f"voltage_level_upper_limit_{level}"):
            return level
    return 4


def get_building_components(orm, session, buildings_df, load_area, scenario=None):
    """
    Heat pumps, home batteries and home charging points of the buildings of a
    load area.

    Heat pumps and home batteries are assigned by building id. Home charging
    points (use cases home_detached and home_apartment, voltage level 7) are
    assigned to the nearest building within
    `[components] charging_point_home_max_distance`.

    Parameters
    ----------
    buildings_df : :pandas:`pandas.DataFrame<dataframe>`
        Buildings as returned by :func:`get_egon_buildings`
    load_area : :pandas:`pandas.Series<series>`
        Load area as returned by :func:`get_lv_load_areas`

    Returns
    -------
    components_df : :pandas:`pandas.DataFrame<dataframe>`
        Index building_id, columns heat_pump_p_set, charging_point_p_set,
        home_storage_p_nom (kW) and home_storage_capacity (kWh)
    charging_location_ids : :obj:`list`
        Charging locations assigned to buildings
    """
    scenario = get_scenario(scenario)
    mw2kw = 10**3
    components_df = pd.DataFrame(
        0.0, index=buildings_df.index, columns=BUILDING_COMPONENT_COLUMNS
    )
    charging_location_ids = []
    building_ids = [int(_) for _ in buildings_df.index]
    if not building_ids:
        return components_df, charging_location_ids

    if component_enabled(orm, "heat_pumps", ["heat_pumps"]):
        table = orm["heat_pumps"]
        query = session.query(
            table.building_id, (table.hp_capacity * mw2kw).label("p_set")
        ).filter(table.scenario == scenario, table.building_id.in_(building_ids))
        heat_pumps_df = pd.read_sql(sql=query.statement, con=session.bind)
        heat_pumps = heat_pumps_df.groupby("building_id").p_set.sum()
        components_df.loc[heat_pumps.index, "heat_pump_p_set"] = heat_pumps

    if component_enabled(orm, "storage_units", ["home_batteries"]):
        table = orm["home_batteries"]
        query = session.query(
            table.building_id,
            (table.p_nom * mw2kw).label("p_nom"),
            (table.capacity * mw2kw).label("capacity"),
        ).filter(table.scenario == scenario, table.building_id.in_(building_ids))
        batteries_df = pd.read_sql(sql=query.statement, con=session.bind)
        batteries = batteries_df.groupby("building_id")[["p_nom", "capacity"]].sum()
        components_df.loc[batteries.index, "home_storage_p_nom"] = batteries.p_nom
        components_df.loc[batteries.index, "home_storage_capacity"] = batteries.capacity

    if component_enabled(orm, "charging_points", ["charging_locations"]):
        table = orm["charging_locations"]
        query = session.query(
            table.location_id,
            (table.charging_points * table.average_charging_capacity).label("p_set"),
            func.ST_AsText(table.geom).label("geometry"),
        ).filter(
            table.scenario == scenario,
            table.use_case.in_(HOME_CHARGING_USE_CASES),
            table.voltage_level == 7,
            func.ST_Within(
                table.geom,
                func.ST_GeomFromText(load_area.geo_area, get_config_osm("srid")),
            ),
        )
        charging_df = pd.read_sql(sql=query.statement, con=session.bind)
        charging_df["geometry"] = charging_df.geometry.apply(shapely.wkt.loads)
        assigned = assign_to_nearest_buildings(
            charging_df,
            buildings_df,
            cfg_ding0.get("components", "charging_point_home_max_distance"),
        )
        charging = assigned.groupby("building_id").p_set.sum()
        components_df.loc[charging.index, "charging_point_p_set"] = charging
        charging_location_ids = assigned.location_id.astype(int).tolist()

    return components_df, charging_location_ids


def assign_to_nearest_buildings(locations_df, buildings_df, max_distance):
    """
    Assigns locations to the nearest building within a distance.

    Parameters
    ----------
    locations_df : :pandas:`pandas.DataFrame<dataframe>`
        Column geometry (shapely points, EPSG:3035) and further columns
    buildings_df : :pandas:`pandas.DataFrame<dataframe>`
        Index building_id, column geometry (shapely points, EPSG:3035)
    max_distance : :obj:`float`
        In m

    Returns
    -------
    :pandas:`pandas.DataFrame<dataframe>`
        Assigned locations with column building_id; a location equally near
        to several buildings is assigned to the one with the lowest id
    """
    import geopandas as gpd

    if locations_df.empty or buildings_df.empty:
        return locations_df.assign(building_id=pd.Series(dtype=int)).iloc[0:0]
    srid = get_config_osm("srid")
    locations_gdf = gpd.GeoDataFrame(
        locations_df.reset_index(drop=True).rename_axis("location_index"),
        geometry=list(locations_df.geometry),
        crs=srid,
    )
    buildings_gdf = gpd.GeoDataFrame(
        {"building_id": buildings_df.index},
        geometry=list(buildings_df.geometry),
        crs=srid,
    )
    assigned = gpd.sjoin_nearest(
        locations_gdf, buildings_gdf, max_distance=max_distance, how="inner"
    )
    assigned = (
        assigned.reset_index()
        .sort_values(["location_index", "building_id"])
        .drop_duplicates("location_index")
        .set_index("location_index")
        .sort_index()
    )
    return pd.DataFrame(assigned.drop(columns=["index_right"]))


def add_building_components(buildings_df, components_df):
    """
    Adds the components of buildings as columns and their share for grid
    dimensioning (`[components] *_dimensioning_factor`) to the building's
    `capacity`.

    Parameters
    ----------
    buildings_df : :pandas:`pandas.DataFrame<dataframe>`
    components_df : :pandas:`pandas.DataFrame<dataframe>`
        As returned by :func:`get_building_components`

    Returns
    -------
    :pandas:`pandas.DataFrame<dataframe>`
    """
    buildings_df = buildings_df.join(components_df)
    buildings_df[BUILDING_COMPONENT_COLUMNS] = buildings_df[
        BUILDING_COMPONENT_COLUMNS
    ].fillna(0.0)
    buildings_df["capacity"] = (
        buildings_df["capacity"]
        + cfg_ding0.get("components", "heat_pump_dimensioning_factor")
        * buildings_df["heat_pump_p_set"]
        + cfg_ding0.get("components", "charging_point_home_dimensioning_factor")
        * buildings_df["charging_point_p_set"]
    )
    return buildings_df


def get_point_components(
    orm, session, mv_grid_district, scenario=None, exclude_charging_locations=()
):
    """
    Loads and storage units at locations of their own within an MV grid
    district: charging points, bus depots, rail substations (DC, MV level)
    and battery storage units.

    Parameters
    ----------
    mv_grid_district : :class:`~.ding0.core.structure.regions.MVGridDistrictDing0`
    exclude_charging_locations : :obj:`list`
        Charging locations already assigned to buildings

    Returns
    -------
    :pandas:`pandas.DataFrame<dataframe>`
        Columns component ('load' or 'storage_unit'), type, sector (eDisGo),
        use_case (egon-data), p_nom (kW), v_level, geometry (EPSG:3035),
        source_id and max_hours
    """
    scenario = get_scenario(scenario)
    mw2kw = 10**3
    srid = get_config_osm("srid")
    district = func.ST_GeomFromText(mv_grid_district.geo_data.wkt, srid)
    frames = []

    if component_enabled(orm, "charging_points", ["charging_locations"]):
        table = orm["charging_locations"]
        query = session.query(
            table.location_id,
            table.use_case,
            (table.charging_points * table.average_charging_capacity).label("p_nom"),
            table.voltage_level.label("v_level"),
            func.ST_AsText(table.geom).label("geometry"),
        ).filter(
            table.scenario == scenario,
            table.voltage_level.in_([4, 5, 6, 7]),
            func.ST_Within(table.geom, district),
        )
        df = pd.read_sql(sql=query.statement, con=session.bind)
        df = df.loc[~df.location_id.isin(set(exclude_charging_locations))]
        df["component"] = "load"
        df["type"] = "charging_point"
        df["sector"] = df.use_case.map(CHARGING_POINT_SECTORS)
        df["source_id"] = "charging_location_" + df.location_id.astype(str)
        frames.append(df.drop(columns="location_id"))

    if component_enabled(orm, "bus_depots", ["bus_depots"]):
        table = orm["bus_depots"]
        query = session.query(
            table.depot_id,
            (table.peak_load_mw * mw2kw).label("p_nom"),
            table.voltage_level.label("v_level"),
            func.ST_AsText(table.geom).label("geometry"),
        ).filter(
            table.scenario == scenario,
            table.voltage_level.in_([4, 5, 6, 7]),
            func.ST_Within(table.geom, district),
        )
        df = pd.read_sql(sql=query.statement, con=session.bind)
        df["component"] = "load"
        df["type"] = "charging_point"
        df["use_case"] = "bus_depot"
        df["sector"] = CHARGING_POINT_SECTORS["bus_depot"]
        df["source_id"] = "bus_depot_" + df.depot_id.astype(str)
        frames.append(df.drop(columns="depot_id"))

    if component_enabled(
        orm, "rail", ["rail_load_points", "etrago_loads", "etrago_load_timeseries"]
    ):
        # Substations of trams, underground and S-Bahn (DC, MV level). Their
        # peak load is the maximum of their eTraGo load time series.
        points = orm["rail_load_points"]
        timeseries = orm["etrago_load_timeseries"]
        query = (
            session.query(
                points.load_id,
                points.carrier.label("use_case"),
                func.ST_AsText(points.geom).label("geometry"),
                timeseries.p_set,
            )
            .outerjoin(
                timeseries,
                (timeseries.load_id == points.load_id)
                & (timeseries.scn_name == points.scn_name),
            )
            .filter(
                points.scn_name == scenario,
                points.bus_level == "mv",
                func.ST_Within(points.geom, district),
            )
        )
        df = pd.read_sql(sql=query.statement, con=session.bind)
        missing = df.p_set.isna()
        if missing.any():
            logger.warning(
                f"Rail loads {df.loc[missing, 'load_id'].tolist()} have no time "
                f"series and are skipped."
            )
            df = df.loc[~missing]
        df["p_nom"] = df.p_set.apply(max) * mw2kw
        df["v_level"] = df.p_nom.apply(voltage_level_from_power)
        df["component"] = "load"
        df["type"] = "conventional_load"
        df["sector"] = "rail"
        df["source_id"] = "rail_" + df.load_id.astype(str)
        frames.append(df.drop(columns=["load_id", "p_set"]))

    if component_enabled(orm, "storage_units", ["storages"]):
        # large battery storage; home batteries are assigned to buildings
        table = orm["storages"]
        query = session.query(
            table.id,
            (table.el_capacity * mw2kw).label("p_nom"),
            table.voltage_level.label("v_level"),
            func.ST_AsText(func.ST_Transform(table.geom, srid)).label("geometry"),
        ).filter(
            table.scenario == scenario,
            table.carrier == "BESS",
            table.voltage_level.in_([4, 5, 6, 7]),
            # geom is in EPSG:4326; transforming the district keeps the
            # spatial index usable
            func.ST_Within(table.geom, func.ST_Transform(district, 4326)),
        )
        df = pd.read_sql(sql=query.statement, con=session.bind)
        df["component"] = "storage_unit"
        df["type"] = "large_storage"
        df["use_case"] = "BESS"
        df["sector"] = None
        df["max_hours"] = cfg_ding0.get("components", "storage_max_hours")
        df["source_id"] = "storage_" + df.id.astype(str)
        frames.append(df.drop(columns="id"))

    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=POINT_COMPONENT_COLUMNS)
    components_df = pd.concat(frames, ignore_index=True).reindex(
        columns=POINT_COMPONENT_COLUMNS
    )
    components_df = components_df.loc[components_df.p_nom > 0]
    components_df["v_level"] = components_df.v_level.astype(int)
    components_df["geometry"] = components_df.geometry.apply(shapely.wkt.loads)
    return components_df.reset_index(drop=True)
