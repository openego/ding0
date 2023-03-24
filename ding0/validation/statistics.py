import json
import logging
import os

import ding0.tools.egon_data_integration as db_io
import pandas as pd
import pyproj
from edisgo import EDisGo
from edisgo.edisgo import import_edisgo_from_files
from edisgo.network.grids import Grid, LVGrid, MVGrid
from shapely import wkt
from shapely.geometry import Point
from shapely.ops import transform
from shapely.wkt import loads as wkt_loads

proj_3035_to_4326 = pyproj.Transformer.from_crs(3035, 4326, always_xy=True).transform

logger = logging.getLogger(__name__)


class MockMVGridDistrict:
    def __init__(self, mv_data):
        self.id_db = mv_data.index.values[0]
        self.geo_data = wkt_loads(mv_data.loc[self.id_db, "poly_geom"])


class GridStats:
    def __init__(self, **kwargs):

        self.grid_id = kwargs.get("grid_id", None)
        # grid_type: MV_Grid_District, MV_Grid, LV_Grid
        self.grid_type = kwargs.get("grid_type", None)
        self.mv_grid_id = kwargs.get("mv_grid_id", None)
        self.lv_grid_id = kwargs.get("lv_grid_id", None)
        # source: db, file
        self.source = kwargs.get("source", None)
        self.suffix = kwargs.get("suffix", None)

        self.geom_grid_district = kwargs.get("geom_grid_district", None)
        self.geom_substation = kwargs.get("geom_substation", None)
        self.population = kwargs.get("population", None)

        self.integrity_check_messages = kwargs.get("integrity_check_messages", None)
        self.v_nom = kwargs.get("v_nom", None)

        self.n_buses = kwargs.get("n_buses", None)

        self.n_lines = kwargs.get("n_lines", None)
        self.n_types_lines = kwargs.get("n_types_lines", None)
        self.l_lines = kwargs.get("l_lines", None)
        self.l_types_lines = kwargs.get("l_types_lines", None)

        self.n_transformers = kwargs.get("n_transformers", None)
        self.n_types_transformers = kwargs.get("n_types_transformers", None)

        self.n_feeders = kwargs.get("n_feeders", None)
        self.l_feeders = kwargs.get("l_feeders", None)
        self.max_feeder_length = kwargs.get("max_feeder_length", None)

        self.n_loadareas = kwargs.get("n_loadareas", None)
        self.p_loadareas_residential = kwargs.get("p_loadareas_residential", None)
        self.p_loadareas_cts = kwargs.get("p_loadareas_cts", None)
        self.p_loadareas_industrial = kwargs.get("p_loadareas_industrial", None)
        self.p_loadareas_total = kwargs.get("p_loadareas_total", None)

        self.n_loads_residential = kwargs.get("n_loads_residential", None)
        self.n_households = kwargs.get("n_households", None)
        self.p_loads_residential = kwargs.get("p_loads_residential", None)

        self.n_loads_industrial = kwargs.get("n_loads_industrial", None)
        self.p_loads_industrial = kwargs.get("p_loads_industrial", None)
        self.n_loads_cts = kwargs.get("n_loads_cts", None)
        self.p_loads_cts = kwargs.get("p_loads_cts", None)
        self.n_loads_total = kwargs.get("n_loads_total", None)
        self.p_loads_total = kwargs.get("p_loads_total", None)

        self.n_gens_pv_openspace = kwargs.get("n_gens_pv_openspace", None)
        self.n_gens_pv_rooftop = kwargs.get("n_gens_pv_rooftop", None)
        self.n_gens_wind = kwargs.get("n_gens_wind", None)
        self.n_gens_biomass = kwargs.get("n_gens_biomass", None)
        self.n_gens_water = kwargs.get("n_gens_water", None)
        self.n_gens_conventional = kwargs.get("n_gens_conventional", None)
        self.p_gens_pv_openspace = kwargs.get("p_gens_pv_openspace", None)
        self.p_gens_pv_rooftop = kwargs.get("p_gens_pv_rooftop", None)
        self.p_gens_wind = kwargs.get("p_gens_wind", None)
        self.p_gens_biomass = kwargs.get("p_gens_biomass", None)
        self.p_gens_water = kwargs.get("p_gens_water", None)
        self.p_gens_conventional = kwargs.get("p_gens_conventional", None)

        self.n_gens_renewable_total = kwargs.get("n_gens_renewable_total", None)
        self.p_gens_renewable_total = kwargs.get("p_gens_renewable_total", None)
        self.n_gens_conventional_total = kwargs.get("n_gens_conventional_total", None)
        self.p_gens_conventional_total = kwargs.get("p_gens_conventional_total", None)
        self.n_gens_total = kwargs.get("n_gens_total", None)
        self.p_gens_total = kwargs.get("p_gens_total", None)

    def update_from_db(self, orm, session, grid_id):
        logger.info("Update GridStats object from database.")
        self.source = "db"

        # MV Grid District Data
        mv_data = db_io.get_mv_data(orm, session, [grid_id])
        mv_grid_district = MockMVGridDistrict(mv_data)

        mv_data.loc[grid_id, "poly_geom"] = transform(
            proj_3035_to_4326, wkt.loads(mv_data.loc[grid_id, "poly_geom"])
        ).wkt

        self.grid_id = grid_id
        self.geom_grid_district = mv_data.loc[grid_id, "poly_geom"]
        self.geom_substation = mv_data.loc[grid_id, "subs_geom"]

        # LV Load Area Data
        lv_load_areas = db_io.get_lv_load_areas(orm, session, grid_id)

        self.population = lv_load_areas["population"].sum()
        self.n_loadareas = lv_load_areas.shape[0]
        self.p_loadareas_residential = lv_load_areas["peak_load_residential"].sum()
        self.p_loadareas_cts = lv_load_areas["peak_load_cts"].sum()
        self.p_loadareas_industrial = lv_load_areas["peak_load_industrial"].sum()
        self.p_loadareas_total = lv_load_areas["peak_load"].sum()

        # Load Data
        egon_buildings = pd.DataFrame()
        for id_db, row in lv_load_areas.iterrows():
            egon_buildings = pd.concat(
                [egon_buildings, db_io.get_egon_buildings(orm, session, grid_id, row)]
            )

        self.n_loads_residential = egon_buildings.loc[
            egon_buildings["residential_capacity"] > 0.0, "residential_capacity"
        ].count()
        self.p_loads_residential = egon_buildings["residential_capacity"].sum()
        self.n_households = egon_buildings["number_households"].sum()

        self.n_loads_cts = egon_buildings.loc[
            egon_buildings["cts_capacity"] > 0.0, "cts_capacity"
        ].count()
        self.p_loads_cts = egon_buildings["cts_capacity"].sum()

        self.n_loads_industrial = egon_buildings.loc[
            egon_buildings["industrial_capacity"] > 0.0, "industrial_capacity"
        ].count()
        self.p_loads_industrial = egon_buildings["industrial_capacity"].sum()

        self.n_loads_total = (
            self.n_loads_cts + self.n_loads_residential + self.n_loads_industrial
        )
        self.p_loads_total = egon_buildings["capacity"].sum()

        # Generator Data
        res_generators = db_io.get_res_generators(orm, session, mv_grid_district)
        conv_generators = db_io.get_conv_generators(orm, session, mv_grid_district)

        _ = res_generators.loc[
            res_generators["generation_subtype"] == "pv_rooftop", "electrical_capacity"
        ]
        self.n_gens_pv_openspace = _.count()
        self.p_gens_pv_openspace = _.sum()

        _ = res_generators.loc[
            res_generators["generation_subtype"] == "open_space", "electrical_capacity"
        ]

        self.n_gens_pv_rooftop = _.count()
        self.p_gens_pv_rooftop = _.sum()

        _ = res_generators.loc[
            res_generators["generation_type"] == "wind", "electrical_capacity"
        ]
        self.n_gens_wind = _.count()
        self.p_gens_wind = _.sum()

        _ = res_generators.loc[
            res_generators["generation_type"] == "biomass", "electrical_capacity"
        ]
        self.n_gens_biomass = _.count()
        self.p_gens_biomass = _.sum()

        _ = res_generators.loc[
            res_generators["generation_type"] == "water", "electrical_capacity"
        ]
        self.n_gens_water = _.count()
        self.p_gens_water = _.sum()

        _ = res_generators["electrical_capacity"]
        self.n_gens_renewable_total = _.count()
        self.p_gens_renewable_total = _.sum()

        _ = conv_generators["electrical_capacity"]
        self.n_gens_conventional_total = _.count()
        self.p_gens_conventional_total = _.sum()

        self.n_gens_total = self.n_gens_renewable_total + self.n_gens_conventional_total
        self.p_gens_total = self.p_gens_renewable_total + self.p_gens_conventional_total

        for attribute in [
            "p_loadareas_cts",
            "p_loadareas_industrial",
            "p_loadareas_residential",
            "p_loadareas_total",
            "p_loads_residential",
            "p_loads_cts",
            "p_loads_industrial",
            "p_loads_total",
            "p_gens_pv_rooftop",
            "p_gens_pv_openspace",
            "p_gens_water",
            "p_gens_wind",
            "p_gens_biomass",
            "p_gens_renewable_total",
            "p_gens_conventional_total",
            "p_gens_total",
        ]:
            setattr(self, attribute, getattr(self, attribute) / 1e3)

    def update_from_edisgo_obj(self, grid_obj):
        logger.info("Update GridStats object from edisgo object.")

        self.source = "file"

        if isinstance(grid_obj, EDisGo):
            self.grid_type = "MV_Grid_District"
            buses_df = grid_obj.topology.buses_df
            lines_df = grid_obj.topology.lines_df
            transformers_df = pd.concat(
                [
                    grid_obj.topology.transformers_df,
                    grid_obj.topology.transformers_hvmv_df,
                ]
            )
            loads_df = grid_obj.topology.loads_df
            generators_df = grid_obj.topology.generators_df

            # MV Grid District Data
            self.geom_grid_district = grid_obj.topology.grid_district["geom"].wkt
            _ = grid_obj.topology.mv_grid.station
            self.geom_substation = Point(_["x"].values[0], _["y"].values[0]).wkt
            # LV Load Area Data
            self.population = grid_obj.topology.grid_district["population"]

            grid_obj.set_time_series_worst_case_analysis()
            self.integrity_check_messages = grid_obj.check_integrity()
        elif isinstance(grid_obj, Grid):
            self.v_nom = grid_obj.nominal_voltage
            buses_df = grid_obj.buses_df
            lines_df = grid_obj.lines_df
            transformers_df = grid_obj.transformers_df
            loads_df = grid_obj.loads_df
            generators_df = grid_obj.generators_df

            _ = grid_obj.get_feeder_stats()
            self.n_feeders = _.shape[0]
            self.l_feeders = _["length"].to_list()
            self.max_feeder_length = _["length"].max()

            if isinstance(grid_obj, MVGrid):
                self.grid_type = "MV_Grid"
                self.mv_grid_id = grid_obj.id
                self.suffix = f"MVGrid_{grid_obj.id}"
            elif isinstance(grid_obj, LVGrid):
                self.grid_type = "LV_Grid"
                self.lv_grid_id = grid_obj.id
                self.suffix = f"LVGrid_{grid_obj.id}"
            else:
                raise TypeError
        else:
            raise TypeError

        # Topology
        self.n_buses = buses_df.shape[0]
        self.n_lines = lines_df.shape[0]
        self.n_types_lines = lines_df.groupby(["type_info"]).size().to_dict()
        self.l_lines = lines_df["length"].sum()
        self.l_types_lines = (
            lines_df[["type_info", "length"]]
            .groupby(["type_info"])
            .sum()["length"]
            .to_dict()
        )

        self.n_transformers = transformers_df.shape[0]
        self.n_types_transformers = (
            transformers_df.groupby(["type_info"]).size().to_dict()
        )

        # Load Data
        self.n_loads_residential = loads_df[loads_df["sector"] == "residential"].shape[
            0
        ]
        self.p_loads_residential = loads_df.loc[
            loads_df["sector"] == "residential", "p_set"
        ].sum()

        self.n_loads_cts = loads_df[loads_df["sector"] == "cts"].shape[0]
        self.p_loads_cts = loads_df.loc[loads_df["sector"] == "cts", "p_set"].sum()

        self.n_loads_industrial = loads_df[loads_df["sector"] == "industrial"].shape[0]
        self.p_loads_industrial = loads_df.loc[
            loads_df["sector"] == "industrial", "p_set"
        ].sum()

        self.n_loads_total = (
            self.n_loads_cts + self.n_loads_residential + self.n_loads_industrial
        )
        self.p_loads_total = loads_df["p_set"].sum()

        # Generator Data
        _ = generators_df.loc[generators_df["subtype"] == "pv_rooftop", "p_nom"]
        self.n_gens_pv_openspace = _.count()
        self.p_gens_pv_openspace = _.sum()

        _ = generators_df.loc[generators_df["subtype"] == "open_space", "p_nom"]
        self.n_gens_pv_rooftop = _.count()
        self.p_gens_pv_rooftop = _.sum()

        _ = generators_df.loc[generators_df["type"] == "wind", "p_nom"]
        self.n_gens_wind = _.count()
        self.p_gens_wind = _.sum()

        _ = generators_df.loc[generators_df["type"] == "biomass", "p_nom"]
        self.n_gens_biomass = _.count()
        self.p_gens_biomass = _.sum()

        _ = generators_df.loc[generators_df["type"] == "water", "p_nom"]
        self.n_gens_water = _.count()
        self.p_gens_water = _.sum()

        _ = generators_df.loc[
            generators_df["type"].isin(["solar", "wind", "biomass", "water"]), "p_nom"
        ]
        self.n_gens_renewable_total = _.count()
        self.p_gens_renewable_total = _.sum()

        _ = generators_df.loc[generators_df["type"].isin(["conventional"]), "p_nom"]
        self.n_gens_conventional_total = _.count()
        self.p_gens_conventional_total = _.sum()

        self.n_gens_total = self.n_gens_renewable_total + self.n_gens_conventional_total
        self.p_gens_total = self.p_gens_renewable_total + self.p_gens_conventional_total

    def to_json(self, file_name):
        if self.suffix:
            file_name = f"{os.path.splitext(file_name)[0]}_{self.suffix}.json"
        logger.debug(f"Save stats to json file: '{file_name}'")
        pd.Series(self.__dict__).to_json(file_name, indent=4)

    def from_json(self, file_name):
        logger.debug(f"Read stats from json file: '{file_name}'")
        with open(file_name) as json_file:
            self.__dict__ = json.load(json_file)

    def save_stats(self, path):
        return


def load_grid_stats_to_df():
    return


def get_stats_from_edisgo_dir(dir_name=None, file_name=None):
    """
    Exfiltrate stats from a grid obj
    Integrity check
    Feeder length
    """
    logger.debug("Start getting stats from edisgo dir.")

    dir_name = os.path.expanduser(dir_name)
    edisgo_obj = import_edisgo_from_files(dir_name)

    grid_obj_list = list(edisgo_obj.topology.mv_grid.lv_grids)
    grid_obj_list = grid_obj_list + [edisgo_obj, edisgo_obj.topology.mv_grid]

    grid_obj_list[-2].topology.buses_df.iloc[0, 0] = 1
    grid_obj_list[-1].buses_df.iloc[0, 1] = 1

    for grid_obj in grid_obj_list:
        stats_obj = GridStats(grid_id=edisgo_obj.topology.id)
        stats_obj.update_from_edisgo_obj(grid_obj)
        if file_name:
            stats_obj.to_json(file_name=file_name)

    logger.debug("Finished getting stats from edisgo dir.")
    return stats_obj


def get_stats_from_database(orm, session, grid_id=None, file_name=None):
    logger.debug("Start getting stats from DB.")

    stats_obj = GridStats(grid_id=grid_id, grid_type="MV_Grid_District")
    stats_obj.update_from_db(orm, session, grid_id)
    if file_name:
        stats_obj.to_json(file_name=file_name)

    logger.debug("Finished getting stats from DB.")
    return stats_obj


def get_stats_from_file(file_name=None):
    logger.debug("Start getting stats from file.")

    stats_obj = GridStats()
    stats_obj.from_json(file_name=file_name)

    logger.debug("Finished getting stats from file.")
    return stats_obj


def stats_files_to_df(dir_name):
    index = []
    data = []
    for dirpath, dirnames, filenames in os.walk(dir_name):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if os.path.splitext(file_path)[-1] == ".json":
                index.append(os.path.splitext(filename)[0])
                data.append(pd.read_json(file_path, typ="series"))
    return pd.DataFrame(index=index, data=data)
