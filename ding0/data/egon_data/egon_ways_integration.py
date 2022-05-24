import pandas as pd
import saio
from config.config_lv_grids_osm import get_config_osm
from geoalchemy2.shape import to_shape
from sqlalchemy import func, Integer, cast
from ding0.data.egon_data import egon_db


def get_egon_ways(geo_area):
    egon_db.credentials()
    engine = egon_db.engine()

    # get schema metadata from db
    saio.register_schema("openstreetmap", engine)

    # get table metadata from db
    from saio.openstreetmap import osm_ways_with_segments

    # retrieve residential buildings
    with egon_db.session_scope() as session:
        cells_query = session.query(
            osm_ways_with_segments.osm_id,
            osm_ways_with_segments.nodes,
            osm_ways_with_segments.geom.label('geometry'),
            osm_ways_with_segments.highway,
            osm_ways_with_segments.length_segments,
        ).filter(
            func.st_intersects(
                func.ST_GeomFromText(geo_area, get_config_osm("srid")),
                osm_ways_with_segments.geom,
            )
        )

    return cells_query