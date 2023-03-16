import logging

from shapely import wkt

logger = logging.getLogger(__name__)


def get_keys_where_value_not_none(dict_x):
    keys_dict_x = set()
    for key, value in dict_x.items():
        if value is not None:
            keys_dict_x.add(key)
    return keys_dict_x


def compare_to_stats_obj(stats_db_obj=None, stats_edisgo_obj=None):
    logger.debug("Start comparing stats obj.")

    stats_edisgo_keys = get_keys_where_value_not_none(stats_edisgo_obj.__dict__)
    stats_db_keys = get_keys_where_value_not_none(stats_db_obj.__dict__)

    common_keys = stats_db_keys.intersection(stats_edisgo_keys)

    data_matches = True
    for key in common_keys:
        db_value = getattr(stats_db_obj, key)
        edisgo_value = getattr(stats_edisgo_obj, key)
        if key in ["geom_grid_district", "geom_substation"]:
            geom_db = wkt.loads(db_value)
            geom_edisgo_obj = wkt.loads(edisgo_value)
            if not geom_db.equals_exact(geom_edisgo_obj, 1e-3):
                logger.info(f"{key=}: {db_value=} != {edisgo_value=}")
                data_matches = False
        elif db_value == edisgo_value:
            continue
        else:
            logger.error(f"{key=}: {db_value=} != {edisgo_value=}")
            data_matches = False

    logger.debug("Finished comparing stats obj.")
    return data_matches
