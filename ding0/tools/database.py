#!/usr/bin/env python

import logging
from contextlib import contextmanager

from egoio.tools import db
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import ding0
from ding0.tools import config as cfg_ding0

logger = logging.getLogger()

package_path = ding0.__path__[0]
cfg_ding0.load_config("config_db_tables.cfg")


def get_database_type_from_config():
    input_data = cfg_ding0.get("input_data_source", "input_data")
    if input_data in ["model_draft", "versioned"]:
        database = "oedb"
    elif input_data == "local":
        database = "local"
    else:
        raise ValueError("Selected not implemented database.")
    return database


def engine(overwrite_database=None):
    """
    Engine for local database.
    """
    if overwrite_database:
        database = overwrite_database
    else:
        database = get_database_type_from_config()
    if database == "oedb":
        conn = db.connection(readonly=True)
    elif database == "local":
        conn = create_engine(
            f"postgresql+psycopg2://{cfg_ding0.get('local', 'database_user')}:"
            f"{cfg_ding0.get('local', 'database_password')}@{cfg_ding0.get('local', 'database_host')}:"
            f"{int(cfg_ding0.get('local', 'database_port'))}/{cfg_ding0.get('local', 'database_name')}",
            echo=False,
        )
    return conn


@contextmanager
def session_scope(overwrite_database=None):
    """
    Provide a transactional scope around a series of operations.
    """
    if overwrite_database:
        database = overwrite_database
    else:
        database = get_database_type_from_config()
    Session = sessionmaker(bind=engine(overwrite_database=database))
    session = Session()
    logger.info(f"Start session with database: {database}")
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()
