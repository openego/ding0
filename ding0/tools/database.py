#!/usr/bin/env python

import logging
from contextlib import contextmanager

import ding0
from ding0.tools import config as cfg_ding0
from egoio.tools import db
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger()

package_path = ding0.__path__[0]
cfg_ding0.load_config('config_files.cfg')
cfg_ding0.load_config('config_db_tables.cfg')
cfg_ding0.load_config('config_db_credentials.cfg')


def get_database_type_from_config():
    input_data = cfg_ding0.get("input_data_source", "input_data")
    if input_data in ["model_draft", "versioned"]:
        database = "oedb"
    elif input_data == "local":
        database = "local"
    else:
        raise ValueError("Selected not implemented database.")
    return database


def get_engine(overwrite_database=None):
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
            f"postgresql+psycopg2://{cfg_ding0.get('database_credentials', 'user')}:"
            f"{cfg_ding0.get('database_credentials', 'password')}@{cfg_ding0.get('database_credentials', 'host')}:"
            f"{int(cfg_ding0.get('database_credentials', 'port'))}/{cfg_ding0.get('database_credentials', 'name')}",
            echo=False,
        )
    return conn


@contextmanager
def session_scope(overwrite_database=None, engine=None):
    """
    Provide a transactional scope around a series of operations.
    """
    if overwrite_database:
        database = overwrite_database
    else:
        database = get_database_type_from_config()

    if engine is None:
        Session = sessionmaker(bind=get_engine(overwrite_database=database))
    else:
        Session = sessionmaker(bind=engine)

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
