#!/usr/bin/env python

import atexit
import logging
import os
from contextlib import contextmanager

import yaml

import ding0
from ding0.tools import config as cfg_ding0
try:
    # only needed for the legacy data source on the OpenEnergy DataBase
    from egoio.tools import db
except ImportError:
    db = None
from sqlalchemy import URL, create_engine
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger()

package_path = ding0.__path__[0]
cfg_ding0.load_config('config_files.cfg')
cfg_ding0.load_config('config_db_tables.cfg')
cfg_ding0.load_config('config_db_credentials.cfg')

# Engines of egon-data configuration files, so that an SSH tunnel is opened
# only once per file
_egon_data_engines = {}


def get_database_type_from_config():
    input_data = cfg_ding0.get("input_data_source", "input_data")
    if input_data in ["model_draft", "versioned"]:
        database = "oedb"
    elif input_data == "local":
        database = "local"
    else:
        raise ValueError("Selected not implemented database.")
    return database


def get_egon_data_config_path():
    """
    Returns the path of the egon-data configuration file from the config.

    The path is set by `egon_data_config` in section `database_credentials`
    of `config_db_credentials.cfg`.

    Returns
    -------
    :obj:`str` or None
        Path, None if not set
    """
    if not cfg_ding0.cfg.has_option("database_credentials", "egon_data_config"):
        return None
    path = cfg_ding0.cfg.get("database_credentials", "egon_data_config").strip()
    return os.path.expanduser(path) if path else None


def egon_data_credentials(path):
    """
    Reads the connection parameters from an egon-data configuration file.

    The file is the YAML file egon-data writes (egon-data.configuration.yaml),
    as used by eDisGo. The optional section `ssh-tunnel` describes an SSH
    tunnel to the database server (`ssh-host`, `ssh-user`, `ssh-pkey`,
    `pgres-host`), optionally via a jump host (`ssh-jump-host`,
    `ssh-jump-user`, `ssh-jump-pkey`, which defaults to `ssh-pkey`).

    Parameters
    ----------
    path : :obj:`str`
        Path to the egon-data configuration file

    Returns
    -------
    :obj:`dict`
        Database parameters (keys `name`, `user`, `password`, `host`, `port`)
        and, if given, the SSH tunnel parameters in key `ssh` (keys `host`,
        `user`, `pkey`, `pgres_host`, `jump_host`, `jump_user`, `jump_pkey`)
    """
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        raise ValueError(f"egon-data configuration file {path} not found.")
    with open(path) as f:
        configuration = yaml.safe_load(f)

    egon_config = configuration["egon-data"]
    credentials = {
        "name": egon_config["--database-name"],
        "user": egon_config["--database-user"],
        "password": egon_config["--database-password"],
        "host": egon_config.get("--database-host", "localhost"),
        "port": int(egon_config["--database-port"]),
    }

    ssh_config = configuration.get("ssh-tunnel")
    if ssh_config:
        pkey = os.path.expanduser(ssh_config["ssh-pkey"])
        jump_pkey = os.path.expanduser(ssh_config.get("ssh-jump-pkey", pkey))
        for key_file in {pkey, jump_pkey}:
            if not os.path.isfile(key_file):
                raise ValueError(f"SSH key {key_file} not found.")
        credentials["ssh"] = {
            "host": ssh_config["ssh-host"],
            "user": ssh_config["ssh-user"],
            "pkey": pkey,
            "pgres_host": ssh_config.get("pgres-host", "localhost"),
            "jump_host": ssh_config.get("ssh-jump-host"),
            "jump_user": ssh_config.get("ssh-jump-user"),
            "jump_pkey": jump_pkey,
        }
    return credentials


def ssh_tunnel(ssh, port):
    """
    Opens an SSH tunnel to the database server, optionally via a jump host.

    The tunnels are closed when the Python process exits.

    Parameters
    ----------
    ssh : :obj:`dict`
        SSH tunnel parameters, see :func:`egon_data_credentials`
    port : :obj:`int`
        Port of the database on the server

    Returns
    -------
    :obj:`int`
        Local port of the tunnel
    """
    # imported here, as only needed for databases behind an SSH tunnel
    from sshtunnel import SSHTunnelForwarder

    ssh_address = (ssh["host"], 22)
    if ssh["jump_host"]:
        jump = SSHTunnelForwarder(
            (ssh["jump_host"], 22),
            ssh_username=ssh["jump_user"],
            ssh_pkey=ssh["jump_pkey"],
            remote_bind_address=ssh_address,
        )
        jump.start()
        atexit.register(jump.stop)
        ssh_address = ("127.0.0.1", jump.local_bind_port)

    server = SSHTunnelForwarder(
        ssh_address,
        ssh_username=ssh["user"],
        ssh_pkey=ssh["pkey"],
        remote_bind_address=(ssh["pgres_host"], port),
    )
    server.start()
    # registered after the jump tunnel, so it is stopped first
    atexit.register(server.stop)
    logger.info(f"SSH tunnel to the database on {ssh['host']} opened.")
    return server.local_bind_port


def get_egon_data_engine(path):
    """
    Engine for the egon-data database of an egon-data configuration file.

    Opens an SSH tunnel if the file has a section `ssh-tunnel`. Engine and
    tunnel are created once per file and reused.

    Parameters
    ----------
    path : :obj:`str`
        Path to the egon-data configuration file, see
        :func:`egon_data_credentials`

    Returns
    -------
    :sqlalchemy:`sqlalchemy.Engine<sqlalchemy.engine.Engine>`
        Database engine
    """
    path = os.path.realpath(os.path.expanduser(path))
    if path not in _egon_data_engines:
        credentials = egon_data_credentials(path)
        host, port = credentials["host"], credentials["port"]
        if "ssh" in credentials:
            host, port = "127.0.0.1", ssh_tunnel(credentials["ssh"], port)
        _egon_data_engines[path] = create_engine(
            URL.create(
                "postgresql+psycopg2",
                username=credentials["user"],
                password=credentials["password"],
                host=host,
                port=port,
                database=credentials["name"],
            ),
            echo=False,
        )
    return _egon_data_engines[path]


def get_engine(overwrite_database=None, egon_data_config=None):
    """
    Engine for local database.

    The local database is the egon-data database of an egon-data
    configuration file if `egon_data_config` or `egon_data_config` in
    `config_db_credentials.cfg` is set (see :func:`get_egon_data_engine`),
    otherwise the database in `config_db_credentials.cfg`.

    Parameters
    ----------
    overwrite_database : :obj:`str`, optional
        'local' or 'oedb', defaults to the database of
        `[input_data_source] input_data`
    egon_data_config : :obj:`str`, optional
        Path to an egon-data configuration file
    """
    if overwrite_database:
        database = overwrite_database
    else:
        database = get_database_type_from_config()
    if database == "oedb":
        if db is None:
            raise ImportError(
                "The OpenEnergy DataBase needs the package egoio, which is not "
                "installed (it is not compatible with SQLAlchemy 2)."
            )
        conn = db.connection(readonly=True)
    elif database == "local":
        egon_data_config = egon_data_config or get_egon_data_config_path()
        if egon_data_config:
            return get_egon_data_engine(egon_data_config)
        conn = create_engine(
            f"postgresql+psycopg2://{cfg_ding0.get('database_credentials', 'user')}:"
            f"{cfg_ding0.get('database_credentials', 'password')}@{cfg_ding0.get('database_credentials', 'host')}:"
            f"{int(cfg_ding0.get('database_credentials', 'port'))}/{cfg_ding0.get('database_credentials', 'name')}",
            echo=False,
        )
    return conn


@contextmanager
def session_scope(overwrite_database=None, engine=None, egon_data_config=None):
    """
    Provide a transactional scope around a series of operations.

    See :func:`get_engine` for the parameters.
    """
    if overwrite_database:
        database = overwrite_database
    else:
        database = get_database_type_from_config()

    if engine is None:
        Session = sessionmaker(bind=get_engine(
            overwrite_database=database, egon_data_config=egon_data_config))
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
