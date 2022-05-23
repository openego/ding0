#!/usr/bin/env python
# coding: utf-8


from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

import yaml
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


def paths(pid=None):
    """Obtain configuration file paths.

    If no `pid` is supplied, return the location of the standard
    configuration file. If `pid` is the string `"current"`, the
    path to the configuration file containing the configuration specific
    to the currently running process, i.e. the configuration obtained by
    overriding the values from the standard configuration file with the
    values explicitly supplied when the currently running process was
    invoked, is returned. If `pid` is the string `"*"` a list of all
    configuration belonging to currently running `egon-data` processes
    is returned. This can be used for error checking, because there
    should only ever be one such file.
    """
    pid = os.getpid() if pid == "current" else pid
    insert = f".pid-{pid}" if pid is not None else ""
    filename = f"egon-data{insert}.configuration.yaml"
    if pid == "*":
        return [p.absolute() for p in Path(".").glob(filename)]
    else:
        return [(Path(".") / filename).absolute()]


def config_settings() -> dict[str, dict[str, str]]:
    """Return a nested dictionary containing the configuration settings.

    It's a nested dictionary because the top level has command names as keys
    and dictionaries as values where the second level dictionary has command
    line switches applicable to the command as keys and the supplied values
    as values.

    So you would obtain the ``--database-name`` configuration setting used
    by the current invocation of of ``egon-data`` via

    .. code-block:: python

        settings()["egon-data"]["--database-name"]

    """
    files = paths(pid="*") + paths()
    if not files[0].exists():
        #         logger.warning(
        #             f"Configuration file:"
        #             f"\n\n{files[0]}\n\nnot found.\nUsing defaults."
        #         )
        return {
            "egon-data": {
                "--airflow-database-name": "airflow",
                "--airflow-port": 8080,
                "--compose-project-name": "egon-data",
                "--database-host": "127.0.0.1",
                "--database-name": "egon-data",
                "--database-password": "data",
                "--database-port": "59734",
                "--database-user": "egon",
                "--dataset-boundary": "Everything",
                "--docker-container-name": "egon-data-local-database-container",
                "--jobs": 1,
                "--random-seed": 42,
                "--processes-per-task": 1,
            }
        }
    with open(files[0]) as f:
        return yaml.safe_load(f)


def credentials():
    """Return local database connection parameters.

    Returns
    -------
    dict
        Complete DB connection information
    """
    translated = {
        "--database-name": "POSTGRES_DB",
        "--database-password": "POSTGRES_PASSWORD",
        "--database-host": "HOST",
        "--database-port": "PORT",
        "--database-user": "POSTGRES_USER",
    }
    configuration = config_settings()["egon-data"]
    update = {
        translated[flag]: configuration[flag]
        for flag in configuration
        if flag in translated
    }
    configuration.update(update)
    return configuration


def engine():
    """Engine for local database."""
    db_config = credentials()
    return create_engine(
        f"postgresql+psycopg2://{db_config['POSTGRES_USER']}:"
        f"{db_config['POSTGRES_PASSWORD']}@{db_config['HOST']}:"
        f"{db_config['PORT']}/{db_config['POSTGRES_DB']}",
        echo=False,
    )


@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations."""
    Session = sessionmaker(bind=engine())
    session = Session()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()
