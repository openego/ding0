"""This file is part of DINGO, the DIstribution Network GeneratOr.
DINGO is a tool to generate synthetic medium and low voltage power
distribution grids based on open data.

It is developed in the project open_eGo: https://openegoproject.wordpress.com

DING0 lives at github: https://github.com/openego/ding0/
The documentation is available on RTD: http://ding0.readthedocs.io"""

__copyright__ = "Reiner Lemoine Institut gGmbH"
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__url__ = "https://github.com/openego/ding0/blob/master/LICENSE"
__author__ = "nesnoj, gplssm, jh-RLI"

import numpy as np
import pandas as pd
from pathlib import Path
import json
import os

import re

from sqlalchemy import create_engine
from egoio.db_tables import model_draft as md
from egoio.tools.db import connection

from sqlalchemy import MetaData, ARRAY, BigInteger, Boolean, CheckConstraint, Column, Date, DateTime, Float, ForeignKey, ForeignKeyConstraint, Index, Integer, JSON, Numeric, SmallInteger, String, Table, Text, UniqueConstraint, text
from geoalchemy2.types import Geometry, Raster
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql.hstore import HSTORE
from sqlalchemy.dialects.postgresql.base import OID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import ARRAY, DOUBLE_PRECISION, INTEGER, NUMERIC, TEXT, BIGINT, TIMESTAMP, VARCHAR

con = connection()

Base = declarative_base()
metadata = Base.metadata

DING0_TABLES = {'versioning': 'ding0_versioning',
                'line': 'ding0_line',
                'lv_branchtee': 'ding0_lv_branchtee',
                'lv_generator': 'ding0_lv_generator',
                'lv_load': 'ding0_lv_load',
                'lv_grid': 'ding0_lv_grid',
                'lv_station': 'ding0_lv_station',
                'mvlv_transformer': 'ding0_mvlv_transformer',
                'mvlv_mapping': 'ding0_mvlv_mapping',
                'mv_branchtee': 'ding0_mv_branchtee',
                'mv_circuitbreaker': 'ding0_mv_circuitbreaker',
                'mv_generator': 'ding0_mv_generator',
                'mv_load': 'ding0_mv_load',
                'mv_grid': 'ding0_mv_grid',
                'mv_station': 'ding0_mv_station',
                'hvmv_transformer': 'ding0_hvmv_transformer'}


# metadatastring file folder. #ToDO: Test if Path works on other os (Tested on Windows7)
# Modify if folder name is different
FOLDER = Path('C:/ego_grid_ding0_metadatastrings')


def load_json_files():
    """
    Creats a list of all .json files in FOLDER

    Parameters
    ----------
    :return: dict: jsonmetadata
             contains all .json files from the folder
    """

    print(FOLDER)
    full_dir = os.walk(FOLDER.parent / FOLDER.name)
    jsonmetadata = []

    for jsonfiles in full_dir:
        for jsonfile in jsonfiles:
            jsonmetadata = jsonfile

    return jsonmetadata



def prepare_metadatastring_fordb(table):
    """
    Prepares the JSON String for the sql comment on table

    Required: The .json file names must contain the table name (for example from create_ding0_sql_tables())
    Instruction: Check the SQL "comment on table" for each table (f.e. use pgAdmin)

    Parameters
    ----------
    table:  str
            table name of the sqlAlchemy table

    return: mdsstring:str
            Contains the .json file as string
    """

    for file in load_json_files():
        JSONFILEPATH = FOLDER / file
        with open(JSONFILEPATH, encoding='UTF-8') as f:
            if table in file:
                # included for testing / or logging
                # print("Comment on table: " + table + "\nusing this metadata file: " + file + "\n")
                mds = json.load(f)
                mdsstring = json.dumps(mds, indent=4, ensure_ascii=False)
                return mdsstring


def create_ding0_sql_tables(engine, ding0_schema=None):
    """
    Create the ding0 tables

    Parameters
    ----------
    engine: :py:mod:`sqlalchemy.engine.base.Engine`
        Sqlalchemy database engine

    ding0_schema: :obj:`str`
        The schema in which the tables are to be created
        Default: None
    """

    # versioning table
    versioning = Table(DING0_TABLES['versioning'], metadata,
                       Column('run_id', BigInteger, primary_key=True, autoincrement=False, nullable=False),
                       Column('description', String(3000)),
                       schema=ding0_schema,
                       comment=prepare_metadatastring_fordb("versioning")
                       )

    # ding0 lines table
    ding0_line = Table(DING0_TABLES['line'], metadata,
                    Column('id', Integer, primary_key=True),
                    Column('run_id', BigInteger, ForeignKey(versioning.columns.run_id), nullable=False),
                    Column('id_db', BigInteger),
                    Column('edge_name', String(100)),
                    Column('grid_name', String(100)),
                    Column('node1', String(100)),
                    Column('node2', String(100)),
                    Column('type_kind', String(100)),
                    Column('type_name', String(100)),
                    Column('length', Float(10)),
                    Column('u_n', Float(10)),
                    Column('c', Float(10)),
                    Column('l', Float(10)),
                    Column('r', Float(10)),
                    Column('i_max_th', Float(10)),
                    Column('geom', Geometry('LINESTRING', 4326)),
                    schema=ding0_schema,
                    comment=prepare_metadatastring_fordb("ding0_line")
                    )

    # ding0 lv_branchtee table
    ding0_lv_branchtee = Table(DING0_TABLES['lv_branchtee'], metadata,
                    Column('id', Integer, primary_key=True),
                    Column('run_id', BigInteger, ForeignKey(versioning.columns.run_id), nullable=False),
                    Column('id_db', BigInteger),
                    Column('geom', Geometry('POINT', 4326)),
                    Column('name', String(100)),
                    schema=ding0_schema,
                    comment=prepare_metadatastring_fordb("ding0_lv_branchtee")
                    )

    # ding0 lv_generator table
    ding0_lv_generator = Table(DING0_TABLES['lv_generator'], metadata,
                    Column('id', Integer, primary_key=True),
                    Column('run_id', BigInteger, ForeignKey(versioning.columns.run_id), nullable=False),
                    Column('id_db', BigInteger),
                    Column('la_id', BigInteger),
                    Column('name', String(100)),
                    Column('lv_grid_id', BigInteger),
                    Column('geom', Geometry('POINT', 4326)),
                    Column('type', String(22)),
                    Column('subtype', String(22)),
                    Column('v_level', Integer),
                    Column('nominal_capacity', Float(10)),
                    Column('weather_cell_id', BigInteger),
                    Column('is_aggregated', Boolean),
                    schema=ding0_schema,
                    comment=prepare_metadatastring_fordb("ding0_lv_generator")
                    )

    # ding0 lv_load table
    ding0_lv_load = Table(DING0_TABLES['lv_load'], metadata,
                    Column('id', Integer, primary_key=True),
                    Column('run_id', BigInteger, ForeignKey(versioning.columns.run_id), nullable=False),
                    Column('id_db', BigInteger),
                    Column('name', String(100)),
                    Column('lv_grid_id', BigInteger),
                    Column('geom', Geometry('POINT', 4326)),
                    Column('consumption', String(100)),
                    schema=ding0_schema,
                    comment=prepare_metadatastring_fordb("ding0_lv_load")
                    )

    # ding0 lv_station table
    ding0_lv_station = Table(DING0_TABLES['lv_station'], metadata,
                    Column('id', Integer, primary_key=True),
                    Column('run_id', BigInteger, ForeignKey(versioning.columns.run_id), nullable=False),
                    Column('id_db', BigInteger),
                    Column('geom', Geometry('POINT', 4326)),
                    Column('name', String(100)),
                    schema=ding0_schema,
                    comment=prepare_metadatastring_fordb("ding0_lv_station")
                    )

    # ding0 mvlv_transformer table
    ding0_mvlv_transformer = Table(DING0_TABLES['mvlv_transformer'], metadata,
                    Column('id', Integer, primary_key=True),
                    Column('run_id', BigInteger, ForeignKey(versioning.columns.run_id), nullable=False),
                    Column('id_db', BigInteger),
                    Column('geom', Geometry('POINT', 4326)),
                    Column('name', String(100)),
                    Column('voltage_op', Float(10)),
                    Column('s_nom', Float(10)),
                    Column('x', Float(10)),
                    Column('r', Float(10)),
                    schema=ding0_schema,
                    comment=prepare_metadatastring_fordb("ding0_mvlv_transformer")
                    )

    # ding0 mvlv_mapping table
    ding0_mvlv_mapping = Table(DING0_TABLES['mvlv_mapping'], metadata,
                    Column('id', Integer, primary_key=True),
                    Column('run_id', BigInteger, ForeignKey(versioning.columns.run_id), nullable=False),
                    Column('lv_grid_id', BigInteger),
                    Column('lv_grid_name', String(100)),
                    Column('mv_grid_id', BigInteger),
                    Column('mv_grid_name', String(100)),
                    schema=ding0_schema,
                    comment=prepare_metadatastring_fordb("ding0_mvlv_mapping")
                    )

    # ding0 mv_branchtee table
    ding0_mv_branchtee = Table(DING0_TABLES['mv_branchtee'], metadata,
                    Column('id', Integer, primary_key=True),
                    Column('run_id', BigInteger, ForeignKey(versioning.columns.run_id), nullable=False),
                    Column('id_db', BigInteger),
                    Column('geom', Geometry('POINT', 4326)),
                    Column('name', String(100)),
                    schema=ding0_schema,
                    comment=prepare_metadatastring_fordb("ding0_mv_branchtee")
                    )

    # ding0 mv_circuitbreaker table
    ding0_mv_circuitbreaker = Table(DING0_TABLES['mv_circuitbreaker'], metadata,
                    Column('id', Integer, primary_key=True),
                    Column('run_id', BigInteger, ForeignKey(versioning.columns.run_id), nullable=False),
                    Column('id_db', BigInteger),
                    Column('geom', Geometry('POINT', 4326)),
                    Column('name', String(100)),
                    Column('status', String(10)),
                    schema=ding0_schema,
                    comment=prepare_metadatastring_fordb("ding0_mv_circuitbreaker")
                    )

    # ding0 mv_generator table
    ding0_mv_generator = Table(DING0_TABLES['mv_generator'], metadata,
                    Column('id', Integer, primary_key=True),
                    Column('run_id', BigInteger, ForeignKey(versioning.columns.run_id), nullable=False),
                    Column('id_db', BigInteger),
                    Column('name', String(100)),
                    Column('geom', Geometry('POINT', 4326)),
                    Column('type', String(22)),
                    Column('subtype', String(22)),
                    Column('v_level', Integer),
                    Column('nominal_capacity', Float(10)),
                    Column('weather_cell_id', BigInteger),
                    Column('is_aggregated', Boolean),
                    schema=ding0_schema,
                    comment=prepare_metadatastring_fordb("ding0_mv_generator")
                    )

    # ding0 mv_load table
    ding0_mv_load = Table(DING0_TABLES['mv_load'], metadata,
                    Column('id', Integer, primary_key=True),
                    Column('run_id', BigInteger, ForeignKey(versioning.columns.run_id), nullable=False),
                    Column('id_db', BigInteger),
                    Column('name', String(100)),
                    Column('geom', Geometry('LINESTRING', 4326)),
                    Column('is_aggregated', Boolean),
                    Column('consumption', String(100)),
                    schema=ding0_schema,
                    comment=prepare_metadatastring_fordb("ding0_mv_load")
                    )

    # ding0 mv_grid table
    ding0_mv_grid = Table(DING0_TABLES['mv_grid'], metadata,
                    Column('id', Integer, primary_key=True),
                    Column('run_id', BigInteger, ForeignKey(versioning.columns.run_id), nullable=False),
                    Column('id_db', BigInteger),
                    Column('geom', Geometry('MULTIPOLYGON', 4326)),
                    Column('name', String(100)),
                    Column('population', BigInteger),
                    Column('voltage_nom', Float(10)),
                    schema=ding0_schema,
                    comment=prepare_metadatastring_fordb("ding0_mv_grid")
                    )



    # ding0 mv_station table
    ding0_mv_station = Table(DING0_TABLES['mv_station'], metadata,
                    Column('id', Integer, primary_key=True),
                    Column('run_id', BigInteger, ForeignKey(versioning.columns.run_id), nullable=False),
                    Column('id_db', BigInteger),
                    Column('geom', Geometry('POINT', 4326)),
                    Column('name', String(100)),
                    schema=ding0_schema,
                    comment=prepare_metadatastring_fordb("ding0_mv_station")
                    )

    # ding0 hvmv_transformer table
    ding0_hvmv_transformer = Table(DING0_TABLES['hvmv_transformer'], metadata,
                    Column('id', Integer, primary_key=True),
                    Column('run_id', BigInteger, ForeignKey(versioning.columns.run_id), nullable=False),
                    Column('id_db', BigInteger),
                    Column('geom', Geometry('POINT', 4326)),
                    Column('name', String(100)),
                    Column('voltage_op', Float(10)),
                    Column('s_nom', Float(10)),
                    Column('x', Float(10)),
                    Column('r', Float(10)),
                    schema=ding0_schema,
                    comment=prepare_metadatastring_fordb("ding0_hvmv_transformer")
                    )

    # create all the tables
    metadata.create_all(engine, checkfirst=True)


def df_sql_write(dataframe, db_table, engine):
    """
    Convert dataframes such that their column names
    are made small and the index is renamed 'id' so as to
    correctly load its data to its appropriate sql table.

    .. ToDo:  need to check for id_db instead of only 'id' in index label names

    NOTE: This function does not check if the dataframe columns
    matches the db_table fields, if they do not then no warning
    is given.

    Parameters
    ----------
    dataframe: :pandas:`DataFrame<dataframe>`
        The pandas dataframe to be transferred to its
        apprpritate db_table

    db_table: :py:mod:`sqlalchemy.sql.schema.Table`
        A table instance definition from sqlalchemy.
        NOTE: This isn't an orm definition

    engine: :py:mod:`sqlalchemy.engine.base.Engine`
        Sqlalchemy database engine
    """
    sql_write_df = dataframe.copy()
    sql_write_df.columns = sql_write_df.columns.map(str.lower)
    # sql_write_df = sql_write_df.set_index('id')
    sql_write_df.to_sql(db_table.name, con=engine, if_exists='append', index=None)

    # create a dummy dataframe with lines
    line1 = pd.DataFrame({'run_id': [1, 1],
                          'id': [1, 2],
                          'edge_name': ['line1', 'line2'],
                          'grid_name': ['mv_grid5', 'mvgrid5'],
                          'node1': [1, 2],
                          'node2': [2, 3],
                          'type_kind': ['line', 'line'],
                          'type_name': ['NASX2Y', 'NA2SXX2Y'],
                          'length': [1.3, 2.3],
                          'U_n': [10, 10],
                          'C': [0.002, 0.001],
                          'L': [0.01, 0.02],
                          'R': [0.0001, 0.00005],
                          'I_max_th': [5, 6]})


def export_network_to_db(session, schema, table, tabletype, srid):
    print("Exporting table type : {}".format(tabletype))
    if tabletype == 'line':
        pass

    elif tabletype == 'lv_cd':
        pass

    elif tabletype == 'lv_gen':
        pass

    elif tabletype == 'lv_load':
        pass

    elif tabletype == 'lv_grid':
        pass

    elif tabletype == 'lv_station':
        pass

    elif tabletype == 'mvlv_trafo':
        pass

    elif tabletype == 'mvlv_mapping':
        pass

    elif tabletype == 'mv_cd':
        pass

    elif tabletype == 'mv_cb':
        pass

    elif tabletype == 'mv_gen':
        pass

    elif tabletype == 'mv_load':
        pass

    elif tabletype == 'mv_grid':
        pass

    elif tabletype == 'mv_station':
        pass

    elif tabletype == 'hvmv_trafo':
        pass
        # if not engine.dialect.has_table(engine, 'ego_grid_mv_transformer'):
        #     print('helloworld')

    #session.commit()


def export_data_to_db(session, schema, run_id, metadata_json, srid,
                        lv_grid, lv_gen, lv_cd, lv_stations, mvlv_trafos,
                        lv_loads,
                        mv_grid, mv_gen, mv_cb, mv_cd, mv_stations, hvmv_trafos,
                        mv_loads, lines, mvlv_mapping):
    # only for testing
    # engine = create_engine('sqlite:///:memory:')

    # get the run_id from model_draft.ego_grid_ding0_versioning
    # compare the run_id from table to the current run_id

    # oedb_versioning_query = session.query(
    #     schema.EgoGridDing0Versioning.run_id,
    #     schema.EgoGridDing0Versioning.description
    # ).filter(schema.EgoGridDing0Versioning.run_id == run_id)
    #
    # oedb_versioning = pd.read_sql_query(oedb_versioning_query.statement,
    #                                     session.bind)
    db_versioning = pd.DataFrame()

    if db_versioning.empty:
        # if the run_id doesn't exist then
        # create entry into ego_grid_ding0_versioning:
        metadata_df = pd.DataFrame({'run_id': run_id,
                                    'description': metadata_json},
                                   index=[0])
        metadata_df.apply(lambda row:
                          session.add(schema.EgoGridDing0Versioning(
                              run_id=row['run_id'],
                              description=row['description'],
                          ))
                          , axis=1)
        session.commit()

        export_network_to_db(session, lv_grid, 'lv_grid', srid)
        export_network_to_db(session, lv_gen, 'lv_gen', srid)
        export_network_to_db(session, lv_cd, 'lv_cd', srid)
        export_network_to_db(session, lv_stations, 'lv_station', srid)
        export_network_to_db(session, mvlv_trafos, 'mvlv_trafo', srid)
        export_network_to_db(session, lv_loads, 'lv_load', srid)
        export_network_to_db(session, mv_grid, 'mv_grid', srid)
        export_network_to_db(session, mv_gen, 'mv_gen', srid)
        export_network_to_db(session, mv_cb, 'mv_cb', srid)
        export_network_to_db(session, mv_cd, 'mv_cd', srid)
        export_network_to_db(session, mv_stations, 'mv_station', srid)
        export_network_to_db(session, hvmv_trafos, 'hvmv_trafo', srid)
        export_network_to_db(session, mv_loads, 'mv_load', srid)
        export_network_to_db(session, lines, 'line', srid)
        export_network_to_db(session, mvlv_mapping, 'mvlv_mapping', srid)
    else:
        raise KeyError("run_id already present! No tables are input!")


def create_ding0_db_tables(engine, schema,):
    tables = [schema.EgoGridDing0Versioning,
              schema.EgoGridDing0Line,
              schema.EgoGridDing0LvBranchtee,
              schema.EgoGridDing0LvGenerator,
              schema.EgoGridDing0LvLoad,
              schema.EgoGridDing0LvGrid,
              schema.EgoGridDing0LvStation,
              schema.EgoGridDing0MvlvTransformer,
              schema.EgoGridDing0MvlvMapping,
              schema.EgoGridDing0MvBranchtee,
              schema.EgoGridDing0MvCircuitbreaker,
              schema.EgoGridDing0MvGenerator,
              schema.EgoGridDing0MvLoad,
              schema.EgoGridDing0MvGrid,
              schema.EgoGridDing0MvStation,
              schema.EgoGridDing0HvmvTransformer]

    for tab in tables:
        tab().__table__.create(bind=engine, checkfirst=True)


def drop_ding0_db_tables(engine, schema):
    tables = [schema.EgoGridDing0Line,
              schema.EgoGridDing0LvBranchtee,
              schema.EgoGridDing0LvGenerator,
              schema.EgoGridDing0LvLoad,
              schema.EgoGridDing0LvGrid,
              schema.EgoGridDing0LvStation,
              schema.EgoGridDing0MvlvTransformer,
              schema.EgoGridDing0MvlvMapping,
              schema.EgoGridDing0MvBranchtee,
              schema.EgoGridDing0MvCircuitbreaker,
              schema.EgoGridDing0MvGenerator,
              schema.EgoGridDing0MvLoad,
              schema.EgoGridDing0MvGrid,
              schema.EgoGridDing0MvStation,
              schema.EgoGridDing0HvmvTransformer,
              schema.EgoGridDing0Versioning]

    print("Please confirm that you would like to drop the following tables:")
    for n, tab in enumerate(tables):
        print("{: 3d}. {}".format(n, tab))

    print("Please confirm with either of the choices below:\n" +
          "- yes\n" +
          "- no\n" +
          "- the indexes to drop in the format 0, 2, 3, 5")
    confirmation = input(
        "Please type the choice completely as there is no default choice.")
    if re.fullmatch('[Yy]es', confirmation):
        for tab in tables:
            tab().__table__.drop(bind=engine, checkfirst=True)
    elif re.fullmatch('[Nn]o', confirmation):
        print("Cancelled dropping of tables")
    else:
        try:
            indlist = confirmation.split(',')
            indlist = list(map(int, indlist))
            print("Please confirm deletion of the following tables:")
            tablist = np.array(tables)[indlist].tolist()
            for n, tab in enumerate(tablist):
                print("{: 3d}. {}".format(n, tab))
            con2 = input("Please confirm with either of the choices below:\n" +
                         "- yes\n" +
                         "- no")
            if re.fullmatch('[Yy]es', con2):
                for tab in tablist:
                    tab().__table__.drop(bind=engine, checkfirst=True)
            elif re.fullmatch('[Nn]o', con2):
                print("Cancelled dropping of tables")
            else:
                print("The input is unclear, no action taken")
        except ValueError:
            print("Confirmation unclear, no action taken")


def db_tables_change_owner(engine, schema):
    tables = [schema.EgoGridDing0Line,
              schema.EgoGridDing0LvBranchtee,
              schema.EgoGridDing0LvGenerator,
              schema.EgoGridDing0LvLoad,
              schema.EgoGridDing0LvGrid,
              schema.EgoGridDing0LvStation,
              schema.EgoGridDing0MvlvTransformer,
              schema.EgoGridDing0MvlvMapping,
              schema.EgoGridDing0MvBranchtee,
              schema.EgoGridDing0MvCircuitbreaker,
              schema.EgoGridDing0MvGenerator,
              schema.EgoGridDing0MvLoad,
              schema.EgoGridDing0MvGrid,
              schema.EgoGridDing0MvStation,
              schema.EgoGridDing0HvmvTransformer,
              schema.EgoGridDing0Versioning]


    def change_owner(engine, table, role):
        r"""Gives access to database users/ groups
        Parameters
        ----------
        session : sqlalchemy session object
            A valid connection to a database
        table : sqlalchmy Table class definition
            The database table
        role : str
            database role that access is granted to
        """
        tablename = table.__table__.name
        schema = table.__table__.schema

        grant_str = """ALTER TABLE {schema}.{table}
        OWNER TO {role};""".format(schema=schema, table=tablename,
                          role=role)

        # engine.execute(grant_str)
        engine.execution_options(autocommit=True).execute(grant_str)

    # engine.echo=True

    for tab in tables:
        change_owner(engine, tab, 'oeuser')

    engine.close()
