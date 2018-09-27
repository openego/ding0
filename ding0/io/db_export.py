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
import json
import os
from pathlib import Path

import re

from egoio.tools.db import connection

import ding0
from ding0.io.export import export_network
from ding0.core import NetworkDing0

from sqlalchemy import MetaData, ARRAY, BigInteger, Boolean, CheckConstraint, Column, Date, DateTime, Float, ForeignKey, ForeignKeyConstraint, Index, Integer, JSON, Numeric, SmallInteger, String, Table, Text, UniqueConstraint, text
from geoalchemy2.types import Geometry, Raster
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


DECLARATIVE_BASE = declarative_base()
METADATA = DECLARATIVE_BASE.metadata

# Set the Database schema which you want to add the tables to
SCHEMA = "topology"

METADATA_STRING_FOLDER = os.path.join(ding0.__path__[0], 'io', 'metadatastrings')
CLEANED_METADATA_STRING_FOLDER_PATH = Path(METADATA_STRING_FOLDER)

DING0_TABLES = {'versioning': 'ego_ding0_versioning',
                'line': 'ego_ding0_line',
                'lv_branchtee': 'ego_ding0_lv_branchtee',
                'lv_generator': 'ego_ding0_lv_generator',
                'lv_load': 'ego_ding0_lv_load',
                'lv_grid': 'ego_ding0_lv_grid',
                'lv_station': 'ego_ding0_lv_station',
                'mvlv_transformer': 'ego_ding0_mvlv_transformer',
                'mvlv_mapping': 'ego_ding0_mvlv_mapping',
                'mv_branchtee': 'ego_ding0_mv_branchtee',
                'mv_circuitbreaker': 'ego_ding0_mv_circuitbreaker',
                'mv_generator': 'ego_ding0_mv_generator',
                'mv_load': 'ego_ding0_mv_load',
                'mv_grid': 'ego_ding0_mv_grid',
                'mv_station': 'ego_ding0_mv_station',
                'hvmv_transformer': 'ego_ding0_hvmv_transformer'}


def load_json_files():
    """
    Creats a list of all .json files in METADATA_STRING_FOLDER

    Parameters
    ----------
    :return: dict: jsonmetadata
             contains all .json file names from the folder
    """

    full_dir = os.walk(str(CLEANED_METADATA_STRING_FOLDER_PATH))
    jsonmetadata = []

    for jsonfiles in full_dir:
        for jsonfile in jsonfiles:
            jsonmetadata = jsonfile

    return jsonmetadata


def prepare_metadatastring_fordb(table):
    """
    Prepares the JSON String for the sql comment on table

    Required: The .json file names must contain the table name (for example from create_ding0_sql_tables())
    Instruction: Check the SQL "comment on table" for each table (e.g. use pgAdmin)

    Parameters
    ----------
    table:  str
            table name of the sqlAlchemy table

    return: mdsstring:str
            Contains the .json file as string
    """

    for json_file in load_json_files():
        json_file_path = os.path.join(CLEANED_METADATA_STRING_FOLDER_PATH, json_file)
        with open(json_file_path, encoding='UTF-8') as jf:
            if table in json_file:
                # included for testing / or logging
                # print("Comment on table: " + table + "\nusing this metadata string file: " + file + "\n")
                mds = json.load(jf)
                mdsstring = json.dumps(mds, indent=4, ensure_ascii=False)
                return mdsstring


def create_ding0_sql_tables(engine, ding0_schema=SCHEMA):
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

    # 1 versioning table
    versioning = Table(DING0_TABLES['versioning'], METADATA,
                       Column('run_id', BigInteger, primary_key=True, autoincrement=False, nullable=False),
                       Column('description', String(6000)),
                       schema=ding0_schema,
                       comment=prepare_metadatastring_fordb('versioning')
                       )

    # 2 ding0 lines table
    ding0_line = Table(DING0_TABLES['line'], METADATA,
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
                       comment=prepare_metadatastring_fordb('line')
                       )


    # 3 ding0 lv_branchtee table
    ding0_lv_branchtee = Table(DING0_TABLES['lv_branchtee'], METADATA,
                               Column('id', Integer, primary_key=True),
                               Column('run_id', BigInteger, ForeignKey(versioning.columns.run_id), nullable=False),
                               Column('id_db', BigInteger),
                               Column('geom', Geometry('POINT', 4326)),
                               Column('name', String(100)),
                               schema=ding0_schema,
                               comment=prepare_metadatastring_fordb('lv_branchtee')
                               )

    # 4 ding0 lv_generator table
    ding0_lv_generator = Table(DING0_TABLES['lv_generator'], METADATA,
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
                               comment=prepare_metadatastring_fordb('lv_generator')
                               )

    # 5 ding0 lv_load table
    ding0_lv_load = Table(DING0_TABLES['lv_load'], METADATA,
                          Column('id', Integer, primary_key=True),
                          Column('run_id', BigInteger, ForeignKey(versioning.columns.run_id), nullable=False),
                          Column('id_db', BigInteger),
                          Column('name', String(100)),
                          Column('lv_grid_id', BigInteger),
                          Column('geom', Geometry('POINT', 4326)),
                          Column('consumption', String(100)),
                          schema=ding0_schema,
                          comment=prepare_metadatastring_fordb('lv_load')
                          )

    # 6
    ding0_lv_grid = Table(DING0_TABLES['lv_grid'], METADATA,
                          Column('id', Integer, primary_key=True),
                          Column('run_id', BigInteger, ForeignKey(versioning.columns.run_id), nullable=False),
                          Column('id_db', BigInteger),
                          Column('name', String(100)),
                          Column('geom', Geometry('MULTIPOLYGON', 4326)),
                          Column('population', BigInteger),
                          Column('voltage_nom', Float(10)),
                          schema=ding0_schema,
                          comment=prepare_metadatastring_fordb('lv_grid')
                          )

    # 7 ding0 lv_station table
    ding0_lv_station = Table(DING0_TABLES['lv_station'], METADATA,
                             Column('id', Integer, primary_key=True),
                             Column('run_id', BigInteger, ForeignKey(versioning.columns.run_id), nullable=False),
                             Column('id_db', BigInteger),
                             Column('geom', Geometry('POINT', 4326)),
                             Column('name', String(100)),
                             schema=ding0_schema,
                             comment=prepare_metadatastring_fordb('lv_station')
                             )

    # 8 ding0 mvlv_transformer table
    ding0_mvlv_transformer = Table(DING0_TABLES['mvlv_transformer'], METADATA,
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
                                   comment=prepare_metadatastring_fordb("mvlv_transformer")
                                   )

    # 9 ding0 mvlv_mapping table
    ding0_mvlv_mapping = Table(DING0_TABLES['mvlv_mapping'], METADATA,
                               Column('id', Integer, primary_key=True),
                               Column('run_id', BigInteger, ForeignKey(versioning.columns.run_id), nullable=False),
                               Column('lv_grid_id', BigInteger),
                               Column('lv_grid_name', String(100)),
                               Column('mv_grid_id', BigInteger),
                               Column('mv_grid_name', String(100)),
                               schema=ding0_schema,
                               comment=prepare_metadatastring_fordb("mvlv_mapping")
                               )

    # 10 ding0 mv_branchtee table
    ding0_mv_branchtee = Table(DING0_TABLES['mv_branchtee'], METADATA,
                               Column('id', Integer, primary_key=True),
                               Column('run_id', BigInteger, ForeignKey(versioning.columns.run_id), nullable=False),
                               Column('id_db', BigInteger),
                               Column('geom', Geometry('POINT', 4326)),
                               Column('name', String(100)),
                               schema=ding0_schema,
                               comment=prepare_metadatastring_fordb("mv_branchtee")
                               )

    # 11 ding0 mv_circuitbreaker table
    ding0_mv_circuitbreaker = Table(DING0_TABLES['mv_circuitbreaker'], METADATA,
                                    Column('id', Integer, primary_key=True),
                                    Column('run_id', BigInteger, ForeignKey(versioning.columns.run_id), nullable=False),
                                    Column('id_db', BigInteger),
                                    Column('geom', Geometry('POINT', 4326)),
                                    Column('name', String(100)),
                                    Column('status', String(10)),
                                    schema=ding0_schema,
                                    comment=prepare_metadatastring_fordb("mv_circuitbreaker")
                                    )

    # 12 ding0 mv_generator table
    ding0_mv_generator = Table(DING0_TABLES['mv_generator'], METADATA,
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
                               comment=prepare_metadatastring_fordb("mv_generator")
                               )

    # 13 ding0 mv_load table
    ding0_mv_load = Table(DING0_TABLES['mv_load'], METADATA,
                          Column('id', Integer, primary_key=True),
                          Column('run_id', BigInteger, ForeignKey(versioning.columns.run_id), nullable=False),
                          Column('id_db', BigInteger),
                          Column('name', String(100)),
                          Column('geom', Geometry('LINESTRING', 4326)),
                          Column('is_aggregated', Boolean),
                          Column('consumption', String(100)),
                          schema=ding0_schema,
                          comment=prepare_metadatastring_fordb("mv_load")
                          )

    # 14 ding0 mv_grid table
    ding0_mv_grid = Table(DING0_TABLES['mv_grid'], METADATA,
                          Column('id', Integer, primary_key=True),
                          Column('run_id', BigInteger, ForeignKey(versioning.columns.run_id), nullable=False),
                          Column('id_db', BigInteger),
                          Column('geom', Geometry('MULTIPOLYGON', 4326)),
                          Column('name', String(100)),
                          Column('population', BigInteger),
                          Column('voltage_nom', Float(10)),
                          schema=ding0_schema,
                          comment=prepare_metadatastring_fordb("mv_grid")
                          )


    # 15 ding0 mv_station table
    ding0_mv_station = Table(DING0_TABLES['mv_station'], METADATA,
                             Column('id', Integer, primary_key=True),
                             Column('run_id', BigInteger, ForeignKey(versioning.columns.run_id), nullable=False),
                             Column('id_db', BigInteger),
                             Column('geom', Geometry('POINT', 4326)),
                             Column('name', String(100)),
                             schema=ding0_schema,
                             comment=prepare_metadatastring_fordb("mv_station")
                             )

    # 16 ding0 hvmv_transformer table
    ding0_hvmv_transformer = Table(DING0_TABLES['hvmv_transformer'], METADATA,
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
                                   comment=prepare_metadatastring_fordb("hvmv_transformer")
                                   )

    # create all the tables
    METADATA.create_all(engine, checkfirst=True)


def df_sql_write(engine, schema, db_table, dataframe):
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
    schema: DB schema
    """
    # if 'geom' in dataframe.columns:
    #     sql_write_geom = dataframe.filter(items=['geom'])
    #     for i in sql_write_geom['geom']:
    #
    #         # insert_geom = "UPDATE {} SET {}=ST_GeomFromText('{}') WHERE (id={}) ".format(db_table, "geom", i, "1")
    #         insert_geom = "INSERT INTO {} ({}) VALUES (ST_GeomFromText('{}'))".format(db_table, "geom", i)
    #         engine.execute(insert_geom)

    if 'id' in dataframe.columns:
        dataframe.rename(columns={'id':'id_db'}, inplace=True)
        sql_write_df = dataframe.copy()
        sql_write_df.columns = sql_write_df.columns.map(str.lower)
        # sql_write_df = sql_write_df.set_index('id')
        sql_write_df.to_sql(db_table, con=engine, schema=schema, if_exists='append', index=None)
    else:
        sql_write_df = dataframe.copy()
        sql_write_df.columns = sql_write_df.columns.map(str.lower)
        # sql_write_df = sql_write_df.set_index('id')
        sql_write_df.to_sql(db_table, con=engine, schema=schema, if_exists='append', index=None)


def export_df_to_db(engine, schema, df, tabletype):
    """
    Writes values to the connected DB. Values from Pandas data frame.
    Decides which table by tabletype

    Parameters
    ----------
    :param engine:
    :param schema:
    :param df:
    :param tabletype:
    """
    print("Exporting table type : {}".format(tabletype))
    if tabletype == 'line':
        df_sql_write(engine, schema, DING0_TABLES['line'], df)

    elif tabletype == 'lv_cd':
        df = df.drop(['lv_grid_id'], axis=1)
        df['geom'].apply()
        df_sql_write(engine, schema, DING0_TABLES['lv_branchtee'], df)

    elif tabletype == 'lv_gen':
        df_sql_write(engine, schema, DING0_TABLES['lv_generator'], df)

    elif tabletype == 'lv_load':
        df_sql_write(engine, schema, DING0_TABLES['lv_load'], df)

    elif tabletype == 'lv_grid':
        df_sql_write(engine, schema, DING0_TABLES['lv_grid'], df)

    elif tabletype == 'lv_station':
        df_sql_write(engine, schema, DING0_TABLES['lv_station'], df)

    elif tabletype == 'mvlv_trafo':
        df_sql_write(engine, schema, DING0_TABLES['mvlv_transformer'], df)

    elif tabletype == 'mvlv_mapping':
        df_sql_write(engine, schema, DING0_TABLES['mvlv_mapping'], df)

    elif tabletype == 'mv_cd':
        df_sql_write(engine, schema, DING0_TABLES['mv_branchtee'], df)

    elif tabletype == 'mv_cb':
        df_sql_write(engine, schema, DING0_TABLES['mv_circuitbreaker'], df)

    elif tabletype == 'mv_gen':
        df_sql_write(engine, schema, DING0_TABLES['mv_generator'], df)

    elif tabletype == 'mv_load':
        df_sql_write(engine, schema, DING0_TABLES['mv_load'], df)

    elif tabletype == 'mv_grid':
        df_sql_write(engine, schema, DING0_TABLES['mv_grid'], df)

    elif tabletype == 'mv_station':
        df_sql_write(engine, schema, DING0_TABLES['mv_station'], df)

    elif tabletype == 'hvmv_trafo':
        df_sql_write(engine, schema, DING0_TABLES['hvmv_transformer'], df)


def drop_ding0_db_tables(engine, schema):
    """
    Instructions: In order to drop tables all tables need to be stored in METADATA (create tables before dropping them)
    :param engine:
    :param schema:
    :return:
    """
    tables = METADATA.sorted_tables
    reversed_tables = reversed(tables)

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
        for tab in reversed_tables:
            tab.drop(engine, checkfirst=True)
    elif re.fullmatch('[Nn]o', confirmation):
        print("Cancelled dropping of tables")
    else:
        try:
            indlist = confirmation.split(',')
            indlist = list(map(int, indlist))
            print("Please confirm deletion of the following tables:")
            tablist = np.array(reversed_tables)[indlist].tolist()
            for n, tab in enumerate(tablist):
                print("{: 3d}. {}".format(n, tab))
            con2 = input("Please confirm with either of the choices below:\n" +
                         "- yes\n" +
                         "- no")
            if re.fullmatch('[Yy]es', con2):
                for tab in tablist:
                    tab.drop(engine, checkfirst=True)
            elif re.fullmatch('[Nn]o', con2):
                print("Cancelled dropping of tables")
            else:
                print("The input is unclear, no action taken")
        except ValueError:
            print("Confirmation unclear, no action taken")


def db_tables_change_owner(engine, schema):
    tables = METADATA.sorted_tables

    def change_owner(engine, table, role, schema):
        """
        Gives access to database users/ groups

        Parameters
        ----------
        session : sqlalchemy session object
            A valid connection to a database
        table : sqlalchmy Table class definition
            The database table
        role : str
            database role that access is granted to
        """
        tablename = table
        # ToDo: Still using globally def. variable "SCHEMA" inside this func.
        schema = SCHEMA

        grant_str = """ALTER TABLE {schema}.{table}
        OWNER TO {role};""".format(schema=schema, table=tablename,
                          role=role)

        # engine.execute(grant_str)
        engine.execution_options(autocommit=True).execute(grant_str)

    # engine.echo=True

    for tab in tables:
        change_owner(engine, tab, 'oeuser', schema)

    engine.close()


def export_all_dataframes_to_db(engine, schema, srid=None):
    """
    exports all data frames from func. export_network() to the db tables

    Parameters
    ----------
    :param con:
    :param schema:
    """

    if engine.dialect.has_table(engine, DING0_TABLES["versioning"]):

        db_versioning = pd.read_sql_table(DING0_TABLES['versioning'], engine, schema,
                                          columns=['run_id', 'description'])

        if db_versioning.empty:

            metadata_df = pd.DataFrame({'run_id': metadata_json['run_id'],
                                        'description': str(metadata_json)}, index=[0])

            df_sql_write(engine, schema, DING0_TABLES['versioning'], metadata_df)

            # ToDo: UseCase done: 1. run_id from metadata_json to db 2. Insert all data frames
            # ToDo: UseCase maybe?: 1. There are run_id in the db 2. Insert all data frames (might never be the case?)
            # 1
            export_df_to_db(engine, schema, lines, "line")
            # 2
            # export_df_to_db(engine, schema, lv_cd, "lv_cd")
            # # 3
            # export_df_to_db(engine, schema, lv_gen, "lv_gen")
            # # 4
            # export_df_to_db(engine, schema, lv_stations, "lv_station")
            # # 5
            # export_df_to_db(engine, schema, lv_loads, "lv_load")
            # # 6
            # export_df_to_db(engine, schema, lv_grid, "lv_grid")
            # # 7
            # export_df_to_db(engine, schema, mv_cb, "mv_cb")
            # # 8
            # export_df_to_db(engine, schema, mv_cd, "mv_cd")
            # # 9
            # export_df_to_db(engine, schema, mv_gen, "mv_gen")
            # # 10
            # export_df_to_db(engine, schema, mv_stations, "mv_station")
            # # 11
            # export_df_to_db(engine, schema, mv_loads, "mv_load")
            # # 12
            # export_df_to_db(engine, schema, mv_grid, "mv_grid")
            # # 13
            # export_df_to_db(engine, schema, mvlv_trafos, "mvlv_trafo")
            # # 14
            # export_df_to_db(engine, schema, hvmv_trafos, "hvmv_trafo")
            # # 15
            # export_df_to_db(engine, schema, mvlv_mapping, "mvlv_mapping")

        else:
            raise KeyError("run_id already present! No tables are input!")

    else:
        print("There is no " + DING0_TABLES["versioning"] + " table in the schema: " + SCHEMA)



if __name__ == "__main__":

    ##########SQLAlchemy and DB table################
    oedb_engine = connection(section='oedb')
    session = sessionmaker(bind=oedb_engine)()

    # Testing Database
    reiners_engine = connection(section='reiners_db')

    ##########Ding0 Network and NW Metadata################

    # create ding0 Network instance
    nw = NetworkDing0(name='network')


    # choose MV Grid Districts to import
    mv_grid_districts = [3040]

    # run DING0 on selected MV Grid District
    nw.run_ding0(session=session,
                 mv_grid_districts_no=mv_grid_districts)

    # return values from export_network() as tupels
    run_id, nw_metadata, \
    lv_grid, lv_gen, lv_cd, lv_stations, mvlv_trafos, lv_loads, \
    mv_grid, mv_gen, mv_cb, mv_cd, mv_stations, hvmv_trafos, mv_loads, \
    lines, mvlv_mapping = export_network(nw)

    # ToDo: Include the metadata_json variable returned form fun. in export.py
    # any list of NetworkDing0 also provides run_id
    # nw_metadata = json.dumps(nw_metadata)
    metadata_json = json.loads(nw_metadata)

    ######################################################



    # tested with reiners_db
    create_ding0_sql_tables(reiners_engine, SCHEMA)
    # drop_ding0_db_tables(reiners_engine, SCHEMA)
    # db_tables_change_owner(engine, SCHEMA)


    # ToDo: Insert line df: Geometry is wkb and fails to be inserted to db table, get tabletype?
    # parameter: export_network_to_db(engine, schema, df, tabletype, srid=None)
    # export_network_to_db(reiners_engine, SCHEMA, lv_gen, "lv_gen", metadata_json)
    # export_network_to_db(CONNECTION, SCHEMA, mv_stations, "mv_stations", metadata_json)
    export_all_dataframes_to_db(reiners_engine, SCHEMA)

