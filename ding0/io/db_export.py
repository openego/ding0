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

from egoio.tools.db import connection

import ding0
from ding0.io.export import export_network
from ding0.core import NetworkDing0

from sqlalchemy import MetaData, ARRAY, BigInteger, Boolean, CheckConstraint, Column, Date, DateTime, Float, ForeignKey, ForeignKeyConstraint, Index, Integer, JSON, Numeric, SmallInteger, String, Table, Text, UniqueConstraint, text
from geoalchemy2.types import Geometry, Raster
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


##########SQLAlchemy and DB table################
engine2 = connection(section='oedb')
session = sessionmaker(bind=engine2)()

con = connection()

Base = declarative_base()
metadata = Base.metadata

# Set the Database schema which you want to add the tables to
SCHEMA = "topology"

METADATA_STRING_FOLDER = os.path.join(ding0.__path__[0], 'config', 'metadatastrings')


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

    full_dir = os.walk(METADATA_STRING_FOLDER)
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
        json_file_path = os.path.join(METADATA_STRING_FOLDER, json_file)
        with open(json_file_path, encoding='UTF-8') as jf:
            if table in json_file:
                # included for testing / or logging
                # print("Comment on table: " + table + "\nusing this METADATA file: " + file + "\n")
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

    # versioning table
    versioning = Table(DING0_TABLES['versioning'], metadata,
                       Column('run_id', BigInteger, primary_key=True, autoincrement=False, nullable=False),
                       Column('description', String(6000)),
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


def run_id_in_db(engine, schema, df, db_versioning, tabletype):
    """
    Check if the run_id values from the new data frames are available in the DB.
    Creates new run_id in the db if not exist.
    Filter data frame for column=run_id and compares the values with existing run_id.
    db_run_id values are from the DB table: ego_ding0_versioning.

    Parameters
    ----------
    :param engine: DB connection
    :param schema: DB schema
    :param df: pandas data frame -> gets inserted to the db
    :param db_versioning: pandas df created from the versioning table in the DB
    :param tabletype: select the db table "value relevant in export_df_to_db()"
    """

    db_run_id = db_versioning.filter(items=['run_id'])
    df_run_id = df.filter(items=['run_id'])

    # temp stores all run_id values that are available in the DB
    db_0temp = []
    for j in db_run_id["run_id"]:
        db_0temp.append(j)

    # temp stores run_id value from data frame
    df_1temp = []
    for i in df_run_id["run_id"]:
        if i in db_0temp:
            if i not in df_1temp:
                # the run_id value needs to be only present in the run_id column else the filter
                # might not work correctly
                df_by_run_id = df[df["run_id"] == i]
                export_df_to_db(engine, schema, df_by_run_id, tabletype)
                # stores the run_id(i) from the df in order to compare with the next loop iteration run_id(i) ->
                # df with multiple rows which include the same run_id will not be inserted n times to the db
                df_1temp.append(i)
        elif i not in db_0temp:
            metadata_df = pd.DataFrame({'run_id': i,
                                        'description': str(metadata_json)}, index=[0])
            # create the new run_id from df in db table
            df_sql_write(con, SCHEMA, "ego_ding0_versioning", metadata_df)
            db_0temp.append(i)

            # insert df with the new run_id
            df_by_run_id = df[df["run_id"] == i]
            export_df_to_db(engine, schema, df_by_run_id, tabletype)


def export_network_to_db(engine, schema, df, tabletype, srid=None):
    """
    Exports pre created Pands data frames to a connected database schema.
    Creates new entry in ego_ding0_versioning if the table is empty.
    Checks if the given pandas data frame "run_id" is available in the DB table.

    Parameters
    ----------
    :param engine:
    :param schema:
    :param df:
    :param tabletype:
    :param srid:
    """

    db_versioning = pd.read_sql_table(DING0_TABLES['versioning'], engine, schema,
                                      columns=['run_id', 'description'])

    if engine.dialect.has_table(engine, DING0_TABLES["versioning"]):

        run_id_in_db(engine, schema, df, db_versioning, tabletype)

    else:
        print("There is no " + DING0_TABLES["versioning"] + " table in the schema: " + SCHEMA)


def drop_ding0_db_tables(engine, schema=SCHEMA):
    """
    Instructions: In order to drop tables all tables need to be stored in metadata (create tables before dropping them)
    :param engine:
    :param schema:
    :return:
    """
    tables = metadata.sorted_tables
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


def db_tables_change_owner(engine, schema=SCHEMA):
    tables = metadata.sorted_tables

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


def execute_export_network_to_db(con, schema=SCHEMA):
    """
    exports all data frames to the db tables

    Parameters
    ----------
    :param con:
    :param schema:
    """

    # 1
    export_network_to_db(con, schema, lines, "line", metadata_json)
    # 2
    export_network_to_db(con, schema, lv_cd, "lv_cd", metadata_json)
    # 3
    export_network_to_db(con, schema, lv_gen, "lv_gen", metadata_json)
    # 4
    export_network_to_db(con, schema, lv_stations, "lv_station", metadata_json)
    # 5
    export_network_to_db(con, schema, lv_loads, "lv_load", metadata_json)
    # 6
    export_network_to_db(con, schema, lv_grid, "lv_grid", metadata_json)
    # 7
    export_network_to_db(con, schema, mv_cb, "mv_cb", metadata_json)
    # 8
    export_network_to_db(con, schema, mv_cd, "mv_cd", metadata_json)
    # 9
    export_network_to_db(con, schema, mv_gen, "mv_gen", metadata_json)
    # 10
    export_network_to_db(con, schema, mv_stations, "mv_station", metadata_json)
    # 11
    export_network_to_db(con, schema, mv_loads, "mv_load", metadata_json)
    # 12
    export_network_to_db(con, schema, mv_grid, "mv_grid", metadata_json)
    # 13
    export_network_to_db(con, schema, mvlv_trafos, "mvlv_trafo", metadata_json)
    # 14
    export_network_to_db(con, schema, hvmv_trafos, "hvmv_trafo", metadata_json)
    # 15
    export_network_to_db(con, schema, mvlv_mapping, "mvlv_mapping", metadata_json)

    
if __name__ == "__main__":

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
    # nw_metadata = json.dumps(nw.metadata)
    metadata_json = json.loads(nw_metadata)

    ######################################################

    # metadatastring file folder. #ToDO: Test if Path works on other os (Tested on Windows7) and Change to not static
    # Modify if folder name is different -> use: "/"
    FOLDER = Path('/ego_grid_ding0_metadatastrings')

    # tested with reiners_db
    create_ding0_sql_tables(con, "topology")
    # drop_ding0_db_tables(con, "topology")
    # db_tables_change_owner(con, "topology")


    # tested with reiners_db
    create_ding0_sql_tables(con, "topology")
    # drop_ding0_db_tables(con)
    # db_tables_change_owner(con, "topology")

    # ToDo: Insert line df: Geometry is wkb and fails to be inserted to db table, get tabletype?
    # parameter: export_network_to_db(engine, schema, df, tabletype, srid=None)
    export_network_to_db(con, SCHEMA, lv_gen, "lv_gen", metadata_json)
    # export_network_to_db(con, SCHEMA, mv_stations, "mv_stations", metadata_json)

