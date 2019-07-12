"""This file is part of DINGO, the DIstribution Network GeneratOr.
DINGO is a tool to generate synthetic medium and low voltage power
distribution grids based on open data.

It is developed in the project open_eGo: https://openegoproject.wordpress.com

DING0 lives at github: https://github.com/openego/ding0/
The documentation is available on RTD: http://ding0.readthedocs.io"""

__copyright__ = "Reiner Lemoine Institut gGmbH"
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__url__ = "https://github.com/openego/ding0/blob/master/LICENSE"
__author__ = "jh-RLI"

import numpy as np
import pandas as pd
import json
import re
from egoio.tools.db import connection
from ding0.io.export import export_network
from ding0.core import NetworkDing0
from ding0.io.ding0_db_tables import DING0_TABLES, METADATA, create_ding0_sql_tables
from ding0.io.io_settings import exporter_config
from geoalchemy2.types import Geometry, WKTElement
from sqlalchemy.orm import sessionmaker


# init SRID
SRID = None


def create_wkt_element(geom):
    """
    Use GeoAlchemy's WKTElement to create a geom with SRID
    GeoAlchemy2 WKTElement (PostGis func:ST_GeomFromText)

    Parameters
    ----------
    geom: Shaply geometry from script export.py

    Returns
    -------
    None : None
        Returns None if the data frame does not contain any geometry
    """

    if geom is not None:
        if SRID is None:
            try:
                from ding0.io.ding0_pkl2db import PICKLE_SRID
                return WKTElement(geom, srid=PICKLE_SRID, extended=True)
            except:
                print('You need to provide a SRID or PICKLE_SRID')
                print('PICKLE_SRID will be set to 4326')
                PICKLE_SRID = 4326
                return WKTElement(geom, srid=PICKLE_SRID, extended=True)
        else:
            return WKTElement(geom, srid=SRID, extended=True)
    else:
        return None


def df_sql_write(engine, schema, db_table, dataframe, SRID=None, geom_type=None):
    """
    Convert data frames such that their column names
    are made small and the index is renamed 'id_db' so as to
    correctly load its data to its appropriate sql table. Also handles the
    upload to a DB data frames with different geometry types.

    NOTE: This function does not check if the data frame columns
    matches the db_table fields, if they do not then no warning
    is given.

    Parameters
    ----------
    pandas.DataFrame
        dataframe : The pandas dataframe to be transferred to its
                    apprpritate db_table

    db_table: :py:mod:`sqlalchemy.sql.schema.Table`
        A table instance definition from sqlalchemy.
        NOTE: This isn't an orm definition

    engine: :py:mod:`sqlalchemy.engine.base.Engine`
        Sqlalchemy database engine

    schema: str
        The schema in which the tables are to be created

    geom_type: str
        Prameter for handling data frames with different geometry types
    SRID: int
        The current srid provided by the ding0 network
    """

    # rename data frame column DB like
    if 'id' in dataframe.columns:
        dataframe.rename(columns={'id':'id_db'}, inplace=True)
        sql_write_df = dataframe.copy()
        sql_write_df.columns = sql_write_df.columns.map(str.lower)
        # sql_write_df = sql_write_df.set_index('id_db')

        # Insert pd data frame with geom column
        if 'geom' in dataframe.columns:
            if geom_type is 'POINT':
                sql_write_df['geom'] = sql_write_df['geom'].apply(create_wkt_element)
                sql_write_df.to_sql(db_table, con=engine, schema=schema, if_exists='append', index=None,
                                    dtype={'geom': Geometry('POINT', srid=int(SRID))})

            elif geom_type is 'POLYGON':
                sql_write_df['geom'] = sql_write_df['geom'].apply(create_wkt_element)
                sql_write_df.to_sql(db_table, con=engine, schema=schema, if_exists='append', index=None,
                                    dtype={'geom': Geometry('POLYGON', srid=int(SRID))})

            elif geom_type is 'MULTIPOLYGON':
                sql_write_df['geom'] = sql_write_df['geom'].apply(create_wkt_element)
                sql_write_df.to_sql(db_table, con=engine, schema=schema, if_exists='append', index=None,
                                    dtype={'geom': Geometry('MULTIPOLYGON', srid=int(SRID))})

            elif geom_type is 'LINESTRING':
                sql_write_df['geom'] = sql_write_df['geom'].apply(create_wkt_element)
                sql_write_df.to_sql(db_table, con=engine, schema=schema, if_exists='append', index=None,
                                    dtype={'geom': Geometry('LINESTRING', srid=int(SRID))})

            elif geom_type is 'GEOMETRY':
                sql_write_df['geom'] = sql_write_df['geom'].apply(create_wkt_element)
                sql_write_df.to_sql(db_table, con=engine, schema=schema, if_exists='append', index=None,
                                    dtype={'geom': Geometry('GEOMETRY', srid=int(SRID))})

        else:
            sql_write_df.to_sql(db_table, con=engine, schema=schema, if_exists='append', index=None)
    # If the Dataframe does not contain id named column (like already named id_db)
    else:
        sql_write_df = dataframe.copy()
        sql_write_df.columns = sql_write_df.columns.map(str.lower)
        # sql_write_df = sql_write_df.set_index('id')

        if 'geom' in dataframe.columns:
            # Insert pd Dataframe with geom column
            if geom_type is 'POINT':
                sql_write_df['geom'] = sql_write_df['geom'].apply(create_wkt_element)

                sql_write_df.to_sql(db_table, con=engine, schema=schema, if_exists='append', index=None,
                                    dtype={'geom': Geometry('POINT', srid=int(SRID))})

            elif geom_type is 'POLYGON':
                sql_write_df['geom'] = sql_write_df['geom'].apply(create_wkt_element)
                sql_write_df.to_sql(db_table, con=engine, schema=schema, if_exists='append', index=None,
                                    dtype={'geom': Geometry('POLYGON', srid=int(SRID))})

            elif geom_type is 'MULTIPOLYGON':
                sql_write_df['geom'] = sql_write_df['geom'].apply(create_wkt_element)
                sql_write_df.to_sql(db_table, con=engine, schema=schema, if_exists='append', index=None,
                                    dtype={'geom': Geometry('MULTIPOLYGON', srid=int(SRID))})

            elif geom_type is 'LINESTRING':
                sql_write_df['geom'] = sql_write_df['geom'].apply(create_wkt_element)
                sql_write_df.to_sql(db_table, con=engine, schema=schema, if_exists='append', index=None,
                                    dtype={'geom': Geometry('LINESTRING', srid=int(SRID))})

            elif geom_type is 'GEOMETRY':
                sql_write_df['geom'] = sql_write_df['geom'].apply(create_wkt_element)
                sql_write_df.to_sql(db_table, con=engine, schema=schema, if_exists='append', index=None,
                                    dtype={'geom': Geometry('GEOMETRY', srid=int(SRID))})

        else:
            sql_write_df.to_sql(db_table, con=engine, schema=schema, if_exists='append', index=None)


def export_df_to_db(engine, schema, df, tabletype, srid=None):
    """
    Writes values to the connected DB. Values from Pandas data frame.
    Decides which table by tabletype

    Parameters
    ----------
    engine: sqlalchemy.engine.base.Engine`
        Sqlalchemy database engine
    schema : str
        The schema in which the tables are to be created
    pandas.DataFrame
        df : pandas data frame
    tabletype : str
        Set the destination table where the pd data frame will be stored in
    srid: int
        The current srid provided by the ding0 network
    """
    print("Exporting table type : {}".format(tabletype))
    if tabletype is 'line':
        df_sql_write(engine, schema, DING0_TABLES['line'], df, srid, 'LINESTRING')

    elif tabletype is 'lv_cd':
        df = df.drop(['lv_grid_id'], axis=1)
        df_sql_write(engine, schema, DING0_TABLES['lv_branchtee'], df, srid, 'POINT')

    elif tabletype is 'lv_gen':
        df_sql_write(engine, schema, DING0_TABLES['lv_generator'], df, srid, 'POINT')

    elif tabletype is 'lv_load':
        df_sql_write(engine, schema, DING0_TABLES['lv_load'], df, srid, 'POINT')

    elif tabletype is 'lv_grid':
        df_sql_write(engine, schema, DING0_TABLES['lv_grid'], df, srid, 'GEOMETRY')

    elif tabletype is 'lv_station':
        df_sql_write(engine, schema, DING0_TABLES['lv_station'], df, srid, 'POINT')

    elif tabletype is 'mvlv_trafo':
        df_sql_write(engine, schema, DING0_TABLES['mvlv_transformer'], df, srid, 'POINT')

    elif tabletype is 'mvlv_mapping':
        df_sql_write(engine, schema, DING0_TABLES['mvlv_mapping'], df, srid)

    elif tabletype is 'mv_cd':
        df_sql_write(engine, schema, DING0_TABLES['mv_branchtee'], df, srid, 'POINT')

    elif tabletype is 'mv_cb':
        df_sql_write(engine, schema, DING0_TABLES['mv_circuitbreaker'], df, srid, 'POINT')

    elif tabletype is 'mv_gen':
        df_sql_write(engine, schema, DING0_TABLES['mv_generator'], df, srid, 'POINT')

    elif tabletype is 'mv_load':
        df_sql_write(engine, schema, DING0_TABLES['mv_load'], df, srid, 'GEOMETRY')

    elif tabletype is 'mv_grid':
        df_sql_write(engine, schema, DING0_TABLES['mv_grid'], df, srid, 'MULTIPOLYGON')

    elif tabletype is 'mv_station':
        df_sql_write(engine, schema, DING0_TABLES['mv_station'], df, srid, 'POINT')

    elif tabletype is 'hvmv_trafo':
        df_sql_write(engine, schema, DING0_TABLES['hvmv_transformer'], df, srid, 'POINT')


# ToDo: function works but throws unexpected error (versioning tbl dosent exists)
def drop_ding0_db_tables(engine):
    """
    Instructions: In order to drop tables all tables need to be stored in METADATA (create tables before dropping them)
    Drops the tables in the schema where they have been created.
    Parameters
    ----------
    engine: sqlalchemy.engine.base.Engine`
        Sqlalchemy database engine
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
        engine: sqlalchemy session object
            A valid connection to a database
        schema: The schema in which the tables are to be created
        table : sqlalchmy Table class definition
            The database table
        role : str
            database role that access is granted to
        """
        tablename = table

        grant_str = """ALTER TABLE {schema}.{table}
        OWNER TO {role};""".format(schema=schema, table=tablename.name,
                                   role=role)

        # engine.execute(grant_str)
        engine.execution_options(autocommit=True).execute(grant_str)

    # engine.echo=True

    for tab in tables:
        change_owner(engine, tab, 'oeuser', schema)


def export_all_dataframes_to_db(engine, schema, network=None, srid=None):
    """
    exports all data frames from func. export_network() to the db tables
    This works with a completely generated ding0 network(all grid districts have to be generated at once),
    all provided DataFrames will be uploaded.

    Instructions:
    1. Create a database connection to the "OEDB" for example use the "from egoio.tools.db import connection" function
    2. Create a SA session: session = sessionmaker(bind=oedb_engine)()
    3. Create a ding0 network instance: nw = NetworkDing0(name='network')
    4. SET the srid from network config: SRID = str(int(nw.config['geo']['srid']))
    5. Choose the grid_districts for the ding0 run (nothing chosen all grid_districts will be imported)
        mv_grid_districts = [3040, 3045]
    6. run ding0 on selected mv_grid_district
    7. call function export_network from export.py -> this provides the run_id, network metadata as json
        and all ding0 result data as pandas data frames
    8. json.loads the metadata, it is needed to provide the values for the
        versioning table
    9. Create a database connection to your database for example use the "from egoio.tools.db import connection" function
    10. SET the SCHEMA you want to use within the connected database
    11. Create the ding0 sql tables: create_ding0_sql_tables(engine, SCHEMA)
    12. Call the function: export_all_dataframes_to_db(engine, SCHEMA) with your destination database and SCHEMA
    additionally:
    13. If you used the "OEDB" as destination database change the table owner using the function:
        db_tables_change_owner(engine, schema)
    14. If you need to drop the table call the function drop_ding0_db_tables(engine, schema) immediately after
        the called create function:  create_ding0_sql_tables(oedb_engine, SCHEMA)
                                    drop_ding0_db_tables(oedb_engine, SCHEMA)
    15. Check if all metadata strings are present to the current folder and added as SQL comment on table

    Parameters
    ----------
    engine : sqlalchemy.engine.base.Engine
        Sqlalchemy database engine
    schema : str
        The schema in which the tables are to be created
    network: namedtuple
        All the return values(Data Frames) from export_network()
    srid: int
        The current srid provided by the ding0 network
    """

    if engine.dialect.has_table(engine, DING0_TABLES["versioning"], schema=schema):

        db_versioning = pd.read_sql_table(DING0_TABLES['versioning'], engine, schema,
                                          columns=['run_id', 'description'])
        # Use for another run with different run_id
        # if metadata_json['run_id'] not in db_versioning['run_id']:
        # Use if just one run_id should be present to the DB table
        if db_versioning.empty:
            # json.load the metadata_json
            metadata_json = json.loads(network.metadata_json)
            # this leads to wrong run_id if run_id is SET in __main__ -> 'run_id': metadata_json['run_id']
            # try:
            metadata_df = pd.DataFrame({'run_id': metadata_json['run_id'],
                                        'description': str(metadata_json)}, index=[0])
            df_sql_write(engine, schema, DING0_TABLES['versioning'], metadata_df)
            # except:
            #     print(metadata_json['run_id'])
            #     metadata_df = pd.DataFrame({'run_id': metadata_json['run_id'],
            #                                 'description': str(metadata_json)}, index=[0])
            #     df_sql_write(engine, schema, DING0_TABLES['versioning'], metadata_df)

            # 1
            export_df_to_db(engine, schema, network.lines, "line", srid)
            # 2
            export_df_to_db(engine, schema, network.lv_cd, "lv_cd", srid)
            # 3
            export_df_to_db(engine, schema, network.lv_gen, "lv_gen", srid)
            # 4
            export_df_to_db(engine, schema, network.lv_stations, "lv_station", srid)
            # 5
            export_df_to_db(engine, schema, network.lv_loads, "lv_load", srid)
            # 6
            export_df_to_db(engine, schema, network.lv_grid, "lv_grid", srid)
            # 7
            export_df_to_db(engine, schema, network.mv_cb, "mv_cb", srid)
            # 8
            export_df_to_db(engine, schema, network.mv_cd, "mv_cd", srid)
            # 9
            export_df_to_db(engine, schema, network.mv_gen, "mv_gen", srid)
            # 10
            export_df_to_db(engine, schema, network.mv_stations, "mv_station", srid)
            # 11
            export_df_to_db(engine, schema, network.mv_loads, "mv_load", srid)
            # 12
            export_df_to_db(engine, schema, network.mv_grid, "mv_grid", srid)
            # 13
            export_df_to_db(engine, schema, network.mvlv_trafos, "mvlv_trafo", srid)
            # 14
            export_df_to_db(engine, schema, network.hvmv_trafos, "hvmv_trafo", srid)
            # 15
            export_df_to_db(engine, schema, network.mvlv_mapping, "mvlv_mapping", srid)

        else:
            raise KeyError("a run_id already present! No tables are input!")

    else:
        print("WARNING: There is no " + DING0_TABLES["versioning"] + " table in the schema: " + schema)


def export_all_pkl_to_db(engine, schema, network, srid, grid_no=None):
    """
    This function basically works the same way export_all_dataframes_to_db() does.
    It is implemented to handel the diffrent ways of executing the functions:
        If grids are loaded form pickle files a for loop is included and every grid district will be uploaded one after
        another. This chances the requirements for this function.

    Parameters
    ----------
    engine : sqlalchemy.engine.base.Engine
        Sqlalchemy database engine
    schema : str
        The schema in which the tables are to be created
    network: namedtuple
        All the return values(Data Frames) from export_network()
    srid: int
        The current srid provided by the ding0 network
    grid_no: int
        The Griddistrict number
    """

    if engine.dialect.has_table(engine, DING0_TABLES["versioning"], schema=schema):

        db_versioning = pd.read_sql_table(DING0_TABLES['versioning'], engine, schema,
                                          columns=['run_id', 'description'])

        metadata_json = json.loads(network.metadata_json)

        if db_versioning.empty:
            print("run_id: " + str(metadata_json['run_id']))

            metadata_df = pd.DataFrame({'run_id': metadata_json['run_id'],
                                        'description': str(metadata_json)}, index=[0])
            df_sql_write(engine, schema, DING0_TABLES['versioning'], metadata_df)

            # 1
            export_df_to_db(engine, schema, network.lines, "line", srid)
            # 2
            export_df_to_db(engine, schema, network.lv_cd, "lv_cd", srid)
            # 3
            export_df_to_db(engine, schema, network.lv_gen, "lv_gen", srid)
            # 4
            export_df_to_db(engine, schema, network.lv_stations, "lv_station", srid)
            # 5
            export_df_to_db(engine, schema, network.lv_loads, "lv_load", srid)
            # 6
            export_df_to_db(engine, schema, network.lv_grid, "lv_grid", srid)
            # 7
            export_df_to_db(engine, schema, network.mv_cb, "mv_cb", srid)
            # 8
            export_df_to_db(engine, schema, network.mv_cd, "mv_cd", srid)
            # 9
            export_df_to_db(engine, schema, network.mv_gen, "mv_gen", srid)
            # 10
            export_df_to_db(engine, schema, network.mv_stations, "mv_station", srid)
            # 11
            export_df_to_db(engine, schema, network.mv_loads, "mv_load", srid)
            # 12
            export_df_to_db(engine, schema, network.mv_grid, "mv_grid", srid)
            # 13
            export_df_to_db(engine, schema, network.mvlv_trafos, "mvlv_trafo", srid)
            # 14
            export_df_to_db(engine, schema, network.hvmv_trafos, "hvmv_trafo", srid)
            # 15
            export_df_to_db(engine, schema, network.mvlv_mapping, "mvlv_mapping", srid)

            print('Griddistrict_' + str(grid_no) + '_has been exported to the database')
        else:
            print("run_id: " + str(metadata_json['run_id']))
            # 1
            export_df_to_db(engine, schema, network.lines, "line", srid)
            # 2
            export_df_to_db(engine, schema, network.lv_cd, "lv_cd", srid)
            # 3
            export_df_to_db(engine, schema, network.lv_gen, "lv_gen", srid)
            # 4
            export_df_to_db(engine, schema, network.lv_stations, "lv_station", srid)
            # 5
            export_df_to_db(engine, schema, network.lv_loads, "lv_load", srid)
            # 6
            export_df_to_db(engine, schema, network.lv_grid, "lv_grid", srid)
            # 7
            export_df_to_db(engine, schema, network.mv_cb, "mv_cb", srid)
            # 8
            export_df_to_db(engine, schema, network.mv_cd, "mv_cd", srid)
            # 9
            export_df_to_db(engine, schema, network.mv_gen, "mv_gen", srid)
            # 10
            export_df_to_db(engine, schema, network.mv_stations, "mv_station", srid)
            # 11
            export_df_to_db(engine, schema, network.mv_loads, "mv_load", srid)
            # 12
            export_df_to_db(engine, schema, network.mv_grid, "mv_grid", srid)
            # 13
            export_df_to_db(engine, schema, network.mvlv_trafos, "mvlv_trafo", srid)
            # 14
            export_df_to_db(engine, schema, network.hvmv_trafos, "hvmv_trafo", srid)
            # 15
            export_df_to_db(engine, schema, network.mvlv_mapping, "mvlv_mapping", srid)

            print('Griddistrict_' + str(grid_no) + '_has been exported to the database')

    else:
        print("WARNING: There is no " + DING0_TABLES["versioning"] + " table in the schema: " + schema)


if __name__ == "__main__":

    # #########SQLAlchemy and DB table################
    # provide a config-file with valid connection credentials to access a Database.
    # the config-file should be located in your user directory within a folder named '.config'.
    oedb_engine = connection(section='vpn_oedb')
    session = sessionmaker(bind=oedb_engine)()

    # Set the Database schema which you want to add the tables to.
    # Configure the SCHEMA in config file located in: ding0/config/exporter_config.cfg .
    SCHEMA = exporter_config['EXPORTER_DB']['SCHEMA']

    # hardset for testing
    # SCHEMA = "public"

    # #########Ding0 Network################
    # create ding0 Network instance
    nw = NetworkDing0(name='network')

    # srid
    # ToDo: Check why converted to int and string
    # SRID = str(int(nw.config['geo']['srid']))
    SRID = int(nw.config['geo']['srid'])

    # choose MV Grid Districts to import, use list of integers
    # Multiple grids f. e.: grids = list(range(1, 3609)) - 1 to 3608(all of the existing)
    # Single grids f. e.: grids = [2]
    mv_grid_districts = list(range(2, 6))

    # run DING0 on selected MV Grid District
    nw.run_ding0(session=session,
                 mv_grid_districts_no=mv_grid_districts)

    # return values from export_network() as tupels
    network = export_network(nw)

    #####################################################
    # Creates all defined tables
    create_ding0_sql_tables(oedb_engine, SCHEMA)
    drop_ding0_db_tables(oedb_engine)
    # db_tables_change_owner(oedb_engine, SCHEMA)

    # ########################### !!! Mind existing tables in DB SCHEMA!!! #######################################
    # Export all Dataframes returned form export_network(nw) to DB
    # export_all_dataframes_to_db(oedb_engine, SCHEMA, network=network, srid=SRID)
