# """This file is part of DINGO, the DIstribution Network GeneratOr.
# DINGO is a tool to generate synthetic medium and low voltage power
# distribution grids based on open data.
#
# It is developed in the project open_eGo: https://openegoproject.wordpress.com
#
# DING0 lives at github: https://github.com/openego/ding0/
# The documentation is available on RTD: http://ding0.readthedocs.io"""
#
# __copyright__ = "Reiner Lemoine Institut gGmbH"
# __license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
# __url__ = "https://github.com/openego/ding0/blob/master/LICENSE"
# __author__ = "jh-RLI"

import os

from sqlalchemy import create_engine, MetaData, Table, exc
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

from egoio.tools.db import connection

import ding0
from ding0.io.db_export import prepare_metadatastring_fordb
from ding0.io.ego_scenario_log import write_scenario_log

# set your Table names
DING0_TABLES = {'versioning': 'ego_grid_ding0_versioning',
                'line': 'ego_grid_ding0_line',
                'lv_branchtee': 'ego_grid_ding0_lv_branchtee',
                'lv_generator': 'ego_grid_ding0_lv_generator',
                'lv_load': 'ego_grid_ding0_lv_load',
                'lv_grid': 'ego_grid_ding0_lv_grid',
                'lv_station': 'ego_grid_ding0_lv_station',
                'mvlv_transformer': 'ego_grid_ding0_mvlv_transformer',
                'mvlv_mapping': 'ego_grid_ding0_mvlv_mapping',
                'mv_branchtee': 'ego_grid_ding0_mv_branchtee',
                'mv_circuitbreaker': 'ego_grid_ding0_mv_circuitbreaker',
                'mv_generator': 'ego_grid_ding0_mv_generator',
                'mv_load': 'ego_grid_ding0_mv_load',
                'mv_grid': 'ego_grid_ding0_mv_grid',
                'mv_station': 'ego_grid_ding0_mv_station',
                'hvmv_transformer': 'ego_grid_ding0_hvmv_transformer'}


def get_table_names(t):
    tables = []
    for k, v in t.items():
        tables.append(v)
    return tables


def make_session(engine):
    Session = sessionmaker(bind=engine)
    return Session(), engine


def migrate_tables_to_destination(from_db, s_schema, to_db, d_schema, runid=None):
    """
    Note: This function will throw a exception caused by the already existing index.
    Functionality is still given.

    Copys the table from source to the destination database.schema

    Step-by-Step:
    1. Set up the connection using the egoio.tools.db -> connection() function
    2. SET the SOURCE_SCHEMA and DESTINATION_SCHEMA
    3. Insert your table (key: names) to dict like DING0_TABLES
    4. Call the function get_table_names() with your Table dictionary as parameter save the result in
        variable "tables = get_table_names(dict)"
    5. For ding0 data set the RUN_ID
    6. Save the dynamic path to the metadata_string.json in METADATA_STRING_FOLDER´
        Note: Metadata_string file names need to contain the the table name See:
        https://github.com/openego/ding0/tree/features/stats-export/ding0/io/metadatastrings
    7. Call the function with parameters like:
        migrate_tables_to_destination(oedb_engine, SOURCE_SCHEMA, oedb_engine, DESTINATION_SCHEMA, RUN_ID)
    8. In function migrate_tables_to_destination() check the function write_scenario_log()
    9. Check if the tables in your source schema exist and named equally to the table dict like in DING0_TABLES{}
    Parameters
    ----------
    from_db:
    s_schema:
    to_db:
    d_schema:
    runid:
    """
    source, sengine = make_session(from_db)
    smeta = MetaData(bind=sengine, schema=s_schema)
    destination, dengine = make_session(to_db)

    for table_name in get_table_names(DING0_TABLES):
        print('Processing', table_name)
        print('Pulling schema from source server')
        table = Table(table_name, smeta, autoload=True)
        print('Creating table on destination server or schema')
        try:
            table.schema = d_schema
            table.metadata.create_all(dengine, checkfirst=True)
        except exc.ProgrammingError:
            print("WARNING: The Index on the table already exists, warning can be ignored.")
        table.schema = s_schema
        new_record = quick_mapper(table)
        columns = table.columns.keys()
        print('Transferring records')
        for record in source.query(table).all():
            data = dict(
                [(str(column), getattr(record, column)) for column in columns]
            )
            table.schema = d_schema
            destination.merge(new_record(**data))

        print('Committing changes')
        destination.commit()

        rows = destination.query(table.c.run_id).count()
        json_tbl_name = []
        for k,v in DING0_TABLES.items():
            if v == table_name:
                json_tbl_name.append(k)
        metadata_string_json = prepare_metadatastring_fordb(json_tbl_name[0])
        write_scenario_log(oedb_engine, 'open_eGo', runid, 'output', s_schema, table_name, 'db_export.py',
                           entries=rows, comment='versioning', metadata=metadata_string_json)


def quick_mapper(table):
    Base = declarative_base()

    class GenericMapper(Base):
        __table__ = table
    return GenericMapper


def db_tables_change_owner(engine, schema):
    DECLARATIVE_BASE = declarative_base()
    METADATA = DECLARATIVE_BASE.metadata

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


if __name__ == '__main__':
    # source
    oedb_engine = connection(section='oedb')
    # # Testing Database -> destination
    reiners_engine = connection(section='reiners_db')

    SOURCE_SCHEMA = 'model_draft'
    DESTINATION_SCHEMA = 'grid'
    tables = get_table_names(DING0_TABLES)

    # Enter the current run_id, Inserted in scenario_log
    RUN_ID = '20181022185643'

    # Metadata folder Path
    METADATA_STRING_FOLDER = os.path.join(ding0.__path__[0], 'io', 'metadatastrings')

    migrate_tables_to_destination(oedb_engine, SOURCE_SCHEMA, oedb_engine, DESTINATION_SCHEMA, RUN_ID)
    # db_tables_change_owner(oedb_engine, DESTINATION_SCHEMA)
