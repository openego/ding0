"""This file is part of DINGO, the DIstribution Network GeneratOr.
DINGO is a tool to generate synthetic medium and low voltage power
distribution grids based on open data.

It is developed in the project open_eGo: https://openegoproject.wordpress.com

DING0 lives at github: https://github.com/openego/ding0/
The documentation is available on RTD: http://ding0.readthedocs.io"""

__copyright__ = "Reiner Lemoine Institut gGmbH"
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__url__ = "https://github.com/openego/ding0/blob/master/LICENSE"
__author__ = "nesnoj, gplssm"

import numpy as np
import pandas as pd

import re

from sqlalchemy import create_engine
from egoio.db_tables import model_draft as md

from sqlalchemy import MetaData, ARRAY, BigInteger, Boolean, CheckConstraint, Column, Date, DateTime, Float, ForeignKey, ForeignKeyConstraint, Index, Integer, JSON, Numeric, SmallInteger, String, Table, Text, UniqueConstraint, text
from geoalchemy2.types import Geometry, Raster
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql.hstore import HSTORE
from sqlalchemy.dialects.postgresql.base import OID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import ARRAY, DOUBLE_PRECISION, INTEGER, NUMERIC, TEXT, BIGINT, TIMESTAMP, VARCHAR


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
    sql_write_df = sql_write_df.set_index('id')
    sql_write_df.to_sql(db_table.name, con=engine, if_exists='append')


def create_ding0_sql_tables(engine, ding0_schema):
    """
    Create the ding0 tables

    Parameters
    ----------
    engine: :py:mod:`sqlalchemy.engine.base.Engine`
        Sqlalchemy database engine

    schema: :obj:`str`
        The schema in which the tables are to be created
    Returns
    -------
    """

    # versioning table
    versioning = Table(DING0_TABLES['versioning'], metadata,
                       Column('run_id', BigInteger, primary_key=True, autoincrement=False, nullable=False),
                       Column('description', String(3000)),
                       schema=ding0_schema,
                       comment="""This is a comment on table for the ding0 versioning table"""
                       )


    # ding0 lines table
    line = Table(DING0_TABLES['ding0_line'], metadata,
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
                    comment="""This is a commment on table for the ding0 lines table"""
                    )

    # ding0 lv_branchtee table
    lv_branchtee = Table(DING0_TABLES['ding0_lv_branchtee'], metadata,
                    Column('id', Integer, primary_key=True),
                    Column('run_id', BigInteger, ForeignKey(versioning.columns.run_id), nullable=False),
                    Column('id_db', BigInteger),
                    Column('geom', Geometry('POINT', 4326)),
                    Column('name', String(100)),
                    schema=ding0_schema,
                    comment="""This is a commment on table for the ding0 lines table"""
                    )

    # ding0 lv_generator table
    lv_generator = Table(DING0_TABLES['ding0_lv_generator'], metadata,
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
                    comment="""This is a commment on table for the ding0 lines table"""
                    )

    # ding0 lv_load table
    lv_load = Table(DING0_TABLES['ding0_lv_load'], metadata,
                    Column('id', Integer, primary_key=True),
                    Column('run_id', BigInteger, ForeignKey(versioning.columns.run_id), nullable=False),
                    Column('id_db', BigInteger),
                    Column('name', String(100)),
                    Column('lv_grid_id', BigInteger),
                    Column('geom', Geometry('POINT', 4326)),
                    Column('consumption', String(100)),
                    schema=ding0_schema,
                    comment="""This is a commment on table for the ding0 lines table"""
                    )

    # ding0 lv_station table
    lv_station = Table(DING0_TABLES['ding0_lv_station'], metadata,
                    Column('id', Integer, primary_key=True),
                    Column('run_id', BigInteger, ForeignKey(versioning.columns.run_id), nullable=False),
                    Column('id_db', BigInteger),
                    Column('geom', Geometry('POINT', 4326)),
                    Column('name', String(100)),
                    schema=ding0_schema,
                    comment="""This is a commment on table for the ding0 lines table"""
                    )

    # ding0 mvlv_transformer table
    mvlv_transformer = Table(DING0_TABLES['ding0_mvlv_transformer'], metadata,
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
                    comment="""This is a commment on table for the ding0 lines table"""
                    )

    # ding0 mvlv_mapping table
    mvlv_mapping = Table(DING0_TABLES['ding0_mvlv_mapping'], metadata,
                    Column('id', Integer, primary_key=True),
                    Column('run_id', BigInteger, ForeignKey(versioning.columns.run_id), nullable=False),
                    Column('lv_grid_id', BigInteger),
                    Column('lv_grid_name', String(100)),
                    Column('mv_grid_id', BigInteger),
                    Column('mv_grid_name', String(100)),
                    )

    # ding0 mv_branchtee table
    mv_branchtee = Table(DING0_TABLES['ding0_mv_branchtee'], metadata,
                    Column('id', Integer, primary_key=True),
                    Column('run_id', BigInteger, ForeignKey(versioning.columns.run_id), nullable=False),
                    Column('id_db', BigInteger),
                    Column('geom', Geometry('POINT', 4326)),
                    Column('name', String(100)),
                    schema=ding0_schema,
                    comment="""This is a commment on table for the ding0 lines table"""
                    )

    # ding0 mv_circuitbreaker table
    mv_circuitbreaker = Table(DING0_TABLES['ding0_mv_circuitbreaker'], metadata,
                    Column('id', Integer, primary_key=True),
                    Column('run_id', BigInteger, ForeignKey(versioning.columns.run_id), nullable=False),
                    Column('id_db', BigInteger),
                    Column('geom', Geometry('POINT', 4326)),
                    Column('name', String(100)),
                    Column('status', String(10)),
                    schema=ding0_schema,
                    comment="""This is a commment on table for the ding0 lines table"""
                    )

    # ding0 mv_generator table
    mv_generator = Table(DING0_TABLES['ding0_mv_generator'], metadata,
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
                    comment="""This is a commment on table for the ding0 lines table"""
                    )

    # ding0 mv_load table
    mv_load = Table(DING0_TABLES['ding0_mv_load'], metadata,
                    Column('id', Integer, primary_key=True),
                    Column('run_id', BigInteger, ForeignKey(versioning.columns.run_id), nullable=False),
                    Column('id_db', BigInteger),
                    Column('name', String(100)),
                    Column('geom', Geometry('LINESTRING', 4326)),
                    Column('is_aggregated', Boolean),
                    Column('consumption', String(100)),
                    schema=ding0_schema,
                    comment="""This is a commment on table for the ding0 lines table"""
                    )

    # ding0 mv_grid table
    mv_grid = Table(DING0_TABLES['ding0_mv_grid'], metadata,
                    Column('id', Integer, primary_key=True),
                    Column('run_id', BigInteger, ForeignKey(versioning.columns.run_id), nullable=False),
                    Column('id_db', BigInteger),
                    Column('geom', Geometry('LINESTRING', 4326)),
                    Column('name', String(100)),
                    Column('population', BigInteger),
                    Column('voltage_nom', Float(10)),
                    schema=ding0_schema,
                    comment="""This is a commment on table for the ding0 lines table"""
                    )



    # ding0 mv_station table
    mv_station = Table(DING0_TABLES['ding0_mv_station'], metadata,
                    Column('id', Integer, primary_key=True),
                    Column('run_id', BigInteger, ForeignKey(versioning.columns.run_id), nullable=False),
                    Column('id_db', BigInteger),
                    Column('geom', Geometry('LINESTRING', 4326)),
                    Column('name', String(100)),
                    schema=ding0_schema,
                    comment="""This is a commment on table for the ding0 lines table"""
                    )

    # ding0 hvmv_transformer table
    hvmv_transformer = Table(DING0_TABLES['ding0_hvmv_transformer'], metadata,
                    Column('id', Integer, primary_key=True),
                    Column('run_id', BigInteger, ForeignKey(versioning.columns.run_id), nullable=False),
                    Column('id_db', BigInteger),
                    Column('geom', Geometry('LINESTRING', 4326)),
                    Column('name', String(100)),
                    Column('voltage_op', Float(10)),
                    Column('s_nom', Float(10)),
                    Column('x', Float(10)),
                    Column('r', Float(10)),
                    schema=ding0_schema,
                    comment="""This is a commment on table for the ding0 lines table"""
                    )

    # create all the tables
    metadata.create_all(engine, checkfirst=True)

def export_network_to_db(session, schema, table, tabletype, srid):
    dataset = []
    engine = create_engine("sqlite:///myexample.db")
    print("Exporting table type : {}".format(tabletype))
    if tabletype == 'line':
        table.apply(lambda row:
                    session.add(schema.EgoGridDing0Line(
                        run_id=row['run_id'],
                        id_db=row['id'],
                        edge_name=row['edge_name'],
                        grid_name=row['grid_name'],
                        node1=row['node1'],
                        node2=row['node2'],
                        type_kind=row['type_kind'],
                        type_name=row['type_name'],
                        length=row['length'],
                        u_n=row['U_n'],
                        c=row['C'],
                        l=row['L'],
                        r=row['R'],
                        i_max_th=row['I_max_th'],
                        geom=row['geom'],
                    ))
                    , axis=1)

    elif tabletype == 'lv_cd':
        table.apply(lambda row:
                    session.add(schema.EgoGridDing0LvBranchtee(
                        run_id=row['run_id'],
                        id_db=row['id'],
                        name=row['name'],
                        geom="SRID={};{}".format(srid, row['geom']) if row[
                            'geom'] else None,
                    ))
                    , axis=1)

    elif tabletype == 'lv_gen':
        table.apply(lambda row:
                    session.add(schema.EgoGridDing0LvGenerator(
                        run_id=row['run_id'],
                        id_db=row['id'],
                        la_id=row['la_id'],
                        name=row['name'],
                        lv_grid_id=str(row['lv_grid_id']),
                        geom="SRID={};{}".format(srid, row['geom']) if row[
                            'geom'] else None,
                        type=row['type'],
                        subtype=row['subtype'],
                        v_level=row['v_level'],
                        nominal_capacity=row['nominal_capacity'],
                        is_aggregated=row['is_aggregated'],
                        weather_cell_id=row['weather_cell_id'] if not(pd.isnull(row[
                            'weather_cell_id'])) else None,

                    ))
                    , axis=1)

    elif tabletype == 'lv_load':
        table.apply(lambda row:
                    session.add(schema.EgoGridDing0LvLoad(
                        run_id=row['run_id'],
                        id_db=row['id'],
                        name=row['name'],
                        lv_grid_id=row['lv_grid_id'],
                        geom="SRID={};{}".format(srid, row['geom']) if row[
                            'geom'] else None,
                        consumption=row['consumption']
                    ))
                    , axis=1)

    elif tabletype == 'lv_grid':
        table.apply(lambda row:
                    session.add(schema.EgoGridDing0LvGrid(
                        run_id=row['run_id'],
                        id_db=row['id'],
                        name=row['name'],
                        geom="SRID={};{}".format(srid, row['geom']) if row[
                            'geom'] else None,
                        population=row['population'],
                        voltage_nom=row['voltage_nom'],
                    ))
                    , axis=1)

    elif tabletype == 'lv_station':
        table.apply(lambda row:
                    session.add(schema.EgoGridDing0LvStation(
                        run_id=row['run_id'],
                        id_db=row['id'],
                        name=row['name'],
                        geom="SRID={};{}".format(srid, row['geom']) if row[
                            'geom'] else None,
                    ))
                    , axis=1)

    elif tabletype == 'mvlv_trafo':
        table.apply(lambda row:
                    session.add(schema.EgoGridDing0MvlvTransformer(
                        run_id=row['run_id'],
                        id_db=row['id'],
                        name=row['name'],
                        geom="SRID={};{}".format(srid, row['geom']) if row[
                            'geom'] else None,
                        voltage_op=row['voltage_op'],
                        s_nom=row['S_nom'],
                        x=row['X'],
                        r=row['R'],
                    ))
                    , axis=1)

    elif tabletype == 'mvlv_mapping':
        table.apply(lambda row:
                    session.add(schema.EgoGridDing0MvlvMapping(
                        run_id=row['run_id'],
                        lv_grid_id=row['lv_grid_id'],
                        lv_grid_name=row['lv_grid_name'],
                        mv_grid_id=row['mv_grid_id'],
                        mv_grid_name=row['mv_grid_name'],
                    ))
                    , axis=1)

    elif tabletype == 'mv_cd':
        table.apply(lambda row:
                    session.add(schema.EgoGridDing0MvBranchtee(
                        run_id=row['run_id'],
                        id_db=row['id'],
                        name=row['name'],
                        geom="SRID={};{}".format(srid, row['geom']) if row[
                            'geom'] else None,
                    ))
                    , axis=1)

    elif tabletype == 'mv_cb':
        table.apply(lambda row:
                    session.add(schema.EgoGridDing0MvCircuitbreaker(
                        run_id=row['run_id'],
                        id_db=row['id'],
                        name=row['name'],
                        geom="SRID={};{}".format(srid, row['geom']) if row[
                            'geom'] else None,
                        status=row['status'],
                    ))
                    , axis=1)

    elif tabletype == 'mv_gen':
        table.apply(lambda row:
                    session.add(schema.EgoGridDing0MvGenerator(
                        run_id=row['run_id'],
                        id_db=row['id'],
                        name=row['name'],
                        geom="SRID={};{}".format(srid, row['geom']) if row[
                            'geom'] else None,
                        type=row['type'],
                        subtype=row['subtype'],
                        v_level=row['v_level'],
                        nominal_capacity=row['nominal_capacity'],
                        is_aggregated=row['is_aggregated'],
                        weather_cell_id=row['weather_cell_id'] if not(pd.isnull(row[
                            'weather_cell_id'])) else None,
                    ))
                    , axis=1)

    elif tabletype == 'mv_load':
        table.apply(lambda row:
                    session.add(schema.EgoGridDing0MvLoad(
                        run_id=row['run_id'],
                        id_db=row['id'],
                        name=row['name'],
                        geom="SRID={};{}".format(srid, row['geom']) if row[
                            'geom'] else None,
                        is_aggregated=row['is_aggregated'],
                        consumption=row['consumption'],
                    ))
                    , axis=1)

    elif tabletype == 'mv_grid':
        table.apply(lambda row:
                    session.add(schema.EgoGridDing0MvGrid(
                        run_id=row['run_id'],
                        id_db=row['id'],
                        name=row['name'],
                        geom="SRID={};{}".format(srid, row['geom']) if row[
                            'geom'] else None,
                        population=row['population'],
                        voltage_nom=row['voltage_nom'],
                    ))
                    , axis=1)

    elif tabletype == 'mv_station':
        table.apply(lambda row:
                    session.add(schema.EgoGridDing0MvStation(
                        run_id=row['run_id'],
                        id_db=row['id'],
                        name=row['name'],
                        geom="SRID={};{}".format(srid, row['geom']) if row[
                            'geom'] else None,
                    ))
                    , axis=1)

    elif tabletype == 'hvmv_trafo':
        table.apply(lambda row:
                    session.add(schema.EgoGridDing0HvmvTransformer(
                        run_id=row['run_id'],
                        id_db=row['id'],
                        name=row['name'],
                        geom="SRID={};{}".format(srid, row['geom']) if row[
                            'geom'] else None,
                        voltage_op=row['voltage_op'],
                        s_nom=row['S_nom'],
                        x=row['X'],
                        r=row['R'],
                    ))
                    , axis=1)
        # if not engine.dialect.has_table(engine, 'ego_grid_mv_transformer'):
        #     print('helloworld')

    session.commit()


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


class EgoGridDing0Versioning(Base):
    __tablename__ = 'ego_grid_ding0_versioning'
    __table_args__ = {'schema': 'model_draft'}

    id = Column(Integer, primary_key=True)
    run_id = Column(BigInteger, unique=True, nullable=False)
    description = Column(String(3000))


class EgoGridDing0MvStation(Base):
    __tablename__ = 'ego_grid_ding0_mv_station'
    __table_args__ = {'schema': 'model_draft'}

    id = Column(BigInteger, primary_key=True)
    run_id = Column(BigInteger,
                    ForeignKey('model_draft.ego_grid_ding0_versioning.run_id'),
                    nullable=False)
    id_db = Column(BigInteger)
    geom = Column(Geometry('POINT', 4326))
    name = Column(String(100))


class EgoGridDing0HvmvTransformer(Base):
    __tablename__ = 'ego_grid_ding0_hvmv_transformer'
    __table_args__ = {'schema': 'model_draft'}

    id = Column(Integer, primary_key=True)
    run_id = Column(BigInteger,
                    ForeignKey('model_draft.ego_grid_ding0_versioning.run_id'),
                    nullable=False)
    id_db = Column(BigInteger)
    geom = Column(Geometry('POINT', 4326))
    name = Column(String(100))
    voltage_op = Column(Float(10))
    s_nom = Column(Float(10))
    x = Column(Float(10))
    r = Column(Float(10))


class EgoGridDing0Line(Base):
    __tablename__ = 'ego_grid_ding0_line'
    __table_args__ = {'schema': 'model_draft'}

    id = Column(Integer, primary_key=True)
    run_id = Column(BigInteger,
                    ForeignKey('model_draft.ego_grid_ding0_versioning.run_id'),
                    nullable=False)
    id_db = Column(BigInteger)
    edge_name = Column(String(100))
    grid_name = Column(String(100))
    node1 = Column(String(100))
    node2 = Column(String(100))
    type_kind = Column(String(20))
    type_name = Column(String(30))
    length = Column(Float(10))
    u_n = Column(Float(10))
    c = Column(Float(10))
    l = Column(Float(10))
    r = Column(Float(10))
    i_max_th = Column(Float(10))
    geom = Column(Geometry('LINESTRING', 4326))


class EgoGridDing0LvBranchtee(Base):
    __tablename__ = 'ego_grid_ding0_lv_branchtee'
    __table_args__ = {'schema': 'model_draft'}

    id = Column(Integer, primary_key=True)
    run_id = Column(BigInteger,
                    ForeignKey('model_draft.ego_grid_ding0_versioning.run_id'),
                    nullable=False)
    id_db = Column(BigInteger)
    geom = Column(Geometry('POINT', 4326))
    name = Column(String(100))


class EgoGridDing0LvGenerator(Base):
    __tablename__ = 'ego_grid_ding0_lv_generator'
    __table_args__ = {'schema': 'model_draft'}

    id = Column(Integer, primary_key=True)
    run_id = Column(BigInteger,
                    ForeignKey('model_draft.ego_grid_ding0_versioning.run_id'),
                    nullable=False)
    id_db = Column(BigInteger)
    la_id = Column(BigInteger)
    name = Column(String(100))
    lv_grid_id = Column(BigInteger)
    geom = Column(Geometry('POINT', 4326))
    type = Column(String(22))
    subtype = Column(String(22))
    v_level = Column(Integer)
    nominal_capacity = Column(Float(10))
    weather_cell_id = Column(BigInteger)
    is_aggregated = Column(Boolean)


class EgoGridDing0LvGrid(Base):
    __tablename__ = 'ego_grid_ding0_lv_grid'
    __table_args__ = {'schema': 'model_draft'}

    id = Column(Integer, primary_key=True)
    run_id = Column(BigInteger,
                    ForeignKey('model_draft.ego_grid_ding0_versioning.run_id'),
                    nullable=False)
    id_db = Column(BigInteger)
    name = Column(String(100))
    geom = Column(Geometry('MULTIPOLYGON', 4326)) #Todo: check if right srid?
    population = Column(BigInteger)
    voltage_nom = Column(Float(10)) #Todo: Check Datatypes


class EgoGridDing0LvLoad(Base):
    __tablename__ = 'ego_grid_ding0_lv_load'
    __table_args__ = {'schema': 'model_draft'}

    id = Column(Integer, primary_key=True)
    run_id = Column(BigInteger,
                    ForeignKey('model_draft.ego_grid_ding0_versioning.run_id'),
                    nullable=False)
    id_db = Column(BigInteger)
    name = Column(String(100))
    lv_grid_id = Column(Integer)
    geom = Column(Geometry('POINT', 4326))
    consumption = Column(String(100))


class EgoGridDing0MvBranchtee(Base):
    __tablename__ = 'ego_grid_ding0_mv_branchtee'
    __table_args__ = {'schema': 'model_draft'}

    id = Column(Integer, primary_key=True)
    run_id = Column(BigInteger,
                    ForeignKey('model_draft.ego_grid_ding0_versioning.run_id'),
                    nullable=False)
    id_db = Column(BigInteger)
    geom = Column(Geometry('POINT', 4326))
    name = Column(String(100))

class EgoGridDing0MvCircuitbreaker(Base):
    __tablename__ = 'ego_grid_ding0_mv_circuitbreaker'
    __table_args__ = {'schema': 'model_draft'}

    id = Column(Integer, primary_key=True)
    run_id = Column(BigInteger,
                    ForeignKey('model_draft.ego_grid_ding0_versioning.run_id'),
                    nullable=False)
    id_db = Column(BigInteger)
    geom = Column(Geometry('POINT', 4326))
    name = Column(String(100))
    status = Column(String(10))

class EgoGridDing0MvGenerator(Base):
    __tablename__ = 'ego_grid_ding0_mv_generator'
    __table_args__ = {'schema': 'model_draft'}

    id = Column(Integer, primary_key=True)
    run_id = Column(BigInteger,
                    ForeignKey('model_draft.ego_grid_ding0_versioning.run_id'),
                    nullable=False)
    id_db = Column(BigInteger)
    name = Column(String(100))
    geom = Column(Geometry('POINT',  4326))
    type = Column(String(22))
    subtype = Column(String(22))
    v_level = Column(Integer)
    nominal_capacity = Column(Float(10))
    weather_cell_id = Column(BigInteger)
    is_aggregated = Column(Boolean)


class EgoGridDing0MvGrid(Base):
    __tablename__ = 'ego_grid_ding0_mv_grid'
    __table_args__ = {'schema': 'model_draft'}

    id = Column(Integer, primary_key=True)
    run_id = Column(BigInteger,
                    ForeignKey('model_draft.ego_grid_ding0_versioning.run_id'),
                    nullable=False)
    id_db = Column(BigInteger)
    geom = Column(Geometry('MULTIPOLYGON', 4326)) #Todo: check if right srid?
    name = Column(String(100))
    population = Column(BigInteger)
    voltage_nom = Column(Float(10)) #Todo: Check Datatypes


class EgoGridDing0MvLoad(Base):
    __tablename__ = 'ego_grid_ding0_mv_load'
    __table_args__ = {'schema': 'model_draft'}

    id = Column(Integer, primary_key=True)
    run_id = Column(BigInteger,
                    ForeignKey('model_draft.ego_grid_ding0_versioning.run_id'),
                    nullable=False)
    id_db = Column(BigInteger)
    name = Column(String(100))
    geom = Column(Geometry('GEOMETRY', 4326))
    is_aggregated = Column(Boolean)
    consumption = Column(String(100))


class EgoGridDing0MvlvMapping(Base):
    __tablename__ = 'ego_grid_ding0_mvlv_mapping'
    __table_args__ = {'schema': 'model_draft'}

    id = Column(Integer, primary_key=True)
    run_id = Column(BigInteger,
                    ForeignKey('model_draft.ego_grid_ding0_versioning.run_id'),
                    nullable=False)
    lv_grid_id = Column(BigInteger)
    lv_grid_name = Column(String(100))
    mv_grid_id = Column(BigInteger)
    mv_grid_name = Column(String(100))


class EgoGridDing0LvStation(Base):
    __tablename__ = 'ego_grid_ding0_lv_station'
    __table_args__ = {'schema': 'model_draft'}

    id = Column(Integer, primary_key=True)
    run_id = Column(BigInteger,
                    ForeignKey('model_draft.ego_grid_ding0_versioning.run_id'),
                    nullable=False)
    id_db = Column(BigInteger)
    geom = Column(Geometry('POINT', 4326))
    name = Column(String(100))


class EgoGridDing0MvlvTransformer(Base):
    __tablename__ = 'ego_grid_ding0_mvlv_transformer'
    __table_args__ = {'schema': 'model_draft'}

    id = Column(Integer, primary_key=True)
    run_id = Column(BigInteger,
                    ForeignKey('model_draft.ego_grid_ding0_versioning.run_id'),
                    nullable=False)
    id_db = Column(BigInteger)
    geom = Column(Geometry('POINT', 4326))
    name = Column(String(100))
    voltage_op = Column(Float(10))
    s_nom = Column(Float(10))
    x = Column(Float(10))
    r = Column(Float(10))
