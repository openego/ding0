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
#
#
# import json
# import os
#
# from egoio.tools.db import connection
#
# from sqlalchemy import MetaData
#
# # set your Table names
# DING0_TABLES = {'versioning': 'ego_ding0_versioning',
#                 'line': 'ego_ding0_line',
#                 'lv_branchtee': 'ego_ding0_lv_branchtee',
#                 'lv_generator': 'ego_ding0_lv_generator',
#                 'lv_load': 'ego_ding0_lv_load',
#                 'lv_grid': 'ego_ding0_lv_grid',
#                 'lv_station': 'ego_ding0_lv_station',
#                 'mvlv_transformer': 'ego_ding0_mvlv_transformer',
#                 'mvlv_mapping': 'ego_ding0_mvlv_mapping',
#                 'mv_branchtee': 'ego_ding0_mv_branchtee',
#                 'mv_circuitbreaker': 'ego_ding0_mv_circuitbreaker',
#                 'mv_generator': 'ego_ding0_mv_generator',
#                 'mv_load': 'ego_ding0_mv_load',
#                 'mv_grid': 'ego_ding0_mv_grid',
#                 'mv_station': 'ego_ding0_mv_station',
#                 'hvmv_transformer': 'ego_ding0_hvmv_transformer'}
#
# # #########SQLAlchemy and DB table################
# #source
# oedb_engine = connection(section='oedb')
# # Testing Database -> destination
# reiners_engine = connection(section='reiners_db')
#
# REFLICTED_SCHEMA = "model_draft"
# VERSIONING_SCHEMA = "grid"
#
# META = MetaData()
# META.reflect(bind=reiners_engine, schema=REFLICTED_SCHEMA, only=DING0_TABLES['versioning', 'line', 'lv_branchtee',
#                                                                           'lv_generator', 'lv_load', 'lv_grid',
#                                                                           'lv_station', 'mvlv_transformer',
#                                                                           'mvlv_mapping', 'mv_branchtee',
#                                                                           'mv_circuitbreaker', 'mv_generator',
#                                                                           'mv_load', 'mv_grid', 'mv_station',
#                                                                           'hvmv_transformer'])
# # ################################################
#
# tables = META.metadata.tables
# for tbl in tables:
#     print ('##################################')
#     print (tbl)
#     print ( tables[tbl].select())
#     data = oedb_engine.execute(tables[tbl].select()).fetchall()
#     for a in data: print(a)
#     if data:
#         print (tables[tbl].insert())
#         reiners_engine.execute( tables[tbl].insert(), data)

#!/usr/bin/env python

import sys
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

from egoio.tools.db import connection


# # set your Table names
# DING0_TABLES = {'versioning': 'ego_grid_ding0_versioning',
#                 'line': 'ego_grid_ding0_line',
#                 'lv_branchtee': 'ego_grid_ding0_lv_branchtee',
#                 'lv_generator': 'ego_grid_ding0_lv_generator',
#                 'lv_load': 'ego_grid_ding0_lv_load',
#                 'lv_grid': 'ego_grid_ding0_lv_grid',
#                 'lv_station': 'ego_grid_ding0_lv_station',
#                 'mvlv_transformer': 'ego_grid_ding0_mvlv_transformer',
#                 'mvlv_mapping': 'ego_grid_ding0_mvlv_mapping',
#                 'mv_branchtee': 'ego_grid_ding0_mv_branchtee',
#                 'mv_circuitbreaker': 'ego_grid_ding0_mv_circuitbreaker',
#                 'mv_generator': 'ego_grid_ding0_mv_generator',
#                 'mv_load': 'ego_grid_ding0_mv_load',
#                 'mv_grid': 'ego_grid_ding0_mv_grid',
#                 'mv_station': 'ego_grid_ding0_mv_station',
#                 'hvmv_transformer': 'ego_grid_ding0_hvmv_transformer'}

DING0_TABLES = {'mv_generator': 'ego_grid_ding0_mv_generator'}


def get_table_names(t):
    tables = []
    for k, v in t.items():
        tables.append(v)
    return tables


def make_session(engine):
    Session = sessionmaker(bind=engine)
    return Session(), engine


def pull_data(from_db, s_schema, to_db, d_schema, tables):
    source, sengine = make_session(from_db)
    smeta = MetaData(bind=sengine, schema=s_schema)
    destination, dengine = make_session(to_db)

    for table_name in get_table_names(DING0_TABLES):
        print('Processing', table_name)
        print('Pulling schema from source server')
        table = Table(table_name, smeta, autoload=True)
        table.schema = d_schema
        print('Creating table on destination server or schema')
        table.metadata.create_all(dengine, checkfirst=True)
        new_record = quick_mapper(table)
        columns = table.columns.keys()
        print('Transferring records')
        for record in source.query(table).all():
            data = dict(
                [(str(column), getattr(record, column)) for column in columns]
            )
            destination.merge(new_record(**data))
    print('Committing changes')
    destination.commit()


def quick_mapper(table):
    Base = declarative_base()
    class GenericMapper(Base):
        __table__ = table
    return GenericMapper


if __name__ == '__main__':
    # source
    oedb_engine = connection(section='oedb')
    # # Testing Database -> destination
    reiners_engine = connection(section='reiners_db')

    SOURCE_SCHEMA = 'model_draft'
    DESTINATION_SCHEMA = 'grid'
    tables = get_table_names(DING0_TABLES)

    pull_data(oedb_engine, SOURCE_SCHEMA, oedb_engine, DESTINATION_SCHEMA, tables)
