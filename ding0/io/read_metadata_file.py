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

from egoio.tools.db import connection
#from egoio.db_tables import model_draft as md

from sqlalchemy import MetaData, ARRAY, BigInteger, Boolean, CheckConstraint, Column, Date, DateTime, Float, ForeignKey, ForeignKeyConstraint, Index, Integer, JSON, Numeric, SmallInteger, String, Table, Text, UniqueConstraint, text
from geoalchemy2.types import Geometry, Raster
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

from pathlib import Path
import json
import os

con = connection()

#query orm style
#Session = sessionmaker()
#Session.configure(bind=con)
#session = Session()

Base = declarative_base()
metadata = Base.metadata

# metadatastring file folder. #ToDO: Test if Path works on other os (Tested on Windows7)
# Modify if folder name is different
FOLDER = Path('C:/ego_grid_ding0_metadatastrings')

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


# load data from json file
def load_json_files():
    print(FOLDER)
    full_dir = os.walk(FOLDER.parent / FOLDER.name)
    jsonmetadata = []

    for jsonfiles in full_dir:
        for jsonfile in jsonfiles:
            #if jsonfile[:4] == 'json': #ToDo: Add Execption
            jsonmetadata = jsonfile

    return jsonmetadata


# Prepares the JSON String for the sql comment on table
# mds = metadatastring
def prepare_metadatastring_fordb(table):
    for file in load_json_files():
        JSONFILEPATH = FOLDER / file
        with open(JSONFILEPATH) as f:
            if table in file:
                mds = json.load(f)
                mdsstring = json.dumps(mds, indent=4)
                return mdsstring


# Copy from db_export for testing purpose included just for testing and review
def create_ding0_sql_tables(engine, ding0_schema):
    # versioning table
    versioning = Table(DING0_TABLES['versioning'], metadata,
                       Column('run_id', BigInteger, primary_key=True, autoincrement=False, nullable=False),
                       Column('description', String(3000)),
                       schema=ding0_schema,
                       comment=prepare_metadatastring_fordb("versioning")
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

    # create all the tables
    metadata.create_all(engine, checkfirst=True)

# Test
create_ding0_sql_tables(con, "topology")