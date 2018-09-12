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
from egoio.db_tables import model_draft as md

#from sqlalchemy import MetaData, ARRAY, BigInteger, Boolean, CheckConstraint, Column, Date, DateTime, Float, ForeignKey, ForeignKeyConstraint, Index, Integer, JSON, Numeric, SmallInteger, String, Table, Text, UniqueConstraint, text
#from geoalchemy2.types import Geometry, Raster
#from sqlalchemy.orm import sessionmaker

from pathlib import Path
import json
import os

#con = connection()

#query orm style
#Session = sessionmaker()
#Session.configure(bind=con)
#session = Session()



#load data from json file
#mds = metadatastring
def load_json():

    # JSON metadatastring folder. Static path for windows
    FOLDER = Path('C:\ego_grid_ding0_metadatastrings')
    print(FOLDER)
    full_dir = os.walk(FOLDER.parent / FOLDER.name)
    jsonmetadata = []

    for jsonfiles in full_dir:
        for jsonfile in jsonfiles:
            #if jsonfile[-4:] == 'json':
            jsonmetadata = jsonfile


    #with open('JSONMETADATA') as f:
        #mds = json.load(f)

load_json()