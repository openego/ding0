#from dingo.core import NetworkDingo
from dingo.core.network import *
from dingo.core.structure import *

from oemof.core import energy_system as es
from oemof import db

from shapely import geometry as geom

import pandas as pd
import time

#import station data from NEXT
sql = '''SELECT id,
         lon,
         lat
         FROM app_dingo_wd.trans_test;'''

#trans_data = db_import(sql)
conn = db.connection(db_section='ontohub_wdb')
#trans_data = pd.read_sql_query(sql, conn, index_col='id')
columns = ['id', 'lon', 'lat']
trans_data = pd.read_sql_table('trans_test', conn, schema='app_dingo_wd',
                                index_col='id', columns=columns)

#init station data
#buses = {}
#transformers = {}
buses = []
transformers = []
#millis1 = int(round(time.time() * 1000))
for idx, row in trans_data.iterrows():
    bus_temp_in = BusDingo(uid='bus_trans_in'+str(idx),
                           geo_data = geom.Point([row['lon'], row['lat']]))
    bus_temp_out = BusDingo(uid='bus_trans_out'+str(idx),
                           geo_data = geom.Point([row['lon'], row['lat']]))
    trans_temp = TransformerDingo(uid='trans'+str(idx),
                                  inputs=[bus_temp_in],
                                  outputs=[bus_temp_out],
                                  #outputs=bus_temp.uid,
                                  #TODO: To connect VNB TO UNB, define FROM and TO busses here
                                  #trans_id = row['trans_id'],
                                  v_level = 'hmv',
                                  geo_data = geom.Point([row['lon'], row['lat']]))
    #buses[bus_temp_in.uid] = bus_temp_in
    #buses[bus_temp_out.uid] = bus_temp_out
    #transformers[trans_temp.uid] = trans_temp
    transformers.append(trans_temp)
    buses.append(bus_temp_in)
    buses.append(bus_temp_out)
#millis2 = int(round(time.time() * 1000))
#print(millis2-millis1)
components = transformers
entities = components + buses
#entities = list(buses.values())+list(transformers.values())

network = NetworkDingo(buses=buses, transformers=transformers)
energysystem = es.EnergySystem(entities=entities)

graph = network.convert_to_networkx()
network.draw_networkx(graph)
#millis3 = int(round(time.time() * 1000))
#print(millis3-millis1)

#energysystem.plot_as_graph(labels=1)