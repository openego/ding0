#TODO: Merge Maps.Try easy example for gdfs to graph. Expand.
#TODO: Merge Maps idea: Porque no meter los nodos normales con la data que necesitan
#TODO: Calculate Load per Building

crs = 'EPSG:3035'

import matplotlib.cm as cm
import networkx as nx
import numpy as np
from shapely.ops import transform
import osmnx as ox
import pandas as pd
import geopandas as gpd
from osmnx.save_load import graph_to_gdfs,gdfs_to_graph
from shapely.geometry import Point,LineString,Polygon,MultiPolygon
from shapely.ops import nearest_points
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import spatial
from egoio.db_tables import openstreetmap
from egoio.tools import db
from sqlalchemy.orm import sessionmaker
import shapely
from shapely.wkb import dumps, loads
import matplotlib.pyplot as plt
from geopandas.tools import sjoin
import geoplot
from shapely.strtree import STRtree
from functools import partial
import pyproj



def import_footprints_area():

    #Import and project footprint area
    gdf = ox.footprints.footprints_from_place(place = 'Tempelhof, Berlin, Germany')
    gdf_proj = ox.project_gdf(gdf)
    #fig, ax = ox.plot_shape(gdf_proj)

    gdf_save = gdf.drop(labels='nodes', axis=1)
    #ox.save_gdf_shapefile(gdf_save,'tempelhof_buildings.shp')

    # calculate the area in projected units (meters) of each building footprint, then display first five
    gdf_proj['area'] = gdf_proj.area
    head = gdf_proj['area'].head()

    #Return type of building. "yes", "retail" ...
    #Type = gdf_proj.building

    #Possible usable data from Polygon for GeoRef. in OSM-LandUse
    gdf_proj['centroids'] = gdf_proj.geometry.representative_point()
    #
    gdf_proj = gdf_proj.loc[:, ['nodes', 'area', 'geometry', 'height', 'centroids', 'voltage']]

    return gdf_proj

def get_street_graph():
    #MultiDiGraph
    street_graph = ox.graph_from_place('Tempelhof, Berlin, Germany', network_type='drive')
   # ox.plot_graph(street_graph)
    ox.save_graph_shapefile(street_graph,'streetgraphs')
    return street_graph

def make_plot(place, point, network_type='drive', bldg_color='orange', dpi=40,
              dist=805, default_width=4, street_widths=None):
    gdf = ox.footprints.footprints_from_point(point=point, distance=dist)
    fig, ax = ox.plot_figure_ground(point=point, dist=dist, network_type=network_type, default_width=default_width,
                                    street_widths=street_widths, save=False, show=False, close=True)
    fig, ax = ox.footprints.plot_footprints(gdf, fig=fig, ax=ax, color=bldg_color, set_bounds=False,
                                save=True, show=False, close=True, filename=place, dpi=dpi)

def merge_maps(gdf_build,graph_streets,tr_x,tr_y):
    #TODO:
    # convert trafos in nx nodes (dictionaries) -> add edge
    # connect builduings to streets
    # convert res_gdf into a graph,
    # save df to shp,
    # calc_geo_branches_in_buffer(node, mv_grid, radius, radius_inc, proj) in ding0/tools/geo
    # find_nearest_conn_objects(node_shp, branches, proj, conn_dist_weight, debug, branches_only=False):

    coord_trafos = zip(tr_x,tr_y)
    trafos_dict = dict(zip(np.arange(0,len(trafo_posx)),trafo_posx, trafo_posy))

    for i in range(0,len(tr_x)):
        pass

    gdf_street = graph_to_gdfs(graph_streets)
    interesections = gpd.GeoDataFrame(geometry=gdf_street[0]['geometry'])
    streets = gpd.GeoDataFrame(geometry=gdf_street[1]['geometry'])

    #Find nearest point between builidings and streets
    #1. unary union of the gdf_build geomtries
    #pts3 = gdf_build.geometry.unary_union
    ps3 = streets.geometry.unary_union
    build_0 = gdf_build.representative_point().iloc[0,]
    origin,dest = nearest_points(gdf_build.representative_point().iloc[0,], ps3)


    res_gdf = gdfs_to_graph(pd.concat([gdf_build,interesections]),streets)
    ox.plot_shape(res_gdf)
    return res_gdf

def find_mv_clusters_kmeans(gdf_build, plot=False):
    #https://github.com/srafay/Machine_Learning_A-Z/blob/master/Part%204%20-%20Clustering/Section%2024%20-%20K-Means%20Clustering/kmeans.pyv
    x =(gdf_build.representative_point().values.x).reshape(-1,1)
    y =(gdf_build.representative_point().values.y).reshape(-1,1)
    X = np.concatenate((x, y), axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X[:,0], X[:,1], test_size=0.2, random_state=0)

    # Feature Scaling
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train.reshape(-1,1))
    X_test = sc_X.transform(X_test.reshape(-1,1))
    sc_y = StandardScaler()
    y_train = sc_y.fit_transform(y_train.reshape(-1,1))

    kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
    y_kmeans = kmeans.fit_predict(X)

    # Visualising the clusters
    if plot == True:
        plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
        plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
        plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
        plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4')
        plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c='magenta', label='Cluster 5')
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
        plt.title('Clusters of customers')
        plt.xlabel('Annual Income (k$)')
        plt.ylabel('Spending Score (1-100)')
        plt.legend()
        plt.show()

    return kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1]

def osm_lu_import():
    """
    Sectors
    1: Residential
    2: Retail
    3: Industrial
    4: Agricultural
    """
    engine = db.connection(readonly=True)
    session = sessionmaker(bind=engine)()
    table = openstreetmap.OsmDeuPolygonUrban
    query = session.query(table.gid, table.sector, table.geom,)
    #TODO: Properly import table.geom (Multipolygons in wkb)
    geom_data = pd.read_sql_query(query.statement, query.session.bind)
    #geom_data = geom_data[0:500]
    #This operation takes too long
    for i, rows in geom_data.iterrows():
        geom_data['geom'][i] = shapely.wkb.loads(str(geom_data['geom'][i]), hex=True)


    geom_data = gpd.GeoDataFrame(geom_data, geometry= geom_data['geom'])
    geom_data_test = geom_data.drop(columns='geom')
    geom_data_test.to_file("osm_landuse.geojson", driver='GeoJSON')
    return geom_data

def find_building_sector(buildings, osm_lu):
    """
    returns a GPD df with the centroids as geometries and the sector they belong to
    """
    """ TAKEN FROM DING0
    from functools import partial
    import pyproj
    
    project = partial(
        pyproj.transform,
        pyproj.Proj(init='epsg:4326'),
        pyproj.Proj(init='epsg:26913'))
    """

    buildings['geometry'] = buildings.representative_point()

    proj1 = partial(
        pyproj.transform,
        pyproj.Proj(init='epsg:4326'),  # source coordinate system
        pyproj.Proj(init='epsg:3035'))

    #buildings_proj = buildings.apply(lambda x: transform(proj1, x.geometry), axis=1)
    buildings_proj = pd.read_csv('./building_proj')
    buildings.geometry = buildings_proj
    sector_table = sjoin(buildings, osm_lu, how='inner', op = 'within')
    return sector_table

def calculate_load_per_building(sector_table, gdf):
    """
    sector_table: Table containing the centroids and the sector of each building
    gdf_refined: df['building'] contains the shapes of each building

    return: A table containing the Load for every building

    idea: calculate area in gdf_refined, apply function W/m², append that column, do sjoin

    """
    gdf['area'] = ox.project_gdf(gdf).area
    gdf['load'] = ''
    for i, rows in gdf.iterrows():
        if gdf.iloc[i,]['sector'] == 1:
            gdf.iloc[i,]['load'] = 120 * gdf.iloc[i,]['area']
        elif gdf.iloc[i,]['sector'] == 2:
            gdf.iloc[i,]['load'] = 125 * gdf.iloc[i,]['area']
        elif gdf.iloc[i,]['sector'] == 3:
            gdf.iloc[i,]['load'] = 150 * gdf.iloc[i,]['area']
        elif gdf.iloc[i,]['sector'] == 4:
            gdf.iloc[i,]['load'] = 150 * gdf.iloc[i,]['area']

    return gdf

def calc_geo_branches_in_buffer(gdf, street_graph, radius, radius_inc):
    #TODO: implement for all points in geom_data
    # TODO: Projektion muss übereinsitmmen
    # TODO: Add the other python module
    """
    My version of ding0.tools.geo.calc_geo_branches_in_buffer

    Determines branches in nodes' associated graph that are at least partly
    within buffer of `radius` from `node`.

    If there are no nodes, the buffer is successively extended by `radius_inc`
    until nodes are found.

    Parameters
    ----------
    node : LVStationDing0, GeneratorDing0, or CableDistributorDing0
        origin node (e.g. LVStationDing0 object) with associated shapely object
        (attribute `geo_data`) in any CRS (e.g. WGS84)
    radius : float
        buffer radius in m
    radius_inc : float
        radius increment in m
    proj : :obj:`int`
        pyproj projection object: nodes' CRS to equidistant CRS
        (e.g. WGS84 -> ETRS)

    Returns
    -------
    :obj:`list` of :networkx:`NetworkX Graph Obj< >`
        List of branches (NetworkX branch objects)

    """


    crossings, streets = ox.graph_to_gdfs(street_graph)
    # streets.geometry = Geoseries of Linestrings
    tree = STRtree(list(streets.geometry))
    nodes = []
    node_connections = []

    for i,row in gdf.iterrows():
        center = gdf_refined.iloc[i,].geometry.representative_point()
        branches = []
        while not branches:
            branches = tree.query(center.buffer(radius))
            radius += radius_inc
        a, b = nearest_points(gdf_refined.iloc[i,].geometry.representative_point(), branches[0])
        c = LineString([a, b])

        nodes.append(center)
        node_connections.append(c)

    n_gpd = gpd.GeoDataFrame(geometry=nodes)
    nc_gpd = gpd.GeoDataFrame(geometry=node_connections)


    rdf = gpd.GeoDataFrame(pd.concat([streets, nc_gpd], ignore_index=True), crs=streets.crs)
    rdf.gdf_name = ""
    prp = gpd.GeoDataFrame(pd.concat([crossings, n_gpd], ignore_index=True), crs=crossings.crs)
    prp.gdf_name = ""

    G = gdfs_to_graph(rdf, prp)

    """
    Minimal Working Example:
    df.drop(df.columns.difference(['a','b']), 1, inplace=True)
    
    
    
    """


    return branches


#osm_lu_import()
osm_lu = gpd.read_file("./osm_landuse.geojson")

#gdf = import_footprints_area()
gdf = gpd.read_file("data/tempelhof_buildings.shp/tempelhof_buildings.shp")

gdf_refined = gdf[['type','geometry']]
gdf_refined['buildings'] = gdf.geometry #Used to store the Polygons. Needed after sjoin with

street_graph = get_street_graph()
"""
#edges = gpd.read_file("data/streetgraphs/edges/edges.shp")
#nodes = gpd.read_file("data/streetgraphs/nodes/nodes.shp")
#G = nx.read_shp('test.shp')
"""

#trafo_posx, trafo_posy = find_mv_clusters_kmeans(gdf,False)
#sector_table = find_building_sector(gdf_refined,osm_lu)
#calculate_load_per_building(sector_table)

calc_geo_branches_in_buffer(gdf_refined, street_graph, 0, 1e-4)
#merge_maps(gdf,street_graph,trafo_posx, trafo_posy)

#TODO: n_trafos = find_numb_trafos(gdf), get land-use data




##############Trash
def find_mv_clusters_kd_tree(gdf):
    # I create the Tree from my list of point
    """
    zipped = list(zip(gdf.geometry.representative_point().x, gdf.geometry.representative_point().y))
    kdtree = spatial.KDTree(zipped)   # I calculate the nearest neighbours and their distance
    neigh = kdtree.query(zipped, k=8) # tuple containing a matrix with neibors on same column
    neigh[1][i,1]
    kdtree.data[neigh[1][1, 1]]
    hola = 'papa'
"""
    centroids = gdf['geometry'].apply(lambda g:[g.centroid.x,g.centroid.y]).tolist()# I create the Tree from my list of point
    kdtree = spatial.KDTree(centroids)# I calculate the nearest neighbours and their distance
    neigh = kdtree.query(centroids, k=8)


### Info über projektionen
# WGS84 (conformal) to ETRS (equidistant) projection
proj1 = partial(
        pyproj.transform,
        pyproj.Proj(init='epsg:4326'),  # source coordinate system
        pyproj.Proj(init='epsg:3035'))  # destination coordinate system

# ETRS (equidistant) to WGS84 (conformal) projection
proj2 = partial(
        pyproj.transform,
        pyproj.Proj(init='epsg:3035'),  # source coordinate system
        pyproj.Proj(init='epsg:4326'))  # destination coordinate system

