import os
import math
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pyproj import Proj, transform
import logging

logger = logging.getLogger('ding0')
from ding0.tools.logger import get_default_home_dir
from ding0.core.network.grids import MVGridDing0

use_gpd = False
use_ctx = False

if 'READTHEDOCS' not in os.environ:
    use_gpd = True
    try:
        import geopandas as gpd
    except:
        use_gpd = False

    use_ctx = True
    try:
        import contextily as ctx
    except:
        use_ctx = False


def plot_mv_topology(grid, subtitle='', filename=None, testcase='load',
                     line_color=None, node_color='type',
                     limits_cb_lines=None, limits_cb_nodes=None,
                     background_map=True):
    """ Draws MV grid graph using networkx

    Parameters
    ----------
    grid : :obj:`MVGridDing0`
        MV grid to plot.
    subtitle : :obj:`str`
        Extend plot's title by this string.
    filename : :obj:`str`
        If provided, the figure will be saved and not displayed (default path: ~/.ding0/).
        A prefix is added to the file name.
    testcase : :obj:`str`
        Defines which case is to be used. Refer to config_calc.cfg to see further
        assumptions for the cases. Possible options are:
        
        * 'load' (default)
          Heavy-load-flow case
        * 'feedin'
          Feedin-case
    line_color : :obj:`str`
        Defines whereby to choose line colors. Possible options are:

        * 'loading'
          Line color is set according to loading of the line in heavy load case.
          You can use parameter `limits_cb_lines` to adjust the color range.
        * None (default)
          Lines are plotted in black. Is also the fallback option in case of
          wrong input.
    node_color : :obj:`str`
        Defines whereby to choose node colors. Possible options are:

        * 'type' (default)
          Node color as well as size is set according to type of node
          (generator, MV station, etc.). Is also the fallback option in case of
          wrong input.
        * 'voltage'
          Node color is set according to voltage deviation from 1 p.u..
          You can use parameter `limits_cb_nodes` to adjust the color range.
    limits_cb_lines : :obj:`tuple`
        Tuple with limits for colorbar of line color. First entry is the
        minimum and second entry the maximum value. E.g. pass (0, 1) to
        adjust the colorbar to 0..100% loading.
        Default: None (min and max loading are used).
    limits_cb_nodes : :obj:`tuple`
        Tuple with limits for colorbar of nodes. First entry is the
        minimum and second entry the maximum value. E.g. pass (0.9, 1) to
        adjust the colorbar to 90%..100% voltage.
        Default: None (min and max loading are used).
    background_map : bool, optional
        If True, a background map is plotted (default: stamen toner light).
        The additional package `contextily` is needed for this functionality.
        Default: True

    Note
    -----
    WGS84 pseudo mercator (epsg:3857) is used as coordinate reference system (CRS).
    Therefore, the drawn graph representation may be falsified!
    """

    def set_nodes_style_and_position(nodes):

        # TODO: MOVE settings to config
        # node types (name of classes)
        node_types = ['MVStationDing0',
                      'LVStationDing0',
                      'LVLoadAreaCentreDing0',
                      'MVCableDistributorDing0',
                      'GeneratorDing0',
                      'GeneratorFluctuatingDing0',
                      'CircuitBreakerDing0',
                      'n/a']
        
        # node styles
        colors_dict = {'MVStationDing0': '#f2ae00',
                       'LVStationDing0': 'grey',
                       'LVLoadAreaCentreDing0': '#fffc3d',
                       'MVCableDistributorDing0': '#000000',
                       'GeneratorDing0': '#00b023',
                       'GeneratorFluctuatingDing0': '#0078b0',
                       'CircuitBreakerDing0': '#c20000',
                       'n/a': 'orange'}
        sizes_dict = {'MVStationDing0': 120,
                      'LVStationDing0': 7,
                      'LVLoadAreaCentreDing0': 30,
                      'MVCableDistributorDing0': 5,
                      'GeneratorDing0': 50,
                      'GeneratorFluctuatingDing0': 50,
                      'CircuitBreakerDing0': 50,
                      'n/a': 5}
        zindex_by_type = {'MVStationDing0': 16,
                          'LVStationDing0': 12,
                          'LVLoadAreaCentreDing0': 11,
                          'MVCableDistributorDing0': 13,
                          'GeneratorDing0': 14,
                          'GeneratorFluctuatingDing0': 14,
                          'CircuitBreakerDing0': 15,
                          'n/a': 10}

        # dict of node class names: list of nodes
        nodes_by_type = {_: [] for _ in node_types}
        # dict of node class names: list of node-individual color
        node_colors_by_type = {_: [] for _ in node_types}
        # dict of node class names: list of node-individual size
        node_sizes_by_type = {_: [] for _ in node_types}
        node_sizes_by_type['all'] = []
        # dict of nodes:node-individual positions
        nodes_pos = {}

        for n in nodes:
            if type(n).__name__ in node_types:
                nodes_by_type[type(n).__name__].append(n)
                node_colors_by_type[type(n).__name__].append(colors_dict[type(n).__name__])
                node_sizes_by_type[type(n).__name__].append(sizes_dict[type(n).__name__])
                node_sizes_by_type['all'].append(sizes_dict[type(n).__name__])
            else:
                nodes_by_type['n/a'].append(n)
                node_colors_by_type['n/a'].append(colors_dict['n/a'])
                node_sizes_by_type['n/a'].append(sizes_dict['n/a'])
                node_sizes_by_type['all'].append(sizes_dict['n/a'])
            nodes_pos[n] = (n.geo_data.x, n.geo_data.y)

        return node_types, nodes_by_type, node_colors_by_type,\
               node_sizes_by_type, zindex_by_type, nodes_pos

    def reproject_nodes(nodes_pos, model_proj='4326'):
        inProj = Proj(init='epsg:{srid}'.format(srid=model_proj))
        outProj = Proj(init='epsg:3857')
        nodes_pos2 = {}
        for k, v in nodes_pos.items():
            x2, y2 = transform(inProj, outProj,
                               v[0],
                               v[1])
            nodes_pos2[k] = (x2, y2)
        return nodes_pos2

    def plot_background_map(ax):
        url = ctx.sources.ST_TONER_LITE
        xmin, xmax, ymin, ymax = ax.axis()
        basemap, extent = ctx.bounds2img(xmin, ymin, xmax, ymax,
                                         zoom=12, url=url)
        ax.imshow(basemap, extent=extent, interpolation='bilinear', zorder=0)
        ax.axis((xmin, xmax, ymin, ymax))

    def plot_region_data(ax):
        # get geoms of MV grid district, load areas and LV grid districts
        mv_grid_district = gpd.GeoDataFrame({'geometry': grid.grid_district.geo_data},
                                            crs={'init': 'epsg:{srid}'.format(srid=model_proj)})
        load_areas = gpd.GeoDataFrame({'geometry': [la.geo_area for la in grid.grid_district.lv_load_areas()]},
                                      crs={'init': 'epsg:{srid}'.format(srid=model_proj)})
        lv_grid_districts = gpd.GeoDataFrame({'geometry': [lvgd.geo_data
                                                           for la in grid.grid_district.lv_load_areas()
                                                           for lvgd in la.lv_grid_districts()]},
                                             crs={'init': 'epsg:{srid}'.format(srid=model_proj)})

        # reproject to WGS84 pseudo mercator
        mv_grid_district = mv_grid_district.to_crs(epsg=3857)
        load_areas = load_areas.to_crs(epsg=3857)
        lv_grid_districts = lv_grid_districts.to_crs(epsg=3857)

        # plot
        mv_grid_district.plot(ax=ax, color='#ffffff', alpha=0.2, edgecolor='k', linewidth=2, zorder=2)
        load_areas.plot(ax=ax, color='#fffea3', alpha=0.1, edgecolor='k', linewidth=0.5, zorder=3)
        lv_grid_districts.plot(ax=ax, color='#ffffff', alpha=0.05, edgecolor='k', linewidth=0.5, zorder=4)

    if not isinstance(grid, MVGridDing0):
        logger.warning('Sorry, but plotting is currently only available for MV grids but you did not pass an'
                       'instance of MVGridDing0. Plotting is skipped.')
        return

    g = grid._graph
    model_proj = grid.network.config['geo']['srid']

    if testcase == 'feedin':
        case_idx = 1
    else:
        case_idx = 0

    nodes_types, nodes_by_type, node_colors_by_type, node_sizes_by_type, zindex_by_type, nodes_pos =\
        set_nodes_style_and_position(g.nodes())

    # reproject to WGS84 pseudo mercator
    nodes_pos = reproject_nodes(nodes_pos, model_proj=model_proj)

    plt.figure(figsize=(9, 6))
    ax = plt.gca()

    if line_color == 'loading':
        edges_color = []
        for n1, n2 in g.edges():
            edge = g.adj[n1][n2]
            if hasattr(edge['branch'], 's_res'):
                edges_color.append(edge['branch'].s_res[case_idx] * 1e3 /
                                   (3 ** 0.5 * edge['branch'].type['U_n'] * edge['branch'].type['I_max_th']))
            else:
                edges_color.append(0)
        edges_cmap = plt.get_cmap('jet')
        #edges_cmap.set_over('#952eff')
    else:
        edges_color = ['black'] * len(list(grid.graph_edges()))
        edges_cmap = None

    # plot nodes by voltage
    if node_color == 'voltage':
        voltage_station = grid._station.voltage_res[case_idx]
        nodes_color = []
        for n in g.nodes():
            if hasattr(n, 'voltage_res'):
                nodes_color.append(n.voltage_res[case_idx])
            else:
                nodes_color.append(voltage_station)

        if testcase == 'feedin':
            nodes_cmap = plt.get_cmap('Reds')
            nodes_vmax = voltage_station + float(grid.network.config
                                                 ['mv_routing_tech_constraints']
                                                 ['mv_max_v_level_fc_diff_normal'])
            nodes_vmin = voltage_station
        else:
            nodes_cmap = plt.get_cmap('Reds_r')
            nodes_vmin = voltage_station - float(grid.network.config
                                                 ['mv_routing_tech_constraints']
                                                 ['mv_max_v_level_lc_diff_normal'])
            nodes_vmax = voltage_station

        nodes = nx.draw_networkx_nodes(g,
                                       pos=nodes_pos,
                                       node_color=nodes_color,
                                       # node_shape='o', # TODO: Add additional symbols here
                                       cmap=nodes_cmap,
                                       vmin=nodes_vmin,
                                       vmax=nodes_vmax,
                                       node_size=node_sizes_by_type['all'],
                                       linewidths=0.25,
                                       ax=ax)
        nodes.set_zorder(10)
        nodes.set_edgecolor('k')

        # colorbar nodes
        if limits_cb_nodes is None:
            limits_cb_nodes = (math.floor(min(nodes_color)*100)/100,
                               math.ceil(max(nodes_color)*100)/100)
        v_range = np.linspace(limits_cb_nodes[0], limits_cb_nodes[1], 101)
        cb_voltage = plt.colorbar(nodes, boundaries=v_range,
                                  ticks=v_range[0:101:10],
                                  fraction=0.04, pad=0.1)
        cb_voltage.set_clim(vmin=limits_cb_nodes[0],
                            vmax=limits_cb_nodes[1])
        cb_voltage.set_label('Node voltage deviation in %', size='smaller')
        cb_voltage.ax.tick_params(labelsize='smaller')

    # plot nodes by type
    else:
        for node_type in nodes_types:
            if len(nodes_by_type[node_type]) != 0:
                nodes = nx.draw_networkx_nodes(g,
                                               nodelist=nodes_by_type[node_type],
                                               pos=nodes_pos,
                                               # node_shape='o', # TODO: Add additional symbols here
                                               node_color=node_colors_by_type[node_type],
                                               cmap=None,
                                               vmin=None,
                                               vmax=None,
                                               node_size=node_sizes_by_type[node_type],
                                               linewidths=0.25,
                                               label=node_type,
                                               ax=ax)
                nodes.set_zorder(zindex_by_type[node_type])
                nodes.set_edgecolor('k')

    edges = nx.draw_networkx_edges(g,
                                   pos=nodes_pos,
                                   edge_color=edges_color,
                                   edge_cmap=edges_cmap,
                                   edge_vmin=0,
                                   edge_vmax=1,
                                   #width=1,
                                   ax=ax)
    edges.set_zorder(5)

    if line_color == 'loading':
        # colorbar edges
        if limits_cb_lines is None:
            limits_cb_lines = (math.floor(min(edges_color)*100)/100,
                               math.ceil(max(edges_color)*100)/100)
        loading_range = np.linspace(limits_cb_lines[0], limits_cb_lines[1], 101)
        cb_loading = plt.colorbar(edges, boundaries=loading_range,
                                  ticks=loading_range[0:101:10],
                                  fraction=0.04, pad=0.04)
        cb_loading.set_clim(vmin=limits_cb_lines[0],
                            vmax=limits_cb_lines[1])
        cb_loading.set_label('Line loading in % of nominal capacity', size='smaller')
        cb_loading.ax.tick_params(labelsize='smaller')

    if use_ctx and background_map:
        plot_background_map(ax=ax)
    if use_gpd:
        plot_region_data(ax=ax)

    plt.legend(fontsize=7)
    plt.title('MV Grid District {id} - {st}'.format(id=grid.id_db,
                                                    st=subtitle))

    # hide axes labels (coords)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    if filename is None:
        plt.tight_layout()
        plt.show()
    else:
        path = os.path.join(get_default_home_dir(), 'ding0_grid_{id}_{filename}'.format(id=str(grid.id_db),
                                                                                        filename=filename))
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info('==> Figure saved to {path}'.format(path=path))
