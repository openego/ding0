import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import networkx as nx
from pyproj import Proj, transform, Transformer
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
                      'MVLoadDing0',  # PAUL new
                      'LVLoadAreaCentreDing0',
                      'MVCableDistributorDing0',
                      'GeneratorDing0',
                      'GeneratorFluctuatingDing0',
                      'CircuitBreakerDing0',
                      'n/a']

        # node styles
        colors_dict = {'MVStationDing0': '#f2ae00',
                       'LVStationDing0': 'grey',
                       'MVLoadDing0': 'cyan',  # PAUL new
                       'LVLoadAreaCentreDing0': '#fffc3d',
                       'MVCableDistributorDing0': '#ffa500',  # PAUL new'#000000',
                       'GeneratorDing0': '#00b023',
                       'GeneratorFluctuatingDing0': '#0078b0',
                       'CircuitBreakerDing0': '#c20000',
                       'n/a': 'orange'}
        sizes_dict = {'MVStationDing0': 40,
                      'LVStationDing0': 3,
                      'MVLoadDing0': 3,  # PAUL new
                      'LVLoadAreaCentreDing0': 25,
                      'MVCableDistributorDing0': 3,
                      'GeneratorDing0': 15,
                      'GeneratorFluctuatingDing0': 15,
                      'CircuitBreakerDing0': 15,
                      'n/a': 3}
        zindex_by_type = {'MVStationDing0': 3,
                          'LVStationDing0': 12,
                          'MVLoadDing0': 12,  # PAUL new
                          'LVLoadAreaCentreDing0': 2,
                          'MVCableDistributorDing0': 9,
                          'GeneratorDing0': 2,
                          'GeneratorFluctuatingDing0': 2,
                          'CircuitBreakerDing0': 3,
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

        return node_types, nodes_by_type, node_colors_by_type, \
               node_sizes_by_type, zindex_by_type, nodes_pos

    def reproject_nodes(nodes_pos, model_proj='3035'):  # changed from 4326 to new CRS 3035
        # PAUL new: replace transform with Transformer in pyproj, to keep syntax up-to-date
        transformer = Transformer.from_crs(int(model_proj), 3857, always_xy=True)
        nodes_pos2 = {}
        for k, v in nodes_pos.items():
            nodes_pos2[k] = (transformer.transform(v[0], v[1]))
        return nodes_pos2

    def plot_background_map(ax):
        url = ctx.sources.ST_TONER_LITE
        xmin, xmax, ymin, ymax = ax.axis()
        basemap, extent = ctx.bounds2img(xmin, ymin, xmax, ymax,
                                         zoom=12, url=url)
        ax.imshow(basemap, extent=extent, interpolation='bilinear', zorder=0)
        ax.axis((xmin, xmax, ymin, ymax))

    # regions style by type
    mvgd_style = {'facecolor': '#ffffff', 'alpha': 1, 'edgecolor': '#fdcc02', 'linewidth': 1, 'zorder': 1}
    la_style_sat = {'facecolor': '#ffffde', 'alpha': 1, 'edgecolor': 'k', 'linewidth': 0.2, 'zorder': 1}
    la_style_reg = {'facecolor': '#e1c938', 'alpha': 1, 'edgecolor': 'k', 'linewidth': 0.2, 'zorder': 1}
    la_style_agg = {'facecolor': '#c3c3c3', 'alpha': 1, 'edgecolor': 'k', 'linewidth': 0.2, 'zorder': 1}

    def plot_region_data(ax):
        # get geoms of MV grid district, load areas and LV grid districts
        mv_grid_district = gpd.GeoDataFrame({'geometry': grid.grid_district.geo_data},
                                            crs={'init': 'epsg:{srid}'.format(srid=model_proj)})
        load_areas_sat = gpd.GeoDataFrame({'geometry': [la.geo_area for la in grid.grid_district.lv_load_areas()
                                                        if la.is_satellite]},
                                      crs={'init': 'epsg:{srid}'.format(srid=model_proj)})
        load_areas_reg = gpd.GeoDataFrame({'geometry': [la.geo_area for la in grid.grid_district.lv_load_areas()
                                                        if not la.is_satellite and not la.is_aggregated]},
                                          crs={'init': 'epsg:{srid}'.format(srid=model_proj)})
        load_areas_agg = gpd.GeoDataFrame({'geometry': [la.geo_area for la in grid.grid_district.lv_load_areas()
                                                        if la.is_aggregated]},
                                          crs={'init': 'epsg:{srid}'.format(srid=model_proj)})
        lv_grid_districts = gpd.GeoDataFrame({'geometry': [lvgd.geo_data
                                                           for la in grid.grid_district.lv_load_areas()
                                                           for lvgd in la.lv_grid_districts()]},
                                             crs={'init': 'epsg:{srid}'.format(srid=model_proj)})

        # reproject to WGS84 pseudo mercator
        mv_grid_district = mv_grid_district.to_crs(epsg=3857)
        load_areas_sat = load_areas_sat.to_crs(epsg=3857)
        load_areas_reg = load_areas_reg.to_crs(epsg=3857)
        load_areas_agg = load_areas_agg.to_crs(epsg=3857)
        lv_grid_districts = lv_grid_districts.to_crs(epsg=3857)

        # plot
        mv_grid_district.plot(ax=ax, **mvgd_style)
        load_areas_sat.plot(ax=ax, **la_style_sat)
        load_areas_reg.plot(ax=ax, **la_style_reg)
        load_areas_agg.plot(ax=ax, **la_style_agg)
        lv_grid_districts.plot(ax=ax, color='#ffffff', alpha=0.1, edgecolor='k', linewidth=0.5, zorder=4)

        # patches
        patches = [Patch(**mvgd_style, label='MV grid district')]
        if not load_areas_sat.empty:
            patches.append(Patch(**la_style_sat, label='Satellite load area'))
        if not load_areas_reg.empty:
            patches.append(Patch(**la_style_reg, label='Regular load area'))
        if not load_areas_agg.empty:
            patches.append(Patch(**la_style_agg, label='Aggregated load area'))

        return patches


    # plot legend including patches (regions)
    def plt_legend_without_duplicate_labels(ax, patches):
        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        if patches is not None:
            for patch in patches:
                unique.append((patch, patch.get_label()))
        ax.legend(*zip(*unique), fontsize=5)

    if not isinstance(grid, MVGridDing0):
        logger.warning('Sorry, but plotting is currently only available for MV grids but you did not pass an'
                       'instance of MVGridDing0. Plotting is skipped.')
        return

    g = grid.graph
    # model_proj = grid.network.config['geo']['srid'] # changed from 4326 to new CRS 3035 # update in config but be
    # aware, that it will effect import of genearors, load ares, etc
    model_proj = '3035'

    if testcase == 'feedin':
        case_idx = 1
    else:
        case_idx = 0

    nodes_types, nodes_by_type, node_colors_by_type, node_sizes_by_type, zindex_by_type, nodes_pos = \
        set_nodes_style_and_position(g.nodes())

    # reproject to WGS84 pseudo mercator
    nodes_pos = reproject_nodes(nodes_pos, model_proj=model_proj)

    plt.figure(figsize=(9, 6))
    ax = plt.gca()

    if line_color == 'loading':
        color = None
        edges_color = []
        edges_geom = []
        for n1, n2 in g.edges():
            edge = g.adj[n1][n2]
            if hasattr(edge['branch'], 's_res'):
                edges_color.append(edge['branch'].s_res[case_idx] * 1e3 /
                                   (3 ** 0.5 * edge['branch'].type['U_n'] * edge['branch'].type['I_max_th']))
                edges_geom.append(edge['branch'].geometry)
            else:
                edges_color.append(0)
        edges_cmap = plt.get_cmap('jet')
        # edges_cmap.set_over('#952eff')
    else:
        edges_color = ['black'] * len(list(grid.graph_edges()))
        color = 'black'
        edges_geom = [edge['branch'].geometry for edge in list(grid.graph_edges())]
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
            limits_cb_nodes = (math.floor(min(nodes_color) * 100) / 100,
                               math.ceil(max(nodes_color) * 100) / 100)
        v_range = np.linspace(limits_cb_nodes[0], limits_cb_nodes[1], 101)
        cb_voltage = plt.colorbar(nodes, boundaries=v_range,
                                  ticks=v_range[0:101:10],
                                  fraction=0.04, pad=0.1)
        cb_voltage.mappable.set_clim(vmin=limits_cb_nodes[0],
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

    # plotting based on geometry linestring

    # edges with geometry and color
    edges = gpd.GeoDataFrame({'geometry': [edge['branch'].geometry for edge in list(grid.graph_edges())],
                              'ring': [str(edge['branch'].ring) for edge in list(grid.graph_edges())]},
                             crs={'init': 'epsg:{srid}'.format(srid=3035)})

    edges = edges.to_crs(epsg=3857)

    if not edges.empty:

        edges['color'] = edges['ring'].str.extract('(\d+)').astype(int)

        edges.plot(column=edges['color'],
                   ax=ax,
                   cmap='jet',
                   linewidth=0.7,
                   alpha=1,
                   zorder=2)

    else:

        edges.plot(ax=ax,
                   color=edges_color,
                   linewidth=0.7,
                   # lw=1,
                   zorder=2)

    edges = nx.draw_networkx_edges(g,
                                   pos=nodes_pos,
                                   edge_color=edges_color,
                                   edge_cmap=edges_cmap,
                                   edge_vmin=0,
                                   edge_vmax=1,
                                   width=0,
                                   ax=ax)

    if line_color == 'loading':
        # colorbar edges
        if limits_cb_lines is None:
            limits_cb_lines = (math.floor(min(edges_color) * 100) / 100,
                               math.ceil(max(edges_color) * 100) / 100)
        loading_range = np.linspace(limits_cb_lines[0], limits_cb_lines[1], 101)
        cb_loading = plt.colorbar(edges, boundaries=loading_range,
                                  ticks=loading_range[0:101:10],
                                  fraction=0.04, pad=0.04)
        cb_loading.mappable.set_clim(vmin=limits_cb_lines[0],
                                     vmax=limits_cb_lines[1])
        cb_loading.set_label('Line loading in % of nominal capacity', size='smaller')
        cb_loading.ax.tick_params(labelsize='smaller')

    if use_ctx and background_map:
        plot_background_map(ax=ax)
    if use_gpd:
        patches = plot_region_data(ax=ax)

    if patches:
        plt_legend_without_duplicate_labels(ax, patches)
    else:
        plt_legend_without_duplicate_labels(ax, patches=None)

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
