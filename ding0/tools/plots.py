import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import networkx as nx
from pyproj import Proj, transform, Transformer
import logging

logger = logging.getLogger(__name__)
from ding0.tools.logger import get_default_home_dir
from ding0.core.network.grids import MVGridDing0
from ding0.config.config_lv_grids_osm import get_config_osm

from ding0.tools.debug import log_errors

use_gpd = True
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


@log_errors
def plot_mv_topology(grid, path = None, subtitle='', filename=None, testcase='load',
                     line_color='ring', node_color='type',
                     limits_cb_lines=None, limits_cb_nodes=None,
                     background_map=True):
    """ Draws MV grid graph using networkx

    Parameters
    ----------
    grid : :obj:`MVGridDing0`
        MV grid to plot.
    path : :obj:`str`
        Path to save the plot.
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

        * 'ring'
          Line color is set according to related ring of the line.
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
        If True, a background map is plotted (default: carto b positron).
        The additional package `contextily` is needed for this functionality.
        Default: True

    Note
    -----
    WGS84 pseudo mercator (epsg:3857) is used as coordinate reference system (CRS).
    Therefore, the drawn graph representation may be falsified!

    """

    model_proj = get_config_osm('srid')

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

    def reproject_nodes(nodes_pos, model_proj=str(model_proj)):  # changed from 4326 to new CRS 3035
        transformer = Transformer.from_crs(int(model_proj), 3857, always_xy=True)
        nodes_pos2 = {}
        for k, v in nodes_pos.items():
            nodes_pos2[k] = (transformer.transform(v[0], v[1]))
        return nodes_pos2

    def plot_background_map(ax):
        url = ctx.providers.CartoDB.Positron
        xmin, xmax, ymin, ymax = ax.axis()
        basemap, extent = ctx.bounds2img(xmin, ymin, xmax, ymax,
                                         source=url)
        ax.imshow(basemap, extent=extent, interpolation='bilinear', zorder=0)
        ax.axis((xmin, xmax, ymin, ymax))

    # regions style by type
    mvgd_style = {'facecolor': '#ffffff', 'alpha': 1, 'edgecolor': '#fdcc02', 'linewidth': 1, 'zorder': 1}

    if not plot_background_map:
        alpha = 1
    else:
        alpha = 0.4

    la_style_sat = {'facecolor': '#ffffde', 'alpha': alpha, 'edgecolor': 'k', 'linewidth': 0.2, 'zorder': 1}
    la_style_reg = {'facecolor': '#e1c938', 'alpha': alpha, 'edgecolor': 'k', 'linewidth': 0.2, 'zorder': 1}
    la_style_agg = {'facecolor': '#c3c3c3', 'alpha': alpha, 'edgecolor': 'k', 'linewidth': 0.2, 'zorder': 1}

    def plot_region_data(ax):
        # get geoms of MV grid district, load areas and LV grid districts
        mv_grid_district = gpd.GeoDataFrame({'geometry': grid.grid_district.geo_data},
                                            crs='epsg:{srid}'.format(srid=model_proj))
        load_areas_sat = gpd.GeoDataFrame({'geometry': [la.geo_area for la in grid.grid_district.lv_load_areas()
                                                        if la.is_satellite]},
                                      crs='epsg:{srid}'.format(srid=model_proj))
        load_areas_reg = gpd.GeoDataFrame({'geometry': [la.geo_area for la in grid.grid_district.lv_load_areas()
                                                        if not la.is_satellite and not la.is_aggregated]},
                                          crs='epsg:{srid}'.format(srid=model_proj))
        load_areas_agg = gpd.GeoDataFrame({'geometry': [la.geo_area for la in grid.grid_district.lv_load_areas()
                                                        if la.is_aggregated]},
                                          crs='epsg:{srid}'.format(srid=model_proj))
        lv_grid_districts = gpd.GeoDataFrame({'geometry': [lvgd.geo_data
                                                           for la in grid.grid_district.lv_load_areas()
                                                           for lvgd in la.lv_grid_districts()]},
                                             crs='epsg:{srid}'.format(srid=model_proj))

        # reproject to WGS84 pseudo mercator
        mv_grid_district = mv_grid_district.to_crs(epsg=3857)
        load_areas_sat = load_areas_sat.to_crs(epsg=3857)
        load_areas_reg = load_areas_reg.to_crs(epsg=3857)
        load_areas_agg = load_areas_agg.to_crs(epsg=3857)
        lv_grid_districts = lv_grid_districts.to_crs(epsg=3857)

        # plot with
        # patches
        patches = []

        if not load_areas_sat.empty:
            patches.append(Patch(**la_style_sat, label='Satellite load area'))
            load_areas_sat.plot(ax=ax, **la_style_sat)
        if not load_areas_reg.empty:
            patches.append(Patch(**la_style_reg, label='Regular load area'))
            load_areas_reg.plot(ax=ax, **la_style_reg)
        if not load_areas_agg.empty:
            patches.append(Patch(**la_style_agg, label='Aggregated load area'))
            load_areas_agg.plot(ax=ax, **la_style_agg)

        patches.append(Patch(**mvgd_style, label='MV grid district'))
        mv_grid_district.boundary.plot(ax=ax, edgecolor='#fdcc02', linewidth=1, zorder=1)

        if not plot_background_map:
            lv_grid_districts.plot(ax=ax, color='#ffffff', alpha=0.1, edgecolor='k', linewidth=0.1, zorder=2)

        return patches


    # plot legend including patches (regions)
    def plt_legend_without_duplicate_labels(ax, patches):
        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        if patches is not None:
            for patch in patches:
                unique.insert(0, (patch, patch.get_label()))
        ax.legend(*zip(*unique), fontsize=5)

    if not isinstance(grid, MVGridDing0):
        logger.warning('Sorry, but plotting is currently only available for MV grids but you did not pass an'
                       'instance of MVGridDing0. Plotting is skipped.')
        return

    g = grid.graph

    if testcase == 'feedin':
        case_idx = 1
    else:
        case_idx = 0

    nodes_types, nodes_by_type, node_colors_by_type, node_sizes_by_type, zindex_by_type, nodes_pos = \
        set_nodes_style_and_position(g.nodes())

    # reproject to WGS84 pseudo mercator
    nodes_pos = reproject_nodes(nodes_pos, model_proj=str(model_proj))

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

    else:
        color = 'black'
        edges_color = [color] * len(list(grid.graph_edges()))
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
                             crs='epsg:{srid}'.format(srid=model_proj))

    edges = edges.to_crs(epsg=3857)

    if not edges.empty:

        if line_color == 'loading': # plots line loading

            edges = gpd.GeoDataFrame({'geometry': edges_geom,
                                      'color': edges_color},
                                     crs='epsg:{srid}'.format(srid=model_proj))
            edges = edges.to_crs(epsg=3857)

            edges.plot(ax=ax,
                       cmap=edges_cmap,
                       linewidth=0.7,
                       alpha=1,
                       zorder=2)

        elif line_color == 'ring': # plots different color for each ring

            edges = gpd.GeoDataFrame({'geometry': [edge['branch'].geometry for edge in list(grid.graph_edges())],
                                      'ring': [str(edge['branch'].ring) for edge in list(grid.graph_edges())]},
                                     crs='epsg:{srid}'.format(srid=model_proj))
            edges = edges.to_crs(epsg=3857)

            try:
                edges['color'] = edges['ring'].str.extract('(\d+)').astype(int)
            except ValueError:
                edges['color'] = edges['ring'].str.extract('(\d+)').fillna(0).astype(int)
                logger.error("ValueError: Branch has no ring assigned. Fill nan with 0")

            edges.plot(column=edges['color'],
                       ax=ax,
                       cmap='jet',
                       linewidth=0.7,
                       alpha=1,
                       zorder=2)

        else: # plots only black lines

            edges = gpd.GeoDataFrame({'geometry': [edge['branch'].geometry for edge in list(grid.graph_edges())]},
                                     crs='epsg:{srid}'.format(srid=model_proj))
            edges = edges.to_crs(epsg=3857)

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
        if path:
            path = os.path.join(path, str(grid.id_db), "plot_mv")
            if not os.path.exists(path):
                os.makedirs(path)
        else:
            path = get_default_home_dir()
        path = os.path.join(path, f'ding0_grid_{str(grid.id_db)}_{filename}.pdf')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info('==> Figure saved to {path}'.format(path=path))


@log_errors
def plot_lv_topology(grid, path=None, mv_grid_id=None, subtitle="", testcase='load', line_color='feeder', node_color='type',
                     background_map=True, filename=None):
    """ Draws LV grid graph using networkx

    Parameters
    ----------
    grid : :obj:`LVGridDing0`
        LV grid to plot.
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

        * 'feeder'
          Line color is set according to related feeder of the line.
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
        If True, a background map is plotted (default: carto b positron).
        The additional package `contextily` is needed for this functionality.
        Default: True

    Note
    -----
    WGS84 pseudo mercator (epsg:3857) is used as coordinate reference system (CRS).
    Therefore, the drawn graph representation may be falsified!

    """

    model_proj = get_config_osm('srid')

    def plt_legend_without_duplicate_labels(ax, patches):

        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        if patches is not None:
            for patch in patches:
                unique.insert(0, (patch, patch.get_label()))
        ax.legend(*zip(*unique), fontsize=7)

    def set_nodes_style_and_position(nodes):

        # TODO: MOVE settings to config
        # node types (name of classes)
        node_types = ['LVStationDing0',
                      'LVLoadDing0',  # PAUL new
                      'LVCableDistributorDing0',
                      'GeneratorFluctuatingDing0',
                      'GeneratorDing0',
                      'n/a']

        # node styles
        colors_dict = {'LVStationDing0': '#f2ae00',
                       'LVLoadDing0': '#d62728',  # PAUL new
                       'LVCableDistributorDing0': '#ffa500',  # PAUL new'#000000',
                       'GeneratorDing0': '#097969',
                       'n/a': 'orange'}
        sizes_dict = {'LVStationDing0': 50,
                      'LVLoadDing0': 6,  # PAUL new
                      'GeneratorDing0': 12,
                      'LVCableDistributorDing0': 4,
                      'n/a': 3}
        zindex_by_type = {'LVStationDing0': 12,
                          'LVLoadDing0': 12,  # PAUL new
                          'GeneratorDing0': 11,
                          'LVCableDistributorDing0': 9,
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

                type_n = type(n).__name__
                if type_n == 'GeneratorFluctuatingDing0':
                    type_n = 'GeneratorDing0'

                nodes_by_type[type_n].append(n)
                node_colors_by_type[type_n].append(colors_dict[type_n])
                node_sizes_by_type[type_n].append(sizes_dict[type_n])
                node_sizes_by_type['all'].append(sizes_dict[type_n])
            else:
                nodes_by_type['n/a'].append(n)
                node_colors_by_type['n/a'].append(colors_dict['n/a'])
                node_sizes_by_type['n/a'].append(sizes_dict['n/a'])
                node_sizes_by_type['all'].append(sizes_dict['n/a'])
            nodes_pos[n] = (n.geo_data.x, n.geo_data.y)

        return node_types, nodes_by_type, node_colors_by_type, \
               node_sizes_by_type, zindex_by_type, nodes_pos

    def reproject_nodes(nodes_pos, model_proj=str(model_proj)):  # changed from 4326 to new CRS 3035
        # PAUL new: replace transform with Transformer in pyproj, to keep syntax up-to-date
        transformer = Transformer.from_crs(int(model_proj), 3857, always_xy=True)
        nodes_pos2 = {}
        for k, v in nodes_pos.items():
            nodes_pos2[k] = (transformer.transform(v[0], v[1]))
        return nodes_pos2

    def plot_background_map(ax):
        url = ctx.providers.CartoDB.Positron
        xmin, xmax, ymin, ymax = ax.axis()
        basemap, extent = ctx.bounds2img(xmin, ymin, xmax, ymax,
                                         source=url)
        ax.imshow(basemap, extent=extent, interpolation='bilinear', zorder=0)
        ax.axis((xmin, xmax, ymin, ymax))

    lvgd_style = {'facecolor': '#ffffff', 'alpha': 1, 'edgecolor': '#fdcc02', 'linewidth': 1, 'zorder': 1}
    buildings_style = {'alpha': 0.5, 'facecolor': '#1f77b4', 'edgecolor': 'darkblue', 'linewidth': 0.5, 'zorder': 2}

    def plot_region_data(ax):
        # get geoms of MV grid district, load areas and LV grid districts
        buildings = gpd.GeoDataFrame({'geometry': grid.grid_district.buildings.geometry.tolist()},
                                     crs='epsg:{srid}'.format(srid=model_proj))
        lv_grid_district = gpd.GeoDataFrame({'geometry': [grid.grid_district.geo_data]},
                                            crs='epsg:{srid}'.format(srid=model_proj))

        # reproject to WGS84 pseudo mercator
        buildings = buildings.to_crs(epsg=3857)
        lv_grid_district = lv_grid_district.to_crs(epsg=3857)

        # plot with
        # patches
        lv_grid_district.boundary.plot(ax=ax, edgecolor='#fdcc02', linewidth=1, zorder=1)
        patches = []

        if not buildings.empty:
            patches.append(Patch(**buildings_style, label='Building geometries'))
            buildings.plot(ax=ax, **buildings_style)

        patches.append(Patch(**lvgd_style, label='LV grid district'))

        return patches

    g = grid._graph


    if testcase == 'feedin':
        case_idx = 1
    else:
        case_idx = 0

    nodes_types, nodes_by_type, node_colors_by_type, node_sizes_by_type, zindex_by_type, nodes_pos = \
        set_nodes_style_and_position(g.nodes())

    # reproject to WGS84 pseudo mercator
    nodes_pos = reproject_nodes(nodes_pos, model_proj=str(model_proj))

    plt.figure(figsize=(9, 6))
    ax = plt.gca()

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

    # edges with geometry and color
    edges = gpd.GeoDataFrame({'geometry': [edge['branch'].geometry for edge in grid.graph_edges()],
                              'feeder': [str(edge['branch'].feeder) for edge in grid.graph_edges()]},
                             crs='epsg:{srid}'.format(srid=model_proj))

    edges = edges.to_crs(epsg=3857)

    if not edges.empty:

        if line_color == 'feeder': # plots differnt color for feeder
            try:
                edges['color'] = edges['feeder'].str.extract('(\d+)').astype(int)
            except ValueError:
                edges['color'] = edges['feeder'].str.extract('(\d+)').fillna(0).astype(int)
            edges = edges.sort_values(by='color')
            color = None
            edges_color = edges['color']
            edges_cmap = 'jet'

        elif line_color == 'loading': # plots line loading

            color = None
            edges_color = []
            for edge in grid.graph_edges():
                # plotting based on line loading or node voltages not possible
                # TODO: s_res value from powerflow (pypsa.io) are missing
                # TODO: node voltages are not available as well
                if hasattr(edge['branch'], 's_res'):
                    edges_color.append(edge['branch'].s_res[case_idx] * 1e3 /
                                       (3 ** 0.5 * edge['branch'].type['U_n'] * edge['branch'].type['I_max_th']))
                else:
                    edges_color.append(0)
            edges_cmap = 'jet'

        else: # plots black lines

            color = 'black'
            edges_color = None
            edges_cmap = None

        edges.plot(column=edges_color,
                   ax=ax,
                   color=color,
                   cmap=edges_cmap,
                   linewidth=1,
                   alpha=1,
                   zorder=3)

    if use_gpd:
        patches = plot_region_data(ax=ax)

    if use_ctx and background_map:
        plot_background_map(ax=ax)

    if patches:
        plt_legend_without_duplicate_labels(ax, patches)
    else:
        plt_legend_without_duplicate_labels(ax, patches=None)

    plt.title('LV Grid District {id} - {st}'.format(id=grid.grid_district.id_db,
                                                    st=subtitle))

    # hide axes labels (coords)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    if filename is None:
        plt.tight_layout()
        plt.show()
    else:
        if path:
            path = os.path.join(path, str(mv_grid_id), "plot_lv")
            if not os.path.exists(path):
                os.makedirs(path)
        else:
            path = get_default_home_dir()
        path = os.path.join(path, f'ding0_lv_grid_{str(grid.id_db)}_{filename}.pdf')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info('==> Figure saved to {path}'.format(path=path))
