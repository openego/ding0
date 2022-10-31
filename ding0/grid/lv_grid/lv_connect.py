"""This file is part of DING0, the DIstribution Network GeneratOr.
DING0 is a tool to generate synthetic medium and low voltage power
distribution grids based on open data.

It is developed in the project open_eGo: https://openegoproject.wordpress.com

DING0 lives at github: https://github.com/openego/ding0/
The documentation is available on RTD: http://ding0.readthedocs.io"""

__copyright__  = "Reiner Lemoine Institut gGmbH"
__license__    = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__url__        = "https://github.com/openego/ding0/blob/master/LICENSE"
__author__     = "nesnoj, gplssm"


from ding0.core.network import BranchDing0

from ding0.tools import config as cfg_ding0
from ding0.tools.geo import calc_geo_dist, calc_edge_geometry
from ding0.grid.tools import cable_type
import logging
import random

logger = logging.getLogger(__name__)


def lv_connect_generators(lv_grid_district, graph, debug=False):

    """ Connect LV generators to LV grid
    
    Args
    ----
    lv_grid_district: LVGridDistrictDing0
        LVGridDistrictDing0 object for which the connection process has to be done
    graph: :networkx:`NetworkX Graph Obj< >`
        NetworkX graph object with nodes
    debug: bool, defaults to False
        If True, information is printed during process

    Returns
    -------
    :networkx:`NetworkX Graph Obj< >`
        NetworkX graph object with nodes and newly created branches
    """

    cable_lf = cfg_ding0.get('assumptions',
                             'load_factor_lv_cable_fc_normal')
    cos_phi_gen = cfg_ding0.get('assumptions',
                                'cos_phi_gen')
    v_nom = cfg_ding0.get('assumptions', 'lv_nominal_voltage') / 1e3  # v_nom in kV
    seed = int(cfg_ding0.get('random', 'seed'))
    random.seed(a=seed)

    # generate random list (without replacement => unique elements)
    # of loads (residential) to connect genos (P <= 30kW) to.
    lv_loads_res = sorted(lv_grid_district.lv_grid.loads_sector(sector='res'),
                          key=lambda _: repr(_))
    if len(lv_loads_res) > 0:
        lv_loads_res_rnd = (random.sample(lv_loads_res,
                                             len(lv_loads_res)))
    else:
        lv_loads_res_rnd = None

    # generate random list (without replacement => unique elements)
    # of loads (retail, industrial, agricultural) to connect genos
    # (30kW <= P <= 100kW) to.
    lv_loads_ria = sorted(lv_grid_district.lv_grid.loads_sector(sector='ria'),
                          key=lambda _: repr(_))
    if len(lv_loads_ria) > 0:
        lv_loads_ria_rnd = (random.sample(lv_loads_ria,
                                             len(lv_loads_ria)))
    else:
        lv_loads_ria_rnd = None

    for generator in sorted(lv_grid_district.lv_grid.generators(), key=lambda x: repr(x)):

        # generator is of v_level 6 -> connect to LV station
        if generator.v_level == 6:
            lv_station = lv_grid_district.lv_grid.station()

            branch_shp, branch_length = calc_edge_geometry(generator, lv_station)
            branch_length = calc_geo_dist(generator, lv_station)
            branch_type = cable_type(
                generator.capacity / (cable_lf * cos_phi_gen),
                v_nom,
                lv_grid_district.lv_grid.network.static_data['LV_cables'])

            branch = BranchDing0(length=branch_length,
                                 kind='cable',
                                 grid=lv_grid_district.lv_grid,
                                 type=branch_type,
                                 geometry=branch_shp)

            graph.add_edge(generator, lv_station, branch=branch)

        # generator is of v_level 7 -> assign geno to load
        elif generator.v_level == 7:

            # connect genos with P <= 30kW to residential loads, if available
            if (generator.capacity <= 30) and (lv_loads_res_rnd is not None):
                if len(lv_loads_res_rnd) > 0:
                    lv_load = lv_loads_res_rnd.pop()
                # if random load list is empty, create new one
                else:
                    lv_loads_res_rnd = (random.sample(lv_loads_res,
                                                     len(lv_loads_res))
                                       )
                    lv_load = lv_loads_res_rnd.pop()

                # get cable distributor of building
                lv_conn_target = list(graph.neighbors(lv_load))[0]

            # connect genos with 30kW <= P <= 100kW to residential loads
            # to retail, industrial, agricultural loads, if available
            elif (generator.capacity > 30) and (lv_loads_ria_rnd is not None):
                if len(lv_loads_ria_rnd) > 0:
                    lv_load = lv_loads_ria_rnd.pop()
                # if random load list is empty, create new one
                else:
                    lv_loads_ria_rnd = (random.sample(lv_loads_ria,
                                                         len(lv_loads_ria))
                                           )
                    lv_load = lv_loads_ria_rnd.pop()

                # get cable distributor of building
                lv_conn_target = list(graph.neighbors(lv_load))[0]

            # fallback: connect to station
            else:
                lv_conn_target = lv_grid_district.lv_grid.station()

                logger.warning(
                    'No valid conn. target found for {}.'
                    'Connected to {}.'.format(
                        repr(generator),
                        repr(lv_conn_target)
                    ))

            # determine appropriate type of cable
            branch_type = cable_type(
                generator.capacity / (cable_lf * cos_phi_gen),
                v_nom,
                lv_grid_district.lv_grid.network.static_data['LV_cables'])

            # connect to cable dist. of building
            branch_shp, branch_length = calc_edge_geometry(generator, lv_conn_target)
            branch = BranchDing0(length=1,
                                 kind='cable',
                                 grid=lv_grid_district.lv_grid,
                                 type=branch_type,
                                 geometry=branch_shp)

            graph.add_edge(generator, lv_conn_target, branch=branch)

    return graph
