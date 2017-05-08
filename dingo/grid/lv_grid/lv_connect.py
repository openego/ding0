from dingo.core.network import BranchDingo

from dingo.tools import config as cfg_dingo
from dingo.tools.geo import calc_geo_dist_vincenty
import logging
import random

logger = logging.getLogger('dingo')


def lv_connect_generators(lv_grid_district, graph, debug=False):
    """ Connect LV generators to LV grid

    Args:
        lv_grid_district: LVGridDistrictDingo object for which the connection process has to be done
        graph: NetworkX graph object with nodes
        debug: If True, information is printed during process

    Returns:
        graph: NetworkX graph object with nodes and newly created branches
    """

    # get predefined random seed and initialize random generator
    seed = int(cfg_dingo.get('random', 'seed'))
    random.seed(a=seed)

    # generate random list (without replacement => unique elements)
    # of loads (residential) to connect genos (P <= 30kW) to.
    lv_loads_res = sorted(lv_grid_district.lv_grid.loads_sector(sector='res'),
                          key=lambda _: repr(_))
    if len(lv_loads_res) > 0:
        lv_loads_res_rnd = set(random.sample(lv_loads_res,
                                             len(lv_loads_res)))
    else:
        lv_loads_res_rnd = None

    # generate random list (without replacement => unique elements)
    # of loads (retail, industrial, agricultural) to connect genos
    # (30kW <= P <= 100kW) to.
    lv_loads_ria = sorted(lv_grid_district.lv_grid.loads_sector(sector='ria'),
                          key=lambda _: repr(_))
    if len(lv_loads_ria) > 0:
        lv_loads_ria_rnd = set(random.sample(lv_loads_ria,
                                             len(lv_loads_ria)))
    else:
        lv_loads_ria_rnd = None

    for generator in sorted(lv_grid_district.lv_grid.generators(), key=lambda x: repr(x)):

        # generator is of v_level 6 -> connect to LV station
        if generator.v_level == 6:
            lv_station = lv_grid_district.lv_grid.station()

            branch_length = calc_geo_dist_vincenty(generator, lv_station)

            # TODO: Set type of cable
            branch = BranchDingo(length=branch_length,
                                 kind='cable',
                                 type=None)

            graph.add_edge(generator, lv_station, branch=branch)

        # generator is of v_level 7 -> assign geno to load
        elif generator.v_level == 7:

            # connect genos with P <= 30kW to residential loads, if available
            if (generator.capacity <= 30) and (lv_loads_res_rnd is not None):
                if len(lv_loads_res_rnd) > 0:
                    lv_load = lv_loads_res_rnd.pop()
                # if random load list is empty, create new one
                else:
                    lv_loads_res_rnd = set(random.sample(lv_loads_res,
                                                     len(lv_loads_res))
                                       )
                    lv_load = lv_loads_res_rnd.pop()

                # get cable distributor of building
                lv_conn_target = graph.neighbors(lv_load)[0]

            # connect genos with 30kW <= P <= 100kW to residential loads
            # to retail, industrial, agricultural loads, if available
            elif (generator.capacity > 30) and (lv_loads_ria_rnd is not None):
                if len(lv_loads_ria_rnd) > 0:
                    lv_load = lv_loads_ria_rnd.pop()
                # if random load list is empty, create new one
                else:
                    lv_loads_ria_rnd = set(random.sample(lv_loads_ria,
                                                         len(lv_loads_ria))
                                           )
                    lv_load = lv_loads_ria_rnd.pop()

                # get cable distributor of building
                lv_conn_target = graph.neighbors(lv_load)[0]

            # fallback: connect to station
            else:
                lv_conn_target = lv_grid_district.lv_grid.station()

                logger.warning(
                    'No valid conn. target found for {}.'
                    'Connected to {}.'.format(
                        repr(generator),
                        repr(lv_conn_target)
                    ))
                    

            # connect to cable dist. of building
            # TODO: Set type of cable
            branch = BranchDingo(length=1,
                                 kind='cable',
                                 type=None)

            graph.add_edge(generator, lv_conn_target, branch=branch)

    return graph