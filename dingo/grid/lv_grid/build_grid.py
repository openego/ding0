"""This file is part of DINGO, the DIstribution Network GeneratOr.
DINGO is a tool to generate synthetic medium and low voltage power
distribution grids based on open data.

It is developed in the project open_eGo: https://openegoproject.wordpress.com

DINGO lives at github: https://github.com/openego/dingo/
The documentation is available on RTD: http://dingo.readthedocs.io"""

__copyright__  = "Reiner Lemoine Institut gGmbH"
__license__    = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__url__        = "https://github.com/openego/dingo/blob/master/LICENSE"
__author__     = "nesnoj, gplssm"


from dingo.tools import config as cfg_dingo

from dingo.core.network import TransformerDingo, BranchDingo
from dingo.core.network.cable_distributors import LVCableDistributorDingo
from dingo.core.network.loads import LVLoadDingo
import logging
import math

logger = logging.getLogger('dingo')


def select_transformers(grid, s_max=None):
    """ Selects MV-LV transformers for the MV-LV substation.

    The transformers are chosen according to max. of load case and feedin-case
    considering load factors and power factor.
    The MV-LV transformer with the next higher available nominal apparent power is
    chosen. Therefore, a max. allowed transformer loading of 100% is implicitly
    assumed. If the peak load exceeds the max. power of a single available
    transformer, multiple transformer are build.

    By default `peak_load` and `peak_generation` are taken from `grid` instance.
    The behavior can be overridden providing `s_max` as explained in
    ``Arguments``.

    Parameters
    ----------
    grid: dingo.core.network.LVGridDingo
        LV grid data

    Arguments
    ---------
    s_max : dict
        dict containing maximum apparent power of load or generation case and
        str describing the case. For example

        .. code-block:: python

            {
                's_max': 480,
                'case': 'load'
            }

        or

        .. code-block:: python

            {
                's_max': 120,
                'case': 'gen'
            }

        s_max passed overrides `grid.grid_district.peak_load` respectively
        `grid.station().peak_generation`.

    Returns
    -------
    transformer: DataFrame
        Parameters of chosen Transformer
    transformer_cnt: int
        Count of transformers
    """

    load_factor_lv_trans_lc_normal = cfg_dingo.get('assumptions',
                                                   'load_factor_lv_trans_lc_normal')
    load_factor_lv_trans_fc_normal = cfg_dingo.get('assumptions',
                                                   'load_factor_lv_trans_fc_normal')

    cos_phi_load = cfg_dingo.get('assumptions',
                                 'lv_cos_phi_load')
    cos_phi_gen = cfg_dingo.get('assumptions',
                                'lv_cos_phi_gen')

    # get equipment parameters of LV transformers
    trafo_parameters = grid.network.static_data['LV_trafos']

    # determine s_max from grid object if not provided via arguments
    if s_max is None:
        # get maximum from peak load and peak generation
        s_max_load = grid.grid_district.peak_load / cos_phi_load
        s_max_gen = grid.station().peak_generation / cos_phi_gen

        # check if load or generation is greater respecting corresponding load factor
        if s_max_load > s_max_gen:
            # use peak load and load factor from load case
            load_factor_lv_trans = load_factor_lv_trans_lc_normal
            s_max = s_max_load
        else:
            # use peak generation and load factor for feedin case
            load_factor_lv_trans = load_factor_lv_trans_fc_normal
            s_max = s_max_gen
    else:
        if s_max['case'] == 'load':
            load_factor_lv_trans = load_factor_lv_trans_lc_normal
        elif s_max['case'] == 'gen':
            load_factor_lv_trans = load_factor_lv_trans_fc_normal
        else:
            logger.error('No proper \'case\' provided for argument s_max')
            raise ValueError('Please provide proper \'case\' for argument '
                             '`s_max`.')
        s_max = s_max['s_max']

    # get max. trafo
    transformer_max = trafo_parameters.iloc[trafo_parameters['S_max'].idxmax()]

    # peak load is smaller than max. available trafo
    if s_max < (transformer_max['S_max'] * load_factor_lv_trans ):
        # choose trafo
        transformer = trafo_parameters.iloc[
            trafo_parameters[
                trafo_parameters['S_max'] * load_factor_lv_trans > s_max][
                'S_max'].idxmin()]
        transformer_cnt = 1
    # peak load is greater than max. available trafo -> use multiple trafos
    else:
        transformer_cnt = 2
        # increase no. of trafos until peak load can be supplied
        while not any(trafo_parameters['S_max'] * load_factor_lv_trans > (
                    s_max / transformer_cnt)):
            transformer_cnt += 1
        transformer = trafo_parameters.iloc[
            trafo_parameters[
                trafo_parameters['S_max'] * load_factor_lv_trans
                > (s_max / transformer_cnt)]['S_max'].idxmin()]

    return transformer, transformer_cnt


def transformer(grid):
    """
    Choose transformer and add to grid's station

    Parameters
    ----------
    grid: dingo.core.network.LVGridDingo
        LV grid data
    """

    # choose size and amount of transformers
    transformer, transformer_cnt = select_transformers(grid)

    # create transformers and add them to station of LVGD
    for t in range(0, transformer_cnt):
        lv_transformer = TransformerDingo(
            grid=grid,
            id_db=id,
            v_level=0.4,
            s_max_longterm=transformer['S_max'],
            r=transformer['R'],
            x=transformer['X'])

        # add each transformer to its station
        grid._station.add_transformer(lv_transformer)


def select_grid_model_ria(lvgd, sector):
    """
    Select a typified grid for retail/industrial and agricultural

    Parameters
    ----------
    lvgd : dingo.core.structure.regions.LVGridDistrictDingo
        Low-voltage grid district object
    sector : str
        Either 'retail/industrial' or 'agricultural'. Depending on choice
        different parameters to grid topology apply

    Returns
    -------
    grid_model : dict
        Parameters that describe branch lines of a sector
    """

    cable_lf = cfg_dingo.get('assumptions',
                             'load_factor_lv_cable_lc_normal')

    cos_phi_load = cfg_dingo.get('assumptions',
                                 'lv_cos_phi_load')

    max_lv_branch_line_load = cfg_dingo.get('assumptions',
                                            'max_lv_branch_line')

    # make a distinction between sectors
    if sector == 'retail/industrial':
        max_branch_length = cfg_dingo.get(
            "assumptions",
            "branch_line_length_retail_industrial")
        peak_load = lvgd.peak_load_retail + \
                    lvgd.peak_load_industrial
        count_sector_areas = lvgd.sector_count_retail + \
                             lvgd.sector_count_industrial
    elif sector == 'agricultural':
        max_branch_length = cfg_dingo.get(
            "assumptions",
            "branch_line_length_agricultural")
        peak_load = lvgd.peak_load_agricultural
        count_sector_areas = lvgd.sector_count_agricultural
    else:
        raise ValueError('Sector {} does not exist!'.format(sector))

    # determine size of a single load
    single_peak_load = peak_load / count_sector_areas

    # if this single load exceeds threshold of 300 kVA it is splitted
    while single_peak_load > (max_lv_branch_line_load * (cable_lf * cos_phi_load)):
        single_peak_load = single_peak_load / 2
        count_sector_areas = count_sector_areas * 2

    grid_model = {}

    # determine parameters of branches and loads connected to the branch
    # line
    if 0 < single_peak_load:
        grid_model['max_loads_per_branch'] = math.floor(
            (max_lv_branch_line_load * (cable_lf * cos_phi_load)) / single_peak_load)
        grid_model['single_peak_load'] = single_peak_load
        grid_model['full_branches'] = math.floor(
            count_sector_areas / grid_model['max_loads_per_branch'])
        grid_model['remaining_loads'] = count_sector_areas - (
            grid_model['full_branches'] * grid_model['max_loads_per_branch']
        )
        grid_model['load_distance'] = max_branch_length / (
            grid_model['max_loads_per_branch'] + 1)
        grid_model['load_distance_remaining'] = max_branch_length / (
            grid_model['remaining_loads'] + 1)
    else:
        if count_sector_areas > 0:
            logger.warning(
                'LVGD {lvgd} has in sector {sector} no load but area count'
                'is {count}. This is maybe related to #153'.format(
                    lvgd=lvgd,
                    sector=sector,
                    count=count_sector_areas))
            grid_model = None

    return grid_model


def grid_model_params_ria(lvgd):
    """
    Determine grid model parameters for LV grids of sectors
    retail/industrial and agricultural
    
    lvgd : dingo.core.structure.regions.LVGridDistrictDingo
        Low-voltage grid district object

    Returns
    -------
    model_params_ria : dict
        Structural description of (parts of) LV grid topology
    """

    # Choose retail/industrial and agricultural grid model
    model_params_ria = {}
    if ((lvgd.sector_count_retail +
             lvgd.sector_count_industrial > 0) or
            (lvgd.peak_load_retail +
                 lvgd.peak_load_industrial > 0)):
        model_params_ria['retail/industrial'] = select_grid_model_ria(
            lvgd, 'retail/industrial')
    else:
        model_params_ria['retail/industrial'] = None

    if ((lvgd.sector_count_agricultural > 0) or
            (lvgd.peak_load_agricultural > 0)):
        model_params_ria['agricultural'] = select_grid_model_ria(lvgd,
                                                                 'agricultural')
    else:
        model_params_ria['agricultural'] = None

    return model_params_ria


def build_lv_graph_ria(lvgd, grid_model_params):
    """
    Build graph for LV grid of sectors retail/industrial and agricultural

    Based on structural description of LV grid topology for sectors
    retail/industrial and agricultural (RIA) branches for these sectors are
    created and attached to the LV grid's MV-LV substation bus bar.

    LV loads of the sectors retail/industrial and agricultural are located
    in separat branches for each sector (in case of large load multiple of
    these).
    These loads are distributed across the branches by an equidistant
    distribution.

    This function accepts the dict `grid_model_params` with particular
    structure

    >>> grid_model_params = {
    >>> ... 'agricultural': {
    >>> ...     'max_loads_per_branch': 2
    >>> ...     'single_peak_load': 140,
    >>> ...     'full_branches': 2,
    >>> ...     'remaining_loads': 1,
    >>> ...     'load_distance': 800/3,
    >>> ...     'load_distance_remaining': 400}}

    Parameters
    ----------
    lvgd : dingo.core.structure.regions.LVGridDistrictDingo
        Low-voltage grid district object
    grid_model_params : dict
        Dict of structural information of sectoral LV grid branch

    Notes
    -----
    We assume a distance from the load to the branch it is connected to of
    30 m. This assumption is defined in the config files
    """

    def lv_graph_attach_branch():
        """
        Attach a single branch including its equipment (cable dist, loads
        and line segments) to graph of `lv_grid`
        """

        # determine maximum current occuring due to peak load
        # of this load load_no
        I_max_load = val['single_peak_load'] / (3 ** 0.5 * 0.4) / cos_phi_load

        # determine suitable cable for this current
        suitable_cables_stub = lvgd.lv_grid.network.static_data['LV_cables'][
            (lvgd.lv_grid.network.static_data['LV_cables'][
                'I_max_th'] * cable_lf) > I_max_load]
        cable_type_stub = suitable_cables_stub.ix[
            suitable_cables_stub['I_max_th'].idxmin()]

        # cable distributor to divert from main branch
        lv_cable_dist = LVCableDistributorDingo(
            grid=lvgd.lv_grid,
            branch_no=branch_no,
            load_no=load_no)
        # add lv_cable_dist to graph
        lvgd.lv_grid.add_cable_dist(lv_cable_dist)

        # cable distributor within building (to connect load+geno)
        lv_cable_dist_building = LVCableDistributorDingo(
            grid=lvgd.lv_grid,
            branch_no=branch_no,
            load_no=load_no,
            in_building=True)
        # add lv_cable_dist_building to graph
        lvgd.lv_grid.add_cable_dist(lv_cable_dist_building)

        # create an instance of Dingo LV load
        lv_load = LVLoadDingo(grid=lvgd.lv_grid,
                              branch_no=branch_no,
                              load_no=load_no,
                              peak_load=val['single_peak_load'])

        # add lv_load to graph
        lvgd.lv_grid.add_load(lv_load)

        # create branch line segment between either (a) station
        # and cable distributor or (b) between neighboring cable
        # distributors
        if load_no == 1:
            # case a: cable dist <-> station
            lvgd.lv_grid._graph.add_edge(
                lvgd.lv_grid.station(),
                lv_cable_dist,
                branch=BranchDingo(
                    length=val['load_distance'],
                    kind='cable',
                    type=cable_type,
                    id_db='branch_{sector}{branch}_{load}'.format(
                        branch=branch_no,
                        load=load_no,
                        sector=sector_short)
                ))
        else:
            # case b: cable dist <-> cable dist
            lvgd.lv_grid._graph.add_edge(
                lvgd.lv_grid._cable_distributors[-4],
                lv_cable_dist,
                branch=BranchDingo(
                    length=val['load_distance'],
                    kind='cable',
                    type=cable_type,
                    id_db='branch_{sector}{branch}_{load}'.format(
                        branch=branch_no,
                        load=load_no,
                        sector=sector_short)))

        # create branch stub that connects the load to the
        # lv_cable_dist located in the branch line
        lvgd.lv_grid._graph.add_edge(
            lv_cable_dist,
            lv_cable_dist_building,
            branch=BranchDingo(
                length=cfg_dingo.get(
                    'assumptions',
                    'lv_ria_branch_connection_distance'),
                kind='cable',
                type=cable_type_stub,
                id_db='stub_{sector}{branch}_{load}'.format(
                    branch=branch_no,
                    load=load_no,
                    sector=sector_short))
        )

        lvgd.lv_grid._graph.add_edge(
            lv_cable_dist_building,
            lv_load,
            branch=BranchDingo(
                length=1,
                kind='cable',
                type=cable_type_stub,
                id_db='stub_{sector}{branch}_{load}'.format(
                    branch=branch_no,
                    load=load_no,
                    sector=sector_short))
        )

    cable_lf = cfg_dingo.get('assumptions',
                             'load_factor_lv_cable_lc_normal')
    cos_phi_load = cfg_dingo.get('assumptions',
                                'lv_cos_phi_load')

    # iterate over branches for sectors retail/industrial and agricultural
    for sector, val in grid_model_params.items():
        if sector == 'retail/industrial':
            sector_short = 'RETIND'
        elif sector == 'agricultural':
            sector_short = 'AGR'
        else:
            sector_short = ''
        if val is not None:
            for branch_no in list(range(1, val['full_branches'] + 1)):

                # determine maximum current occuring due to peak load of branch
                I_max_branch = (val['max_loads_per_branch'] *
                                val['single_peak_load']) / (3 ** 0.5 * 0.4) / (
                    cos_phi_load)

                # determine suitable cable for this current
                suitable_cables = lvgd.lv_grid.network.static_data['LV_cables'][
                    (lvgd.lv_grid.network.static_data['LV_cables'][
                        'I_max_th'] * cable_lf) > I_max_branch]
                cable_type = suitable_cables.ix[
                    suitable_cables['I_max_th'].idxmin()]

                # create Dingo grid objects and add to graph
                for load_no in list(range(1, val['max_loads_per_branch'] + 1)):
                    # create a LV grid string and attached to station
                    lv_graph_attach_branch()

            # add remaining branch
            if val['remaining_loads'] > 0:
                if 'branch_no' not in locals():
                    branch_no = 0
                # determine maximum current occuring due to peak load of branch
                I_max_branch = (val['max_loads_per_branch'] *
                                val['single_peak_load']) / (3 ** 0.5 * 0.4) / (
                    cos_phi_load)

                # determine suitable cable for this current
                suitable_cables = lvgd.lv_grid.network.static_data['LV_cables'][
                    (lvgd.lv_grid.network.static_data['LV_cables'][
                        'I_max_th'] * cable_lf) > I_max_branch]
                cable_type = suitable_cables.ix[
                    suitable_cables['I_max_th'].idxmin()]

                branch_no = branch_no + 1

                for load_no in list(range(1, val['remaining_loads'] + 1)):
                    # create a LV grid string and attach to station
                    lv_graph_attach_branch()


def build_ret_ind_agr_branches(lvgd):
    """
    Determine topology of LV grid for retail/industrial and agricultural sector
    and create representative graph of the grid

    Parameters
    ----------
    lvgd : dingo.core.structure.regions.LVGridDistrictDingo
        Low-voltage grid district object
    """

    # determine topology of grid branches
    model_params = grid_model_params_ria(lvgd)

    # attach branches for sectors retail/industrial and agricultural
    build_lv_graph_ria(lvgd, model_params)


def select_grid_model_residential(lvgd):
    """
    Selects typified model grid based on population

    Parameters
    ----------
    lvgd : dingo.core.structure.regions.LVGridDistrictDingo
        Low-voltage grid district object

    Returns
    -------
    selected_strings_df: DataFrame
        Selected string of typified model grid
    transformer: Dataframe
        Parameters of chosen Transformer

    Notes
    -----
    In total 196 distinct LV grid topologies are available that are chosen
    by population in the LV grid district. Population is translated to
    number of house branches. Each grid model fits a number of house
    branches. If this number exceeds 196, still the grid topology of 196
    house branches is used. The peak load of the LV grid district is
    uniformly distributed across house branches.
    """

    # Load properties of LV typified model grids
    string_properties = lvgd.lv_grid.network.static_data['LV_model_grids_strings']
    # Load relational table of apartment count and strings of model grid
    apartment_string = lvgd.lv_grid.network.static_data[
        'LV_model_grids_strings_per_grid']

    # load assumtions
    apartment_house_branch_ratio = cfg_dingo.get("assumptions",
                                                 "apartment_house_branch_ratio")
    population_per_apartment = cfg_dingo.get("assumptions",
                                             "population_per_apartment")

    # calc count of apartments to select string types
    apartments = round(lvgd.population / population_per_apartment)

    if apartments > 196:
        apartments = 196

    # select set of strings that represent one type of model grid
    strings = apartment_string.loc[apartments]
    selected_strings = [int(s) for s in strings[strings >= 1].index.tolist()]

    # slice dataframe of string parameters
    selected_strings_df = string_properties.loc[selected_strings]

    # add number of occurences of each branch to df
    occurence_selector = [str(i) for i in selected_strings]
    selected_strings_df['occurence'] = strings.loc[occurence_selector].tolist()

    return selected_strings_df


def build_lv_graph_residential(lvgd, selected_string_df):
    """
    Builds nxGraph based on the LV grid model

    Parameter
    ---------
    lvgd : dingo.core.structure.regions.LVGridDistrictDingo
        Low-voltage grid district object
    selected_string_df: Dataframe
        Table of strings of the selected grid model

    Notes
    -----
    To understand what is happening in this method a few data table columns
    are explained here

    * `count house branch`: number of houses connected to a string
    * `distance house branch`: distance on a string between two house
        branches
    * `string length`: total length of a string
    * `length house branch A|B`: cable from string to connection point of a
        house

    A|B in general brings some variation in to the typified model grid and
    refer to different length of house branches and different cable types
    respectively different cable widths.
    """

    houses_connected = (
        selected_string_df['occurence'] * selected_string_df[
            'count house branch']).sum()

    average_load = lvgd.peak_load_residential / \
                   houses_connected

    hh_branch = 0

    # iterate over each type of branch
    for i, row in selected_string_df.iterrows():

        # get overall count of branches to set unique branch_no
        branch_count_sum = len(lvgd.lv_grid._graph.neighbors(lvgd.lv_grid.station()))

        # iterate over it's occurences
        for branch_no in range(1, int(row['occurence']) + 1):

            hh_branch += 1
            # iterate over house branches
            for house_branch in range(1, row['count house branch'] + 1):
                if house_branch % 2 == 0:
                    variant = 'B'
                else:
                    variant = 'A'

                # cable distributor to divert from main branch
                lv_cable_dist = LVCableDistributorDingo(
                    grid=lvgd.lv_grid,
                    string_id=i,
                    branch_no=branch_no + branch_count_sum,
                    load_no=house_branch)
                # add lv_cable_dist to graph
                lvgd.lv_grid.add_cable_dist(lv_cable_dist)

                # cable distributor within building (to connect load+geno)
                lv_cable_dist_building = LVCableDistributorDingo(
                    grid=lvgd.lv_grid,
                    string_id=i,
                    branch_no=branch_no + branch_count_sum,
                    load_no=house_branch,
                    in_building=True)
                # add lv_cable_dist_building to graph
                lvgd.lv_grid.add_cable_dist(lv_cable_dist_building)

                lv_load = LVLoadDingo(grid=lvgd.lv_grid,
                                      string_id=i,
                                      branch_no=branch_no + branch_count_sum,
                                      load_no=house_branch,
                                      peak_load=average_load)

                # add lv_load to graph
                lvgd.lv_grid.add_load(lv_load)

                cable_name = row['cable type'] + \
                             ' 4x1x{}'.format(row['cable width'])
                cable_type = lvgd.lv_grid.network.static_data[
                    'LV_cables'].loc[cable_name]

                # connect current lv_cable_dist to station
                if house_branch == 1:
                    # edge connect first house branch in branch with the station
                    lvgd.lv_grid._graph.add_edge(
                        lvgd.lv_grid.station(),
                        lv_cable_dist,
                        branch=BranchDingo(
                            length=row['distance house branch'],
                            kind='cable',
                            type=cable_type,
                            id_db='branch_{sector}{branch}_{load}'.format(
                                branch=hh_branch,
                                load=house_branch,
                                sector='HH')
                        ))
                # connect current lv_cable_dist to last one
                else:
                    lvgd.lv_grid._graph.add_edge(
                        lvgd.lv_grid._cable_distributors[-4],
                        lv_cable_dist,
                        branch=BranchDingo(
                            length=row['distance house branch'],
                            kind='cable',
                            type=lvgd.lv_grid.network.static_data[
                                'LV_cables'].loc[cable_name],
                            id_db='branch_{sector}{branch}_{load}'.format(
                                branch=hh_branch,
                                load=house_branch,
                                sector='HH')))

                # connect house to cable distributor
                house_cable_name = row['cable type {}'.format(variant)] + \
                                   ' 4x1x{}'.format(
                                       row['cable width {}'.format(variant)])
                lvgd.lv_grid._graph.add_edge(
                    lv_cable_dist,
                    lv_cable_dist_building,
                    branch=BranchDingo(
                        length=row['length house branch {}'.format(
                            variant)],
                        kind='cable',
                        type=lvgd.lv_grid.network.static_data['LV_cables']. \
                            loc[house_cable_name],
                        id_db='branch_{sector}{branch}_{load}'.format(
                            branch=hh_branch,
                            load=house_branch,
                            sector='HH'))
                )

                lvgd.lv_grid._graph.add_edge(
                    lv_cable_dist_building,
                    lv_load,
                    branch=BranchDingo(
                        length=1,
                        kind='cable',
                        type=lvgd.lv_grid.network.static_data['LV_cables']. \
                            loc[house_cable_name],
                        id_db='branch_{sector}{branch}_{load}'.format(
                            branch=hh_branch,
                            load=house_branch,
                            sector='HH'))
                )


def build_residential_branches(lvgd):
    """
    Based on population and identified peak load data, the according grid
    topology for residential sector is determined and attached to the grid graph

    Parameters
    ----------
    lvgd : dingo.core.structure.regions.LVGridDistrictDingo
        Low-voltage grid district object
    """

    # Choice of typified lv model grid depends on population within lv
    # grid district. If no population is given, lv grid is omitted and
    # load is represented by lv station's peak load
    if lvgd.population > 0 \
            and lvgd.peak_load_residential > 0:
        model_grid = select_grid_model_residential(lvgd)

        build_lv_graph_residential(lvgd, model_grid)

    # no residential load but population
    elif lvgd.population > 0 \
            and lvgd.peak_load_residential == 0:
        logger.warning(
            '{} has population but no residential load. '
            'No grid is created.'.format(
                repr(lvgd)))

    # residential load but no population
    elif lvgd.population == 0 \
            and lvgd.peak_load_residential > 0:
        logger.warning(
            '{} has no population but residential load. '
            'No grid is created and thus this load is '
            'missing in overall balance!'.format(
                repr(lvgd)))

    else:
        logger.info(
            '{} has got no residential load. '
            'No grid is created.'.format(
                repr(lvgd)))