# reinforcement measures according to Ackermann
import os
import dingo
import pandas as pd
from dingo.tools import config as cfg_dingo
package_path = dingo.__path__[0]


def reinforce_branches(grid, crit_branches, mode):
    """ Reinforce MV or LV grid by installing a new branch/line type

    Args:
        grid: GridDingo object
        crit_branches; Dict of critical branches with max. relative overloading
        mode: 'determined' (used for reinforcement measures due to current violations: set appropriate cable using
                            the relative overload -> appropriate type can be determined) or
              'next' (used for reinforcement measures due to voltage violations: set next larger cable available)

    Returns:

    Notes:
        The branch type to be installed is determined per branch using the rel. overloading. According to Ackermann
        only cables are used.
    """
    # TODO: Move equipment data to attr. of nd
    # load cable/line assumptions, file_names and parameter
    load_factor_normal = float(cfg_dingo.get('assumptions',
                                             'load_factor_mv_cable_lc_normal'))
    equipment_parameters_file = cfg_dingo.get('equipment',
                                              'equipment_mv_parameters_cables')
    branch_parameters = pd.read_csv(os.path.join(package_path, 'data',
                                    equipment_parameters_file),
                                    comment='#',
                                    converters={'I_max_th': lambda x: int(x), 'U_n': lambda x: int(x)})
    branch_parameters = branch_parameters[branch_parameters['U_n'] == grid.v_level].sort_values('I_max_th')

    for branch, rel_overload in crit_branches.items():
        try:
            type = branch_parameters.ix[branch_parameters[branch_parameters['I_max_th'] >=
                   branch['branch'].type['I_max_th'] * rel_overload]['I_max_th'].idxmin()]
            branch['branch'].type = type
        except:
            print('Branch', branch, 'could not be reinforced as there is no appropriate cable type available.',
                                    'Original type is retained.')
            pass

def extend_substation(grid):
    """ Reinforce MV or LV substation by exchanging the existing trafo and installing a parallel one if necessary with

    Args:
        grid: GridDingo object

    Returns:

    """
    pass


def new_substation(grid):
    """ Reinforce MV grid by installing a new primary substation opposite to the existing one

    Args:
        grid: MVGridDingo object

    Returns:

    """