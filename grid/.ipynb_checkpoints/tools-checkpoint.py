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


def cable_type(nom_power, nom_voltage, avail_cables):
    """Determine suitable type of cable for given nominal power

    Based on maximum occurring current which is derived from nominal power
    (either peak load or max. generation capacity) a suitable cable type is
    chosen. Thus, no line overloading issues should occur.

    Parameters
    ----------
    nom_power : float
        Nominal power of generators or loads connected via a cable
    nom_voltage : float
        Nominal voltage in kV
    avail_cables : :pandas:`pandas.DataFrame<dataframe>`
        Available cable types including it's electrical parameters
    
    Returns
    -------
    :pandas:`pandas.DataFrame<dataframe>`
        Parameters of cable type
    """

    I_max_load = nom_power / (3 ** 0.5 * nom_voltage)

    # determine suitable cable for this current
    suitable_cables = avail_cables[avail_cables['I_max_th'] > I_max_load]
    if not suitable_cables.empty:
        cable_type = suitable_cables.loc[suitable_cables['I_max_th'].idxmin(), :]
    else:
        cable_type = avail_cables.loc[avail_cables['I_max_th'].idxmax(), :]

    return cable_type