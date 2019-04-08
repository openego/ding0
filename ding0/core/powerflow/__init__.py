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


from datetime import datetime
from pandas import DatetimeIndex


class PFConfigDing0:
    """ Defines the PF scenario configuration

    Parameters
    ----------
    scenarios:  :any:`list` of  :obj:`str`
        List of strings describing the scenarios
    timerange: :any:`list` of :pandas:`pandas.DatetimeIndex<datetimeindex>`  
        List of Pandas DatetimeIndex objects
    timesteps_count: int
        count of timesteps the timesteps to be created
    timestep_start: :pandas:`pandas.DatetimeIndex<datetimeindex>`  
        Description #TODO
    resolution: str
        String or pandas offset object, e.g. 'H'=hourly resolution,
        
        to learn more see http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
    srid: type
        partial reference system indentifier used by PyPSA's plots #TODO

    Notes
    -----
    This class can be called as follows:
    
    i) With scenarios and timeranges::
    
         scenarios = ['scn_1', ...,  'scn_n'],
         timeranges= [timerange_1, ..., timerange_n]
    ii) With scenarios, start time and count of timesteps::
     
         scenarios = ['scn_1', ...,  'scn_n'],
         timesteps_count = m,
         timestep_start = datetime()
    
    (in this case, n timeranges with m timesteps starting from datetime will be created)
    """

    def __init__(self, **kwargs):
        self._scenarios = kwargs.get('scenarios', None)
        self._timeranges = kwargs.get('timeranges', None)
        self._resolution = kwargs.get('resolution', 'H')
        self._srid = kwargs.get('srid', 4326)

        timesteps_count = kwargs.get('timesteps_count', None)
        timestep_start = kwargs.get('timestep_start', None)

        if self._scenarios is None:
            raise ValueError('PF config: Please set at least one scenario.')

        if not isinstance(self._timeranges, DatetimeIndex):
            if not isinstance(timesteps_count, int) or not isinstance(timestep_start, datetime):
                raise ValueError('PF config: Either timerange (pandas DatetimeIndex object) or ' +
                                 'timesteps_count (int) with start time (datetime object) must be set.')
            else:
                # create pandas DatetimeIndex object for given values
                self._timeranges = []
                for _ in enumerate(self._scenarios):
                    self._timeranges.append(DatetimeIndex(freq=self._resolution,
                                                          periods=timesteps_count,
                                                          start=timestep_start))

        elif len(self._scenarios) != len(self._timeranges):
            raise ValueError('PF config: Count of scenarios has to equal count of timeranges.')


    @property
    def scenarios(self):
        """ Returns a generator for iterating over PF scenarios """
        for scenario in self._scenarios:
            yield scenario

    @property
    def timesteps(self):
        """ Returns a generator for iterating over PF timesteps """
        for timerange in self._timeranges:
            yield timerange

    @property
    def resolution(self):
        """ Returns resolution """
        return self._resolution

    @property
    def srid(self):
        """ Returns SRID"""
        return self._srid


def q_sign(reactive_power_mode_string, sign_convention):
    """
    Gets the correct sign for Q time series given 'inductive' and 'capacitive' and the 'generator'
    or 'load' convention.

    Parameters
    ----------
    reactive_power_mode_string: :obj:`str`
        Either 'inductive' or 'capacitive'
    sign_convention: :obj:`str`
        Either 'load' or 'generator'
    Return
    ------
    :obj: `int` : +1 or -1
        A sign to mulitply to Q time sereis
    """

    comparestr = reactive_power_mode_string.lower()

    if comparestr == 'inductive':
        if sign_convention == 'generator':
            return -1
        elif sign_convention == 'load':
            return 1
        else:
            raise ValueError("Unknown sign conention {}".format(sign_convention))
    elif comparestr == 'capacitive':
        if sign_convention =='generator':
            return 1
        elif sign_convention =='load':
            return -1
        else:
            raise ValueError("Unknown sign convention {}".format(sign_convention))
    else:
        raise ValueError("Unknown value {} in power_factor_mode".format(reactive_power_mode_string))