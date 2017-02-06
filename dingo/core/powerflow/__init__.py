from datetime import datetime
from pandas import DatetimeIndex


class PFConfigDingo:
    """ Defines the PF scenario configuration

    Args:
        scenarios: List of strings describing the scenarios
        timerange: List of Pandas DatetimeIndex objects
        timesteps_count: int, count of timesteps the timesteps to be created
        timestep_start: Datetime datetime object
        resolution: String or pandas offset object, e.g. 'H'=hourly resolution,
                    to learn more see http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
        srid: Spatial reference system indentifier used by PyPSA's plots

    Notes:
        This class can be called as follows:
            1. With scenarios and timeranges: scenarios = ['scn_1', ...,  'scn_n'],
                                              timeranges= [timerange_1, ..., timerange_n]
            2. With scenarios, start time and count of timesteps: scenarios = ['scn_1', ...,  'scn_n'],
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
        elif len(self._scenarios) != len(self._timeranges):
            raise ValueError('PF config: Count of scenarios has to equal count of timeranges.')

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
        """ Returns SRID """
        return self._srid
