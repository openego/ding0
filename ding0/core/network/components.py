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


class GridComponentDing0:
    """Load or storage unit at a location of its own, connected to the grid
    by its voltage level like a generator.

    Voltage level 4 is connected to the HV/MV station by a line of its own,
    5 to the nearest MV node or line, 6 to the nearest LV station and 7 to the
    nearest LV load.

    Attributes
    ----------
    id_db : :obj:`str`
        Id of the component, unique per component type
    geo_data : :shapely:`Shapely Point object<points>`
        Location in EPSG:3035
    mv_grid : :class:`~.ding0.core.network.grids.MVGridDing0`
        MV grid the component belongs to
    lv_grid : :class:`~.ding0.core.network.grids.LVGridDing0`
        LV grid the component is connected to, None in the MV grid
    lv_load_area : :class:`~.ding0.core.structure.regions.LVLoadAreaDing0`
    v_level : :obj:`int`
        Voltage level 4 to 7
    p_nom : :obj:`float`
        Grid connection power in kW
    type : :obj:`str`
        Component type as in eDisGo, e.g. 'charging_point'
    building_id : :obj:`int`
    source_id : :obj:`str`
        Id of the component in egon-data
    """

    # prefix of names and bus names
    name_prefix = 'Component'

    def __init__(self, **kwargs):
        self.id_db = kwargs.get('id_db', None)
        self.geo_data = kwargs.get('geo_data', None)
        self.mv_grid = kwargs.get('mv_grid', None)
        self.lv_grid = kwargs.get('lv_grid', None)
        self.lv_load_area = kwargs.get('lv_load_area', None)
        self.v_level = kwargs.get('v_level', None)
        self.p_nom = kwargs.get('p_nom', None)
        self.type = kwargs.get('type', None)
        self.building_id = kwargs.get('building_id', None)
        self.source_id = kwargs.get('source_id', None)

    @property
    def network(self):
        return self.mv_grid.network

    @property
    def capacity(self):
        """Grid connection power in kW, as `capacity` of generators; used to
        choose cables when the component is connected."""
        return self.p_nom

    @property
    def pypsa_bus_id(self):
        if self.lv_grid is not None:
            return '_'.join(['Bus', 'mvgd', str(self.mv_grid.id_db), 'lvgd',
                             str(self.lv_grid.id_db), self.type, str(self.id_db)])
        return '_'.join(['Bus', 'mvgd', str(self.mv_grid.id_db), self.type,
                         str(self.id_db)])

    def __repr__(self):
        if self.lv_grid is not None:
            return '_'.join([self.name_prefix, 'mvgd', str(self.mv_grid.id_db),
                             'lvgd', str(self.lv_grid.id_db), self.type,
                             str(self.id_db)])
        return '_'.join([self.name_prefix, 'mvgd', str(self.mv_grid.id_db),
                         self.type, str(self.id_db)])


class PointLoadDing0(GridComponentDing0):
    """Load at a location of its own: charging points (also of bus depots)
    and rail substations.

    Attributes
    ----------
    type : :obj:`str`
        'charging_point' or 'conventional_load', as in eDisGo
    sector : :obj:`str`
        eDisGo sector; for charging points one of 'home', 'work', 'public'
        and 'hpc'
    use_case : :obj:`str`
        Use case in egon-data, e.g. 'urban_fast' or 'bus_depot'
    """

    name_prefix = 'Load'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sector = kwargs.get('sector', None)
        self.use_case = kwargs.get('use_case', None)


class StorageUnitDing0(GridComponentDing0):
    """Battery storage unit at a location of its own.

    Attributes
    ----------
    max_hours : :obj:`float`
        Energy capacity divided by `p_nom`
    control : :obj:`str`
        Control of the storage in power flows, 'PQ' as in eDisGo
    """

    name_prefix = 'Storage'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_hours = kwargs.get('max_hours', None)
        self.control = kwargs.get('control', 'PQ')
