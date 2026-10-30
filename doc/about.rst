.. _theoretical_background:

############################
Background and Methodology
############################
Ding0 is a tool designed for the generation of synthetic distribution
networks, serving as a valuable resource for simulating and analyzing electrical
distribution systems. This section provides an in-depth understanding of the
theoretical foundations and core processes that underlie the functionality of ding0.
The fundamental data basis is described in [Huelk2017]_ and its extension is
detailed by [Amme2017]_. Further refinements are contributed by [Dubielzig]_ and [John]_.
:ref:`Ding0 flowchart` indicates the methodology of ding0 which is described more in detail within this section.

    .. _Ding0 flowchart:
    .. figure:: images/flowchart_mv_grids.png

            Ding0 flowchart

Throughout the tool, a nesting has been implemented that is shown in
:ref:`Ding0 Structure`.

    .. _Ding0 Structure:
    .. figure:: images/ding0_basic_structure.png

            Ding0 Structure

1. Data Import and Integration
===============================

The first step in the ding0 workflow involves the import of essential input data from two primary sources:

1.1 **eGon-data**
    The input data is read from a database created by `eGon-data <https://github.com/openego/eGon-data>`_,
    supplying critical grid-related information. For each distribution network, the following data types are imported:

    ======================== =====
    Data Type                    Definition
    ======================== =====
    Load Areas                  Geographical clusters designed for electrical loads
    Building Loads              Electrical loads assigned to specific buildings
    MVGD                        Polygon defining the geographical area of the Medium Voltage Grid District (MVGD)
    HV/MV station               Location of the grid connection point between high and medium voltage level
    Generators                  Renewable and conventional power generation sources located in the regarded MVGD
    ======================== =====

1.2 **OpenStreetMap (OSM)** [OSM]_
    OSM provides the street network data, forming the geographical basis of the distribution network.
    The street network is also read from the eGon-data database.

2. Data Processing
===================
2.1 **Allocation of Loads and Generators to belonging voltage level**

    **Loads**. High-resolution load data, which are allocated to specific buildings within the network area is categorized.
    This categorization involves associating them with a specific voltage level, thereby determining the type of
    grid connection they require. All loads **>1MVA** are considered for the MV ring topology (if connectivity is high).
    Others are connected via stub connection to the assigned grid level.

    ================== ================== =========================
    Voltage level       Nominal capacity    Allocation target
    ================== ================== =========================
    MV                  0.2 - 5.5 MW        MV grid
    MV-LV               0.1 - 0.2 MW        MV/LV substation
    LV                  < 0.1 MW            LV grid
    ================== ================== =========================

    Loads with a capacity of 5.5 MW or more are currently not considered.

    **Generators**. Depending on their nominal capacity, power generation sources are tagged with different allocation targets.
    Ding0 does not contemplate generators when building the initial grid topology. They are connected through stub
    connections to the grid topology afterwards as they play an important role for power flow analysis.

    ================== ================== =========================
    Voltage level       Nominal capacity    Allocation target
    ================== ================== =========================
    level 4 (HV-MV)     4.5 - 17.5 MW       HV/MV substation
    level 5 (MV)        0.3 - 4.5 MW        MV grid
    level 6 (MV-LV)     0.1 - 0.3 MW        MV/LV substation
    level 7 (LV)        ≤ 0.1               LV grid
    ================== ================== =========================

2.2 **Clustering, Partitioning and Positioning**

    **Clustering and Partitioning.** The clustering process divides the road graph into subnetworks while considering
    the inherent structure of the graph during cluster analysis. These resulting clusters, when
    combined with the associated buildings, collectively constitute the Low Voltage Grid Districts (LVGDs)
    within a Load Area (LA).

    **Positioning.** In each LVGD, the street point located at the load center of the LV grid is calculated,
    considering only LV loads. This load center is utilized to determine the positioning of the MV/LV substation.

    .. figure:: images/clustering_positioning_partitioning_.png

        Clustering, Partitioning and Positioning of substations

    Details are presented in [Dubielzig]_.

2.3 **Parametrization and Validation**

    **Parametrization**. In the parameterization phase, technical specifications for the MVGD are determined, including voltage levels
    based on the distance between LA centers and the HV/MV substation: if the maximum distance is 15 km or more, the grid is operated
    at Vₙ = 20 kV, otherwise at Vₙ = 10 kV. Alternatively, the voltage level can be chosen by load density (1 MVA/km² or more results
    in 10 kV). This defines the use of underground cables (10 kV) or overhead lines (20 kV) in the network. **Aggregated LAs** are identified when

    .. math::
        P_{LA} > \sqrt{3} \cdot V_n \cdot I_{max,th}


    classifying them as urban regions. :math:`P_{LA}` refers to the peak load of the regarded LA and :math:`I_{max,th}` to
    the thermal current limit of the largest available line or cable type. Transformer types are chosen based on peak loads,
    with HV/MV transformers operating at up to 60% load and MV/LV transformers at up to 100% load, ensuring redundancy and (n-1) security
    for the substation.

3. Low Voltage (LV) Grid Construction
=======================================
    .. figure:: images/LVGD.png

        Example of LVGD

3.1 **Clustering and partitioning**

    - The LV grid generation process begins with the clustering of LV Load Areas (LAs) based on the capacity of LV loads within each LA.
    - Clustering results in the formation of clusters, and a clustered graph is positioned according to street topology.
    - These clusters serve as the foundation for creating LV Grid Districts (LVGDs).

3.2 **MV/LV substation Placement**

    - LVGDs are defined, partitioning LAs into LVGDs, each of which is associated with a load center.
    - MV/LV stations are strategically positioned at the load centers of LVGDs.

3.3 **Building the LV Grid**

    - The LV grid is then constructed within each LVGD, utilizing the OSM network.
    - Loads with a demand of less than 100 kW are directly connected to the street graph.
    - Loads with a demand between 100 kW and 200 kW are linked directly to the MV/LV station.
    - Branching occurs from the MV/LV station based on capacity constraints, finalizing the LV grid topology.
    - LV generators are integrated into the LV grid topology, with two possible connection levels:
        - Level 6: Connects generators to LV stations.
        - Level 7: Connects generators to the closest LV-grid node.

4. Medium Voltage (MV) Grid Construction
=========================================

    .. figure:: images/MVGD.png

        Example of MVGD

**4.1 MV Grid in rural areas (regular & satellite LAs)**

    **Assumptions** are established that underlie the entire MV grid generation process

    ========================================  ==================================================================================================
     Assumption                               Value
    ========================================  ==================================================================================================
     Type of topology                         Open ring topology
     Voltage Level (MV)                       20 kV (if the distance to an LA center is 15 km or more), 10 kV (otherwise)                    
     Preferred Cable Type (10 kV)             Earth Cables
     Preferred Cable Type (20 kV)             Overhead Lines
     Maximum Line Loading normal              Up to 60%
     Maximum Line Loading failure             Up to 100%
     Maximum Voltage Drop normal              5 %
     Maximum Voltage Drop failure             10 %
     Detour Factor                            1.3
     Reactive Power requirements loads        cos(φ)=0.97
     Reactive Power requirements generators   cos(φ)=1
    ========================================  ==================================================================================================

    **Routing.** The routing for the initial grid topology in the context of the ding0 tool is
    based on the Capacitated Vehicle Routing Problem (CVRP) formulation which is solved by a two-stage
    metaheuristic approach. The classic CVRP optimization problem is adapted for designing the MV grid,
    with the objective of determining the most efficient routes for supplying electricity to different
    MV grid connection points (MV/LV substations and MV loads).
    First, the initial routes are constructed using a parallel savings heuristic of
    Clarke and Wright. This heuristic identifies potential savings by combining routes and iteratively
    improves the solution. Only centers of LAs are regarded as potential points for the algorithm.
    Second, local search heuristics are used to refine the routes further.

    ========================================  =============================== ====================================================================
    CVRP notation                             Application to grid planning      Explanation
    ========================================  =============================== ====================================================================
    customers                                 Centers of LAs                    Location that needs to be visited
    depot                                     HV/MV substation                  Central point from which the MV grid routes originate and return
    ========================================  =============================== ====================================================================

    Throughout the routing process, various technical constraints are considered, including current carrying capacity,
    voltage stability, load factor, line loading, and operational modes (normal and faulty). These constraints ensure that the designed grid
    remains technically feasible and reliable.

    **Grid extension.** The initial MV grid topology is extended by those MV grid connection points that were left out above
    due to proximity reasons or technical constraints. Three sequential steps are executed to connect satellite LAs, MV/LV substations,
    and generation units to the existing grid are proceeded by applying the order of connection respectively.

        **(1)** Nodes within a proximity of ≤ 100 meters to an existing grid route are integrated into the grid by adjusting the route's path.

        **(2)** Geographic Information System (GIS) methods are used to find and connect remaining nodes, starting with a search radius of 2000 meters.
        The radius expands incrementally if no suitable points are found.

        **(3)** If the above connection options are infeasible due to technical constraints, nodes are directly connected to the main route using
        separate branch lines and the standard line type.

    Details on routing principles for MV grid topology are presented in [Amme2017]_. Be aware that there have been major changes in
    methodology since the publishing of this paper:

    ==================================================   ======================================================================================================
     Initial methodology                                    Update
    ==================================================   ======================================================================================================
     Sector-specific electricity demand                     High resolution load data for each building
     Equidistant grid of points for MV/LV substations       Location of MV/LV substations are based on load center of each LVGD
     Voronoi partition for LVGD definition                  Definition of LVGDs as a result of clustering by loads
     No MV loads                                            Heavy load electricity consumers are classified as MV loads and connected within that voltage level
    ==================================================   ======================================================================================================

**4.2 MV Grid in urban areas (aggregated LAs)**

Aggregated Load Areas (LAs) are characterized by a high cumulative power demand, classifying them as urban regions.
In these LAs, each is connected to the HV/MV substation through at least one direct connection, which depends on the cumulative load.
To design the MV grid within aggregated LAs, actual road network distances (OSM) are considered. The graph representing this network
is preprocessed and divided into two components: G_core, which contains potential MV-ring topology customers, and G_stub, which
includes the remaining customers for MV grid connection via stubs.

Initially, all well-connected customers (those with at least two neighbors) are primarily assigned to G_core.
Subsequently, the stub connection criteria are evaluated, considering a load threshold of <1 MVA. Customers exceeding
this threshold are also moved from G_stub to G_core to ensure grid stability. The final G_core graph encompasses all customers considered
for the initial MV ring topology. The routing procedure aligns with that used in rural areas, with the additional constraint of
network distances being considered instead of air distances. The Dijkstra Algorithm plays a crucial role in determining the precise
road geometries of ring circuits and their lengths.

    ====================================  ==============================  ===================================================
    Criterion                             Rural area                      Urban area
    ====================================  ==============================  ===================================================
    Distance calculation                  Air distances * detour factor   Street network
    Customers for initial ring topology   LA centers                      Final G_core components (MV/LV stations + MV loads)
    Customers for stub connection         Missing LA centers,             Final G_stub components (MV/LV stations + MV loads)
                                          generators, MV/LV stations
    Voltage level                         10 kV and 20 kV                 Mostly 10 kV
    Connection type                       Cables and overhead lines       Mostly cables (underground)
    ====================================  ==============================  ===================================================

Details on MV grid design in urban areas are presented in [Dubielzig]_.

5. Grid Analysis and Reinforcement
===================================


**5.1 Relocating Switch Disconnectors and Power Flow Analysis**

Switch disconnectors are used to realize redundancy within the grid in case of an outage of a component.
MV-rings can be operated as isolated half-rings. The grid thus does
meet minimum technical requirements. A ring’s switch disconnector was initially located on the line where power
flow is minimal in closed condition. Due to the extension of the grid, additional loads are
connected to the rings. Therefore, the switch disconnectors are subsequently relocated to fulfill
this requirement. For further power flow analysis, results are exported to a PyPSA format.

**5.2 Grid Reinforcement**

Grid reinforcement measures are employed to enhance the MV grid robustness (LV grid reinforcement and transformer reinforcement is not implemented yet).
The main purpose is to address overloading issues within the grid and ensure stable operation.
The following steps are applied:

        **(1)** Identify critical branches and stations with overloading issues.

        **(2)** Reinforce critical branches by selecting appropriate cable types.
        If no suitable cable type is found for a branch, the branch's original type is retained.

        **(3)** Run a power flow analysis to check for voltage issues.

        **(4)** Select larger cable types for the critical branches to address voltage issues.
        For each node where voltage exceeds 2 % of nominal voltage in feed-in case or 5 %
        of nominal voltage in load case, branch segments connecting the node with the substation
        are reinforced until no further issues remain.

Once grid reinforcement is complete, switch disconnectors are closed, finalizing the network configuration.

References
----------
.. [Amme2017] J. Amme, G. Pleßmann, J. Bühler, L. Hülk, E. Kötter, P. Schwaegerl:
    *The eGo grid model: An open-source and open-data based synthetic medium-voltage
    grid model for distribution power supply systems*. Journal of Physics Conference
    Series 977(1):012007, 2018, `doi:10.1088/1742-6596/977/1/012007 
    <http://iopscience.iop.org/article/10.1088/1742-6596/977/1/012007>`_
.. [Huelk2017] L. Hülk, L. Wienholt, I. Cussmann, U. Mueller, C. Matke and E.
    Kötter: *Allocation of annual electricity consumption and power
    generation capacities across multi voltage levels in a high spatial
    resolution*. International Journal of Sustainable Energy Planning and Management
    Vol. 13 2017 79–92, `doi:10.5278/ijsepm.2017.13.6 <https://doi.org/10.5278/ijsepm.2017.13.6>`_
.. [Dubielzig] P. Dubielzig: Modellierung synthetischer Verteilnetztopologien
    in urbanen Gebieten, Master's thesis, TU Berlin, 2022
.. [John] R. John: Planning of Synthetic Low Voltage Networks
    with Geographical Constraints, Master's thesis, Offenburg University, 2021
.. [OSM] OpenStreetMap contributors:
    `Open street map <https://www.openstreetmap.org>`_, 2017
