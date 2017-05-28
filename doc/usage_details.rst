.. _dingo-examples:

How to use dingo?
~~~~~~~~~~~~~~~~~

Examples
========

We provide two examples of how to use Dingo along with two example for analysis
of resulting data. The
:download:`first example <../examples/example_single_grid_district.py>` shows how Dingo
is applied to a single medium-voltage grid district. Grid topology for the
medium- and low-voltage grid level is generated an export to the *OEDB* and
save to file (.pkl).
The :download:`analysis script <../examples/example_analyze_single_grid_district.py>`
takes data generated the first example and produces exemplary output: key
figures and plots.

The second example shows how to generate a larger number of grid topology data
sets.
As the current data source sometimes produces unuseful data or leads to program
execution interruption, these are excluded from grid topology generation. This
is enable by setting :code:`failsafe=` to `True`.
The according analysis script provides examplary plot for data of multiple grid
districts.


Analysis of grid data
=====================

With the two above mentioned analysis scripts we show the principle of curent
results data analysis.
The principle can be cut down to

**First**, obtain data on nodes and edges objects of a single or multiple grid
districts

.. code-block:: python

    # export key numbers from `nd`-object
    nodes, edges = nd.to_dafatrame()


**Second**, generate statistics based nodes and edges data

.. code-block:: python

    # put key figures of each grid district into separate DataFrame
    stats = results.calculate_mvgd_stats(nodes, edges)

Thereof, you can create plot and additional analyses. Feel free to add yours!