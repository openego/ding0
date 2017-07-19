Notes to developers
~~~~~~~~~~~~~~~~~~~

If you're interested to contribute and join the project. Feel free to submit
PR, contact us, or just create an issue if something seems odd.


Test the package installation
=============================

We use `Docker <https://www.docker.com/>`_ to test the build of
dingo on a fresh Ubuntu OS. In order to run such a test make sure docker is
installed

.. code-block:: bash

    chmod +x install_docker.sh
    ./install_docker.sh

Afterwards you can test if installation of dingo builts successfully by
executing

.. code-block:: bash

    ./check_dingo_installation.sh


The script :code:`./check_dingo_installation.sh` must be executed in root
directory of dingo repository. Then it
installs currently checked out version. The installation process can be observed
in the terminal.


Test dingo runs
===============

The outcome of different runs of dingo can be compared with the functions in
`~/dingo/tools/tests.py <api/dingo.tools.html#module-dingo.tools.tests>`_.

To compare the default configuration of a fresh run of dingo and a saved run use

.. code-block:: python

    manual_dingo_test()

The default behavior is using district [3545] in oedb database and the data in
file 'dingo_tests_grids_1.pkl'.
For other filenames or districts use, for example:

.. code-block:: python

    manual_dingo_test([438],'dingo_tests_grids_2.pkl')

To create a file with the output of a dingo run in the default configuration
(disctrict [3545] in oedb database and
filename 'dingo_tests_grids_1.pkl') use:

.. code-block:: python

    init_files_for_tests()

For other filenames or districts use, for example:

.. code-block:: python

    init_files_for_tests([438],'dingo_tests_grids_2.pkl')

To run the automatic unittest suite use:

.. code-block:: python

    support.run_unittest(DingoRunTest)

The suite assumes that there are two files allocated in the directory:

* 'dingo_tests_grids_1.pkl'

* 'dingo_tests_grids_2.pkl'

It is assummed that these files store the outcome of different runs of dingo
over different districts.

This suite will run three tests:

* Compare the results stored in the files,
  testing for equality between the data in 'dingo_tests_grids_1.pkl' and itself;
  and for difference between both files.

* Compare the results of a fresh dingo run over district [3545] and the data in
'dingo_tests_grids_1.pkl'.

* Compare the results of two fresh runs of dingo in district [3545].