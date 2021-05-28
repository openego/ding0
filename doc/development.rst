Notes to developers
~~~~~~~~~~~~~~~~~~~

If you're interested to contribute and join the project, feel free to submit
PR, contact us, or just create an issue if something seems odd.


Test the package installation
=============================

.. warning::

    The scripts for testing the installation might be outdated and will be
    revised in v0.2.1.

We use `Docker <https://www.docker.com/>`_ to test the build of
ding0 on a fresh Ubuntu OS. In order to run such a test make sure docker is
installed.

.. code-block:: bash

    cd ./test_installation/
    chmod +x install_docker.sh
    ./install_docker.sh

Afterwards you can test if installation of ding0 builts successfully by
executing

.. code-block:: bash

    ./check_ding0_installation.sh


The script :code:`./check_ding0_installation.sh` must be executed in root
directory of ding0 repository. Then it
installs currently checked out version. The installation process can be observed
in the terminal.

Run unit and integration tests
==============================

ding0 comes with a bunch of unit and integration tests on most features. You'll
need additional packages listed in `dev_requirements.txt`. To install, use

.. code-block:: bash

    pip install -r /path/to/ding0/dev_requirements.txt

To run tests with e.g. 4 workers (you may omit this argument), use

.. code-block:: bash

    cd /path/to/ding0/
    pytest --workers 4 -vv

Test ding0 runs
===============

The outcome of different runs of ding0 can be compared with the functions in
`~/ding0/tools/tests.py <api/ding0.tools.html#module-ding0.tools.tests>`_.

To compare the default configuration of a fresh run of ding0 and a saved run use

.. code-block:: python

    manual_ding0_test()

The default behavior is using district [3545] in oedb database and the data in
file 'ding0_tests_grids_1.pkl'.
For other filenames or districts use, for example:

.. code-block:: python

    manual_ding0_test([438],'ding0_tests_grids_2.pkl')

To create a file with the output of a ding0 run in the default configuration
(disctrict [3545] in oedb database and
filename 'ding0_tests_grids_1.pkl') use:

.. code-block:: python

    init_files_for_tests()

For other filenames or districts use, for example:

.. code-block:: python

    init_files_for_tests([438],'ding0_tests_grids_2.pkl')

To run the automatic unittest suite use:

.. code-block:: python

    support.run_unittest(Ding0RunTest)

The suite assumes that there are two files allocated in the directory:

* 'ding0_tests_grids_1.pkl'

* 'ding0_tests_grids_2.pkl'

It is assummed that these files store the outcome of different runs of ding0
over different districts.

This suite will run three tests:

* Compare the results stored in the files,
  testing for equality between the data in 'ding0_tests_grids_1.pkl' and itself;
  and for difference between both files.

* Compare the results of a fresh ding0 run over district [3545] and the data in
  'ding0_tests_grids_1.pkl'.

* Compare the results of two fresh runs of ding0 in district [3545].

