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
