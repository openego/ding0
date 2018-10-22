Getting started
~~~~~~~~~~~~~~~

.. _installation:

Installation
============

.. note::
    Installation is only tested on (debian like) linux OS.

Ding0 is provided through PyPi package management and, thus, installable from
sources of pip3.
You may need to additionally install some specific system packages for a
successful installation of Ding0 and its dependencies.

The script `ding0_system_dependencies.sh` installs required system package
dependencies.

.. code-block:: bash

    cd <your-ding0-install-path>

    chmod +x ding0_system_dependencies.sh

    sudo ./ding0_system_dependencies.sh


We recommend installing Ding0 (and in general third-party) python packages in a
virtual environment, encapsulated from the system python distribution.
This is optional. If you want to follow our suggestion, install the tool
`virtualenv <https://virtualenv.pypa.io/en/stable/>`_ by

.. code-block:: bash

    sudo apt-get install virtualenv # since Ubuntu 16.04

Afterwards `virtualenv` allows you to create multiple parallel python distributions.
Since Ding0 relies on Python 3, we specify this in the virtualenv creation.
Create a new one for Ding0 by

.. code-block:: bash

    # Adjust path to your specific needs
    virtualenv -p python3 ~/.virtualenvs/ding0

Jump into (aka. activate) this python distribution by

.. code-block:: bash

    # Adjust path to your specific needs
    source ~/.virtualenvs/ding0/bin/activate

From that, the latest release of Ding0 is installed by

.. code-block:: python

    pip3 install ding0


Pip allows to install a developer version of a package that uses currently
checked out code. A developer mode installation is achieved by

.. code-block:: python

    pip3 install -e path/to/cloned/ding0/repository
    

Use Ding0
=========

Have a look at the :ref:`ding0-examples`.