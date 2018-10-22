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
    
Setup database connection
==========================
 
Ding0 relies on data provided in the `OpenEnergy DataBase (oedb) <https://openenergy-platform.org/dataedit/>`_.
In order to use ding0 you therefore need an account on the 
`OpenEnergy Platform (OEP) <https://openenergy-platform.org/>`_. You can create a new account
`here <http://openenergy-platform.org/login/>`_.

The package `ego.io <https://github.com/openego/ego.io>`_ gives you a python SQL-Alchemy representations of
the oedb and access to it by using the
`oedialect <https://github.com/openego/oedialect>`_, an SQL-Alchemy dialect used by the
OEP. Your API
access / login data will be saved in the folder ``.egoio`` in the file
``config.ini``. The ``config.ini`` is automatically created from user input when it does not exist. It 
holds the following information:

.. code-block:: bash

  [oedb]
  dialect  = oedialect
  username = <username>
  database = oedb
  host     = openenergy-platform.org
  port     = 80
  password = <token>



Use Ding0
=========

Have a look at the :ref:`ding0-examples`.