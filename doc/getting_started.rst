Getting started
~~~~~~~~~~~~~~~

.. _installation:

Installation
============

.. note::
    The installation is tested on Ubuntu Ubuntu 16.04 (xenial) and `Windows Server
    1803 <https://docs.travis-ci.com/user/reference/windows/#windows-version>`_
    through running tests on `Travis CI <https://travis-ci.org/openego/ding0>`_.

Ding0 is published through PyPi package management and, thus, installable from
sources of pip3.
We recommend installing Ding0 (and in general third-party) python packages in a
virtual environment, encapsulated from the system python distribution.
Conda is the preferred way, but virtualenv works as well. In particular for
Windows users, we recommend to use conda.

Using conda
-----------

Once `conda is installed
<https://docs.conda.io/projects/conda/en/latest/user-guide/install/>`_, use the
`environment file <https://github.com/openego/ding0/blob/dev/ding0_env.yml>`_
to install required packages.

.. code-block:: bash

    conda create -n ding0 -p python3 --file ding0_env.yml

Activate the environment

.. code-block:: bash

    conda activate ding0

and install ding0 in it's latest version

.. code-block:: python

    pip install ding0

or install the developer version by cloning the ding0 repository and
installing it in developer mode.

.. code-block:: python

    pip install -e <path-to-local-ding0-repo>


Using virtualenv
----------------

First, you might need to install the tool
`virtualenv <https://virtualenv.pypa.io/en/stable/>`_ by

.. code-block:: bash

    sudo apt install virtualenv

Subsequently, you can use it to create a virtual environment

.. code-block:: bash

    # Adjust path to your specific needs
    virtualenv -p python3 ~/.virtualenvs/ding0

Activate the environment

.. code-block:: bash

    # Adjust path to your specific needs
    source ~/.virtualenvs/ding0/bin/activate

and install latest version of ding0

.. code-block:: python

    pip3 install ding0

or install the developer version by cloning the ding0 repository and
installing it in developer mode.

.. code-block:: python

    pip3 install -e <path-to-local-ding0-repo>


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


Troubleshooting
===============

If you have trouble with versions of installed python packages, see
`the package list <https://github.com/openego/ding0/wiki/Installed-packages>`_
of the last release.