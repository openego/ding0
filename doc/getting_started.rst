Getting started
~~~~~~~~~~~~~~~

.. warning::
    Note, Dingo relies on data provided by the
    `OEDB <http://oep.iks.cs.ovgu.de/dataedit/>`_. Currently, only
    members of the openego project team have access to this database. Public
    access (SQL queries wrapped by HTML) to the `OEDB` will be provided soon


Installation
============

.. note::
    Installation is only tested on (debian like) linux OS.

Dingo is provided though PyPi package management and, thus, installable from
sources of pip3.
The package relies on a bunch of dependencies. These are defined by package
meta data of Dingo and installed via during installation of Dingo. Nevertheless,
you may need to have some specific package system packages installed for a
successful installation of Dingo and its dependencies.

The script `dingo_system_dependencies.sh` installs required system package
dependencies.

.. code-block:: bash

    cd <your-dingo-install-path>

    chmod +x dingo_system_dependencies.sh

    sudo ./dingo_system_dependencies.sh


We recommend install Dingo (and in general third-party) python packages in a
virtual enviroment, encapsulated from system python distribution.
This is optional. If you want to follow our suggestion, install the tool
`virtualenv <https://virtualenv.pypa.io/en/stable/>`_ by

.. code-block:: bash

    sudo apt-get install virtualenv # since Ubuntu 16.04

Afterwards `virtualenv` allows to create multiple parallel python distributions.
Create a new one for Dingo by

.. code-block:: bash

    # Adjust path to your specific needs
    virtualenv ~/.virtualenvs/dingo

Jump into (aka. activate) this python distribution by

.. code-block:: bash

    # Adjust path to your specific needs
    source ~/.virtualenvs/dingo/bin/activate

Now, your shell executed python command by this specific python distribution.

From that, the latest release of Dingo is installed by

.. code-block:: python

    pip3 install dingo


Pip allows to install a developer version of a package that uses currently
checked code of the repository. A developer mode installation is achieved by

.. code-block:: python

    pip3 install -e dingo
    

Use Dingo
=========

We set up some examples about how to use Dingo that are explained in
:ref:`dingo-examples`.