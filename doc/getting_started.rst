Getting started
~~~~~~~~~~~~~~~

.. _installation:

Installation on Linux
=====================

.. note::
    Installation is only tested on (Debian-like) Linux OS.

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
    virtualenv -p python3.8 ~/virtualenvs/ding0

Jump into (aka. activate) this python distribution by

.. code-block:: bash

    # Adjust path to your specific needs
    source ~/virtualenvs/ding0/bin/activate

From that, the latest release of ding0 is installed by

.. code-block:: bash

    pip install ding0


Pip allows to install a developer version of a package that uses currently
checked out code. A developer mode installation is achieved by cloning the
repository to an arbitrary path (e.g. `~/repos/` in the following example)
and installing manually via pip:

.. code-block:: bash

    mkdir ~/repos/
    cd ~/repos/
    git clone git@github.com:openego/ding0.git
    pip install -e ~/repos/ding0/
    

Installation on Windows
-----------------------
To install Ding0 in windows, it is currently recommended to use
`Anaconda <https://www.anaconda.com/distribution/>`_ or
`Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_
and create an environment with the ding0_env.yml file provided.

.. note::
    Normally both miniconda and Anaconda are packaged with the Anaconda
    Prompt to be used in Windows. Within typical installations, this
    restricts the use of the conda command to only within this prompt.
    Depending on your convenience, it may be a wise choice to add
    the conda command to your path during the installation by checking
    the appropriate checkbox. This would allow conda to be used
    from anywhere in the operating system except for PowerShell

.. note::
    Conda and Powershell don't seem to be working well together at
    the moment. There seems to be an issue with Powershell spawning
    a new command prompt for the execution of every command.
    This makes the environment activate in a different prompt
    from the one you may be working with after activation.
    This may eventually get fixed later on but for now,
    we would recommend using only the standard cmd.exe on windows.

To create a ding0 environment using the yaml file in conda,
use the command:

.. code-block:: bash

    conda env create -f ding0_env.yml

By default this environment will be called ding0_env. If you would
like to use a custom name for your environment use the following variant
of the command:

.. code-block:: bash

    conda env create -n custom_env_name -f ding0_env.yml

An to activate this environment, from any folder in the operating system,
use the command:

.. code-block:: bash

    conda activate ding0_env

Once the environment is activated, you have two options to install ding0.
Either install it from the local repository with the commands:

.. code-block:: bash

    conda activate ding0_env
    pip install -U -e \path\to\ding0\

Or install it from the pypi repository with the command:

.. code-block:: bash

    conda activate ding0_env
    pip install ding0



after this, it is possible to install ding0 directly from pip within the
conda enviornment

.. code-block:: bash

    conda activate ding0_env

Use Ding0
=========

Have a look at the :ref:`ding0-examples`.