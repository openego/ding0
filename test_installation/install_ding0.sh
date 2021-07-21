# This script must be executed in the directory where ding0 setup.py lives

./ding0_system_dependencies.sh

# update package index
apt-get update

# install virtual environment
apt-get install virtualenv -y

# create new virtual environment
virtualenv .virtualenvs/ding0 --python=python3
source .virtualenvs/ding0/bin/activate

# test developer installation mode
# TODO: get sources without github account. Currently it assumes ding0 source
# TODO: code is located in ding0 (in docker)
pip3 install ding0

# Test if installation worked correctly
#python3 ding0/examples/example_single_grid_district.py

python3 -c "from ding0.core import NetworkDing0; nd = NetworkDing0(name='network'); print('\\nDing0 was successfully installed!\\n\\n')"