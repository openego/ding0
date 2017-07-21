./dingo_system_dependencies.sh

# update package index
apt-get update

# install virtual environment
apt-get install virtualenv -y

# create new virtual environment
virtualenv .virtualenvs/dingo --python=python3
source .virtualenvs/dingo/bin/activate

# test developer installation mode
# TODO: get sources without github account. Currently it assumes dingo source
# TODO: code is located in ./dingo
pip3 install -e dingo

#ls
#ls dingo
#ls dingo/examples

#python3 dingo/examples/example_single_grid_district.py

python3 -c "from dingo.core import NetworkDingo; nd = NetworkDingo(name='network'); print(nd)"
