# update package index
apt-get update

# install virtual environment
apt-get install virtualenv -y

# create new virtual environment
virtualenv .virtualenvs/dingo --python=python3
source .virtualenvs/dingo/bin/activate

# test developer installation mode
# TODO: get sources without github account
pip3 install -e dingo