#!/bin/bash

# update package index
apt-get update -y

# non-required dependencies
#apt-get install -y liblapack-dev
# apt-get install -y libatlas-base-dev
# apt-get install -y libfreetype6-dev
# apt-get install -y pkg-config
# apt-get install -y libpq-dev
# apt-get install -y build-essential
# apt-get install -y g++

# definitely needed packages
apt-get install -y libssl-dev
apt-get install -y python3-dev
apt-get install -y libgeos-dev
apt-get install -y libffi-dev
apt-get install -y gcc