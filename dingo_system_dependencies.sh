#!/bin/bash

# update package index
apt-get update -y

# required packages
apt-get install -y libssl-dev
apt-get install -y python3-dev
apt-get install -y libgeos-dev
apt-get install -y libffi-dev
apt-get install -y gcc
apt-get install -y python3-tk

# addionatly required on Solydxk 8
# TODO:test this with particular docker

# apt-get install -y libfreetype6-dev
# apt-get install -y pkg-config
# apt-get install -y libpng3
# apt-get install -y libpq-dev
# apt-get install -y liblapack-dev
# apt-get install -y libatlas-base-dev

# apt-get install -y gfortran
