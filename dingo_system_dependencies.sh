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
