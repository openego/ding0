#!/bin/bash

### Script require sudo (root) privileges ###

# This scripts install docker to test installation of ding0 on a fresh OS
# It is adapted to needs of SolydX 8 respectively Debian 8 (Jessie). Level of adaptation is unknown ;-)
#
# Information taken from https://docs.docker.com/engine/installation/linux/debian/#install-using-the-repository
# Look there for further information

# remove old version
sudo apt-get remove docker docker-engine -y

# install prerequisit packages to allow docker installation from HTTPS sources
sudo apt-get install \
     apt-transport-https \
     ca-certificates \
     curl \
     gnupg2 \
     software-properties-common -y

# Add GPG key
curl -fsSL https://download.docker.com/linux/debian/gpg | sudo apt-key add -

# Verify that the key ID is 9DC8 5822 9FC7 DD38 854A E2D8 8D81 803C 0EBF CD88
sudo apt-key fingerprint 0EBFCD88

# Add repository
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/debian jessie stable"

# Update package index and install docker (community edition)
sudo apt-get update
sudo apt-get install docker-ce -y

# Verify docker is working
sudo docker run hello-world