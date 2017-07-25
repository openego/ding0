# Test installation of latest ding0 version in fresh OS installation
#
# To obtain a fresh installation environment we create a virtualized OS by the help of docker
# If you haven't installed docker yet, check out install_docker.sh
#
# Note: in this script all docker commands are executed as root (via sudo) as this is required on debian (solydX) systems

# make ding0 install script and dependency install script executable
chmod +x install_ding0.sh
chmod +x ding0_system_dependencies.sh

# create fresh container with ubuntu OS and run ding0_system_dependencies.sh and install_ding0.sh script within it
sudo docker run --rm -v $(pwd)/:/ding0/ \
    -v $(pwd)/install_ding0.sh:/install_ding0.sh \
    -v $(pwd)/ding0_system_dependencies.sh:/ding0_system_dependencies.sh ubuntu bash /install_ding0.sh

#sudo docker run --rm -v $(pwd)/:/ding0/ \
#    -v $(pwd)/install_ding0.sh:/install_ding0.sh \
#    -v $(pwd)/ding0_system_dependencies.sh:/ding0_system_dependencies.sh debian bash /install_ding0.sh