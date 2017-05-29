# Test installation of latest dingo version in fresh OS installation
#
# To obtain a fresh installation environment we create a virtualized OS by the help of docker
# If you haven't installed docker yet, check out install_docker.sh
#
# Note: in this script all docker commands are executed as root (via sudo) as this is required on debian (solydX) systems

# make dingo install script and dependency install script executable
chmod +x install_dingo.sh
chmod +x dingo_system_dependencies.sh

# create fresh container with ubuntu OS and run dingo install script within it
# sudo docker run -t -i ubuntu install_dingo.sh
sudo docker run --rm -v $(pwd)/:/dingo/ -v $(pwd)/install_dingo.sh:/install_dingo.sh -v $(pwd)/dingo_system_dependencies.sh:/dingo_system_dependencies.sh ubuntu bash /install_dingo.sh