#!/bin/bash
# Author: Dita Hollmannova

# check if user is root
if [ "$(id -u)" != "0" ]; then
  echo "Sorry, you have to be root to run this script!"
  exit 1
fi

echo "Downloading packages data"
apt-get update

echo "Installing packges"
apt-get install python-pip python-opencv python-scipy libpython-stdlib libfreetype6-dev -y

# to run in quiet mode, add -q to each command
echo "Done"
echo "Installing networkx via pip"
pip install networkx
echo "Done"
echo "Installing scikit-image via pip. This may take a while"
pip install scikit-image
echo "Done"
echo "Installing scikit-learn via pip. This may take a while"
pip install scikit-learn
echo "Done. You are now ready to run FERDA"
exit 0