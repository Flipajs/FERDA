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
echo "Installing PyQt"
apt-get install python-qt4

echo "What is your ubuntu flavour (trusty, wily, xenial, yakkety)?"
read DISTRIBUTION
(echo ""; echo "#graph-tool") >> /etc/apt/sources.list
echo "deb http://downloads.skewed.de/apt/$DISTRIBUTION $DISTRIBUTION universe" >> /etc/apt/sources.list
echo "deb-src http://downloads.skewed.de/apt/$DISTRIBUTION $DISTRIBUTION universe" >> /etc/apt/sources.list
apt-get update
apt-get install python-graph-tool

#upgrade pip
echo "Upgrading pip"
pip install --upgrade pip

pip install --upgrade pip
# to run in quiet mode, add -q to each command
echo "Done"
# echo "Installing networkx via pip"
# pip install networkx
# echo "Done"
echo "Installing scikit-image via pip. This may take a while"
pip install scikit-image
echo "Done"
echo "Installing scikit-learn via pip. This may take a while"
pip install scikit-learn
echo "Installing interval tree"
pip install intervaltree
echo "Done. You are now ready to run FERDA"

exit 0
