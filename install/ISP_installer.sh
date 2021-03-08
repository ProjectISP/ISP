#!/bin/bash

ISP_DIR="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )/.."

# Create/update the isp conda environment

conda env list | grep '^isp\s' > /dev/null
if (( $? )); then
  echo "No isp environment found. Creating one"
  conda create -n isp python=3.6.10
else
  echo "Found a isp environment"  
fi

source $(conda info --base)/etc/profile.d/conda.sh

# Activate environment
if [[ `uname -s` == "Darwin" ]]; then
export OS="MacOSX"  
source activate isp
else
export OS="Linux"
conda activate isp
fi
echo "Identified OS as $OS"
echo "Installing Dependencies"
conda install -c anaconda hdf5 h5py graphviz pydot
conda install -c conda-forge obspy
conda install -c anaconda keras=2.3.1
conda install tensorflow=2.0.0
conda install -c conda-forge owslib Cython deprecated pandas cartopy pywavelets dill numba mtspec nitime pillow

# Install Qt
#conda install -c conda-forge pyqt pyqtwebengine

# second option with pip
pip install PyQt5 PyQtWebEngine

# Compile packages
pushd ${ISP_DIR} > /dev/null
python ${ISP_DIR}/setup.py build_ext --inplace 
popd > /dev/null

read -p "Create alias at .bashrc for ISP?[Y/n] " -r
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "# ISP Installation" >> ~/.bashrc
    echo "alias isp=${ISP_DIR}/isp.sh" >> ~/.bashrc
    echo "# ISP Installation end" >> ~/.bashrc
    echo "Run ISP by typing isp at terminal"
else
    echo "To run ISP execute isp.sh at ${ISP_DIR}"
fi