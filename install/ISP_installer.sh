#!/bin/bash

# Create/update the isp conda environment

conda env list | grep '^isp\s' > /dev/null
if (( $? )); then
  echo "No isp environment found. Creating one"
  conda create -n isp python=3.6.10
else
  echo "Found a isp environment"  
fi

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
conda install -c conda-forge pyqt pyqtwebengine

# second option with pip
#pip install nitime PyQt5 PyQtWebEngine

# Compile packages
cd ..
python setup.py build_ext --inplace

echo run ISP typing python start_isp.py