
#!/bin/bash
if [[ `uname -s` == "Darwin" ]]; then
  export OS="MacOSX"
else
  export OS="Linux"
fi
echo "Identified OS as $OS"

# Run in a temp directory
WORKING_DIR=`mktemp -d`
if (( $? )); then
  # Probably no mktemp, generate a time-based directory name
  WORKING_DIR=`date "+/tmp/isp_standard.%s"`
fi
echo "Working in $WORKING_DIR"
mkdir -p "$WORKING_DIR"
cd "$WORKING_DIR"

export ARCH=`uname -m`

# Where to install conda if it's not there already
CONDA_INSTALL_PATH="$HOME/isp_standard/miniconda"

###
# Install Anaconda if not found

echo "Looking for Anaconda installation"

# Anaconda may have been previously installed but not on the existing path, so add the install
# path to ensure we find it in that case
export PATH="$PATH:$CONDA_INSTALL_PATH/bin"
hash -r

CONDA_VER=`conda -V` 2>/dev/null
if (( $? )); then
  echo "
Anaconda Python not found, do you want to install it? [yes|no]
[no] >>> "
  read ans
  ANS=`echo "$ans" | tr '[:upper:]' '[:lower:]'`
  if [[ ($ANS != 'yes') && ($ANS != 'y') ]]; then
    echo "Aborting installation"
    exit 1
  fi

  echo "Downloading Miniconda";
  curl -Ss -o miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-${OS}-${ARCH}.sh

  bash miniconda.sh -b -p $CONDA_INSTALL_PATH
  conda config --set always_yes yes --set changeps1 no
  conda update -q conda

  # Useful for debugging any issues with conda
  conda info -a

  # Update the version
  CONDA_VER=`conda -V` 2>/dev/null
else
  echo "Found Anaconda at" `conda info --root`
fi

# Environment activation command changed in 4.4
if [[ $CONDA_VER > 'conda 4.4' ]]; then
  ACTIVATE_CMD='conda'
else
  ACTIVATE_CMD='source'
fi

###
# Update conda
conda update --all

# Create/update the isp conda environment

conda env list | grep '^isp_standard\s' > /dev/null
if (( $? )); then
  echo "No isp environment found. Creating one."
  conda create -n isp_standard python=3.6.10
else
  echo "Found a isp environment. Trying to update."
  
fi
echo installing requirements 
#conda install -c conda-forge sqlalchemy owslib Cython deprecated obspy pandas cartopy pywavelets dill numba mtspec nitime keras=2.3.1 pyqt pyqtwebengine
conda install -c anaconda hdf5 h5py graphviz pydot
conda install -c conda-forge obspy
conda install -c anaconda keras=2.3.1
conda install tensorflow=2.0.0
conda install -c conda-forge owslib Cython deprecated pandas cartopy pywavelets dill numba mtspec nitime pyqt pyqtwebengine

###
# User instructions

$ACTIVATE_CMD activate isp_standard
BIN=`command -v isp_standard`
if [[ $BIN != '' ]]; then
  echo "
You can launch ISP from the command line by calling:
$BIN
"
fi

# inside your local ISP folder
cd ..
python setup.py build_ext --inplace

# when testing on a ubuntu system libXss.so.1 was missing. This was fixed by installing the libxss1 package
# example: apt install libxss1
