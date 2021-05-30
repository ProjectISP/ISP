#!/bin/bash

ISP_DIR="/opt/isp"
VENV_NAME="venv"
ISP_VENV_PATH="$ISP_DIR/$VENV_NAME"
ISP_VENV_PIP="$ISP_DIR/bin"

# Not sure if I need to know the OS
if [[ `uname -s` == 'Darwin' ]]; then
  export OS="MacOSX"
elif [[ `uname -s` == 'Linux' ]]; then
  export OS="Linux"
else
  echo "Unknown OS"
fi
echo "Identified OS as $OS"


if ! python3 -c 'import sys; assert sys.version_info >= (3,6)' >/dev/null 2>&1 ; then
  echo "Abort with error."
  echo "Error: Your Python version is not comptible. Please install python 3.6 or latter versions."
  exit 1
fi

# isp () { command  "${ISP_VENV_PATH}/bin/python" "${ISP_DIR}/start_isp.py"; }

function setup_isp {
  { # try
  
    # This command may fail if you dont have python3-venv or python3-pip installed. 
    # User must follow instructions of the error and install by themselfs.  
    python3 -m venv ${ISP_VENV_PATH}
    source  "${ISP_VENV_PATH}/bin/activate"
    echo "Updating pip and setuptools"
    pip install pip==20.2.4  # force version 20.2.4 of pip...version 21.x has some bugs
    pip install setuptools --upgrade
    pip install numpy
    # echo $PWD
    pip install -r requirements.txt 
    python setup.py build_ext --inplace
#if [[ $BIN != '' ]]; then
#  echo "
#You can launch ISP from the command line by calling:
#$BIN

  } || { # catch
    exit 1
  }
}

function copy_isp_source {

   { # try
    echo "Creating ${ISP_DIR}"
    sudo mkdir ${ISP_DIR}
    echo "Creating ${ISP_VENV_PATH}"
    sudo mkdir ${ISP_VENV_PATH}
    # Assumes install.sh is within the isp packege...if not we should pull from git.
    cd ..
    sudo cp -r isp "${ISP_DIR}/isp"
    sudo cp *.py *.tar *.tgz *.gz ${ISP_DIR}
    sudo cp requirements.txt ${ISP_DIR}
    # set 777 permisson to avoid problems.
    sudo chmod 777 ${ISP_DIR}
    sudo chmod 777 "${ISP_DIR}/isp"

  } || { # catch
    echo "Abort with error."
    exit 1
  }

}

if ! [[ -d "$ISP_DIR" ]]; then
  # Take action if $DIR doesn't exists. #
  copy_isp_source
else
  echo "ISP dir ${ISP_DIR} already exists...removing"
  sudo rm -r ${ISP_DIR}
  copy_isp_source
fi

echo "Copy source code to ${ISP_DIR}"
setup_isp
echo "Installation Completed."

# TODO add isp to environmental paramter so the user can lauch it typing isp only.
