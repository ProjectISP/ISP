#!/bin/bash

# ******************************************************************************
# *   ISP: Integrated Seismic Program                                          *
# *                                                                            *
# *   A Python GUI for earthquake seismology and seismic signal processing     *
# *                                                                            *
# *   Copyright (C) 2023     Roberto Cabieces                                  *
# *   Copyright (C) 2023     Andrés Olivar-Castaño                              *
# *   Copyright (C) 2023     Thiago C. Junqueira                               *
# *   Copyright (C) 2023     Jesús Relinque                                    *
# *                                                                            *
# *   This file is part of ISP.                                                *
# *                                                                            *
# *   ISP is free software: you can redistribute it and/or modify it under the *
# *   terms of the GNU Lesser General Public License (LGPL) as published by    *
# *   the Free Software Foundation, either version 3 of the License, or (at    *
# *   your option) any later version.                                          *
# *                                                                            *
# *   ISP is distributed in the hope that it will be useful for any user or,   *
# *   institution, but WITHOUT ANY WARRANTY; without even the implied warranty *
# *   of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU      *
# *   Lesser General Public License (LGPL) for more details.                   *
# *                                                                            *
# *   You should have received a copy of the GNU LGPL license along with       *
# *   ISP. If not, see <http://www.gnu.org/licenses/>.                         *
# ******************************************************************************

# ******************************************************************************
# * INTEGRATED SEISMIC PROGRAM INSTALLER USING CONDA ENVIRONMENT               *
# ******************************************************************************

ISP_DIR="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )/.."

# Create/update the isp conda environment from file.yml

conda env list | grep '^isp\s' > /dev/null
if (( $? )); then
  echo "No 'isp' environment found. Proceeding to create one."

  # Check operating system
  OS_TYPE=`uname -s`
  echo "Operating System detected: $OS_TYPE"

  if [[ $OS_TYPE == "Darwin" ]]; then
    echo "MacOS detected. Checking processor type..."
    
    # Determine processor type
    CPU_INFO=$(sysctl -n machdep.cpu.brand_string)
    echo "Processor Info: $CPU_INFO"

    if [[ $CPU_INFO == *"Apple"* ]]; then
      echo "Apple Silicon (M1/M2) detected."
      export OS="MacOSX-ARM"
      echo "Using Conda environment file: mac_installer/mac_environment_arm.yml"
      conda env create -f ./mac_installer/mac_environment_arm.yml
    else
      echo "Intel processor detected."
      export OS="MacOSX-Intel"
      echo "Using Conda environment file: mac_installer/mac_environment.yml"
      conda env create -f ./mac_installer/mac_environment.yml
    fi

  elif [[ $OS_TYPE == "Linux" ]]; then
    echo "Linux detected."
    export OS="Linux"
    echo "Using Conda environment file: linux_installer/linux_environment.yml"
    conda env create -f ./linux_installer/linux_environment.yml
  else
    echo "Unsupported operating system: $OS_TYPE"
    exit 1
  fi
else
  echo "'isp' environment already exists. No action taken."
fi

echo "ISP environment created"

# Compile packages
pushd ${ISP_DIR} > /dev/null

# Select installation type
read -p 'Which type of installation would you prefer, conventional or advanced ? ' REPLY

if [[ $REPLY = 'conventional' ]]
then
    source activate isp
    python3 ${ISP_DIR}/setup_user.py build_ext --inplace
elif [[ $REPLY = 'advanced' ]]
then
    source activate isp
    python3 ${ISP_DIR}/setup_developer.py build_ext --inplace
fi
popd > /dev/null

# Creat an alias
read -p "Create alias at .bashrc for ISP?[Y/n] " -r
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]
then
     echo "# ISP Installation" >> ~/.bashrc
     echo "# ISP Installation" >> ~/.bash_profile
     echo "# ISP Installation" >> ~/.zshrc
     echo "# ISP Installation" >> ~/.zprofile

     echo "alias isp=${ISP_DIR}/isp.sh" >> ~/.bashrc
     echo "alias isp=${ISP_DIR}/isp.sh" >> ~/.bash_profile
     echo "alias isp=${ISP_DIR}/isp.sh" >> ~/.zshrc
     echo "alias isp=${ISP_DIR}/isp.sh" >> ~/.zprofile

     echo "# ISP Installation end" >> ~/.bashrc
     echo "# ISP Installation end" >> ~/.bash_profile
     echo "# ISP Installation end" >> ~/.zshrc
     echo "# ISP Installation end" >> ~/.zprofile
     echo "Run ISP by typing isp at terminal"
else
    echo "To run ISP execute isp.sh at ${ISP_DIR}"
fi