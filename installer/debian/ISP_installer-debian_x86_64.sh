#!/bin/sh

# ******************************************************************************
# *   ISP: Integrated Seismic Program                                          *
# *                                                                            *
# *   A Python GUI for earthquake seismology and seismic signal processing     *
# *                                                                            *
# *   Copyright (C) 2022     Roberto Cabieces                                  *
# *   Copyright (C) 2022     Thiago C. Junqueira                               *
# *   Copyright (C) 2022     Jesús Relinque                                    *
# *   Copyright (C) 2022     Ángel Vera                                        *
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
# * Installer version: 0.1                                                     *
# * Installer date: 15/6/2021                                                  *
# ******************************************************************************

# Get the current directory.
ISP_DIR="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )/../.."
ISP_CURRENT="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )/"

# Get the ISP current version
line=$(head -n 1 $ISP_DIR/VERSION_DETAILS.md)
stringarray=($line)
version=${stringarray[3]}

# Default install directory and environment.
ISP_ROOT="${HOME}/ISP-$version"
ISP_ENV="isp-$version"

# ISP internal installer using a Conda environment.
clear
printf "\n=====================================================\n"
printf "==    INTEGRATED SEISMIC PROGRAM INSTALLER         ==\n"
printf "=====================================================\n"
printf "=     CONDA ENVIRONMENT VERSION FOR DEBIAN          =\n"
printf "=====================================================\n"
printf "= Installer version: 0.1                            =\n"
printf "= Installer date: 21/6/2021                         =\n"
printf "=====================================================\n\n"
printf "> Welcome to the ISP easy installer script for Debian! ^^ \n"
printf "> ISP (version $version) will be installed using a Conda environment.\n\n"
read -p "> Press enter to start the installation..."
printf "\n\n"

# Check for conda.
printf "> Checking for < conda > program... : "
if ! [ -x "$(command -v conda)" ]
then
    printf "FAIL \n"
    printf "    ERROR: < conda > must be installed. See: \n"
    printf "    https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html\n"
    printf "    Exiting the installer!\n\n"
    exit 1
fi
printf "OK\n\n"

# Setting ISP installation directory.
printf "> ISP will be installed into this location:\n"
printf "%s\n" "$ISP_ROOT"
printf "\n"
printf "  - Press ENTER to confirm the location.\n"
printf "  - Press CTRL-C to abort the installation.\n"
printf "  - Or specify a different location below.\n"
printf "\n"
printf "[%s] >>> " "$ISP_ROOT"
read -r selected_dir
if [ "$selected_dir" != "" ]
then
    case "$selected_dir" in
        *\ *)
            printf "ERROR: Cannot install into directories with spaces\n" >&2
            exit 1
            ;;
        *)
            eval ISP_ROOT="$selected_dir"
            ;;
    esac
fi

# Creates the installation directory.
printf "\n\n-> Checking the installation directory...\n"
if [[ -d "$ISP_ROOT" ]]
then
    printf "The directory < $ISP_ROOT > exists on your filesystem.\n"
    printf "  - Type R to remove the directory. \n"
    printf "  - Press ENTER to keep the directory (not recommended). \n"
    printf "  - Press CTRL-C to abort the installation.\n"
    printf "\n"
    printf "[%s] >>> "
    read -r option
    if [ "$option" == "R" ]
    then
        rm -r $ISP_ROOT
        printf "\n\n> Old directory removed. Creating the new directory. \n"
        if ! mkdir -p "$ISP_ROOT"
        then
            printf "FAIL \n"
            printf "    ERROR: Could not create directory: '%s'\n" "$CWPROOT" >&2
            printf "    Exiting the installer!\n\n"
            exit 1
        fi
        printf "OK\n\n"
    fi
else
    printf "\n\n> Creating the installation directory... : "
    if ! mkdir -p "$ISP_ROOT"
    then
        printf "FAIL \n"
        printf "    ERROR: Could not create directory: '%s'\n" "$CWPROOT" >&2
        printf "    Exiting the installer!\n\n"
        exit 1
    fi
    printf "OK\n\n"
fi

# Setting ISP_ENV
printf "\n> The name of the Conda environment will be: < $ISP_ENV >\n"
printf "\n"
printf "  - Press ENTER to confirm the name.\n"
printf "  - Press CTRL-C to abort the installation.\n"
printf "  - Or specify a different name below.\n"
printf "\n"
printf "[%s] >>> " "$ISP_ENV"
read -r selected_env
if [ "$selected_env" != "" ]
then
    case "$selected_env" in
        *\ *)
            printf "ERROR: Cannot use names with spaces\n" >&2
            exit 1
            ;;
        *)
            eval ISP_ENV="$selected_env"
            ;;
    esac
fi

# Check the environment.
printf "\n> Activating < base > environment...\n"
eval "$(conda shell.bash hook)"
conda activate base > /dev/null 2>&1
printf "\n\n-> Checking the ISP environment...\n"
conda env list | grep "^$ISP_ENV\s" > /dev/null
if (($?))
then
    printf "No ISP environment found. Creating one with name: < $ISP_ENV >\n"
    conda create -y -n $ISP_ENV python=3.8
else
    printf "ISP environment found with name: < $ISP_ENV >\n"
    printf "  - Type R to remove the environment. \n"
    printf "  - Press ENTER to keep the environment (not recommended). \n"
    printf "  - Press CTRL-C to abort the installation.\n"
    printf "\n"
    printf "[%s] >>> "
    read -r option
    if [ "$option" == "R" ]
    then
        conda env remove -n $ISP_ENV
        printf "> Old environment removed. Creating one with name: < $ISP_ENV >\n"
        conda create -y -n $ISP_ENV python=3.8
    fi
fi

# Install requirements.
printf "\n> Installing some dependencies... \n"
sudo apt-get install wget
sudo apt-get install gcc
sudo apt-get install gfortran
sudo apt-get install build-essential
sudo apt-get install zlib1g
sudo apt-get install zlib1g-dev
sudo apt-get install sqlite3
sudo apt-get install checkinstall
sudo apt-get install libxcb-xinerama0 
sudo apt-get install libx11-xcb1
sudo apt-get install python3-dev 
sudo apt-get install libproj-dev
sudo apt-get install proj-data
sudo apt-get install proj-bin
sudo apt-get install libgeos-dev
sudo apt-get install libgfortran4
	
# Activate the Conda environment.
printf "\n> Activating ISP environment < $ISP_ENV > using Conda...\n"
eval "$(conda shell.bash hook)"
conda activate $ISP_ENV > /dev/null 2>&1

# Install Conda packages.
printf "\n> Installing Conda packages...\n"
conda install -y -c conda-forge mtspec 
conda install -y -c conda-forge cartopy=0.18.0 

# Install Python requirements.
printf "\n> Installing Python requirements...\n"
yes | pip install -r "$ISP_CURRENT"requirements.txt

# Install pther Conda packages.
printf "\n> Installing more Conda packages...\n"
conda install -y tensorflow
conda install -y tensorflow-gpu

# TODO: VERY IMPORTANT.
# WE MUST CHECK IF WE CAN USE CUDA WITH OUR GRAPHIC CARD.
graphics="$(lspci | grep VGA)"
if [[ "$graphics" == *"NVIDIA"* ]]
then
  printf "\n> You have a NVIDIA graphics card. The cuda package will be installed"
  printf "\n  in Conda, but you should check that the card is compatible and that" 
  printf "\n  all related dependencies are installed. Next, you will see all your"
  printf "\n  graphics card information. Check for CUDA Version > 11.0 \n\n"
  printf "  - Press ENTER to display the information. \n"
  printf ">>> "
  read -r trash
  sudo nvidia-smi
  printf "\n\n - Press ENTER to continue the installation. \n"
  read -r trash
  conda install -y cudatoolkit
fi

# Compile packages
printf "\n> Compiling third party programs...\n"
pushd ${ISP_DIR} > /dev/null
python ${ISP_DIR}/installer/third_party/setup.py ISP_ThirdParty --build-lib=$ISP_DIR/isp
popd > /dev/null

# Copy the project to the deploy folder.
printf "\n> Deploying ISP in < $ISP_ROOT > ...\n"
cp -r "$ISP_DIR"/isp "$ISP_ROOT"
cp -r "$ISP_DIR"/test "$ISP_ROOT"
cp "$ISP_DIR"/isp.py "$ISP_ROOT"/isp-"$version".py
cp "$ISP_DIR"/start_isp.sh "$ISP_ROOT"
sudo chmod u+x "$ISP_ROOT"/start_isp.sh

# Delete previous environment variables and alias.
printf "\n> Deleting precious environment variables...\n"
sed -i '/# >>> ISP >>>/,/# <<< ISP <<</d' ~/.bashrc
sed -i '/# >>> ISP >>>/,/# <<< ISP <<</d' ~/.bash_profile
sed -i '/# >>> ISP >>>/,/# <<< ISP <<</d' ~/.zshrc

# Create an alias option
printf "\nCreate alias at < .bashrc  > for ISP? [y/n]\n" 
printf ">>> "
read -r option

# Export neccesary environment variables.
printf "\n> Exporting neccesary environment variables...\n"
echo "# >>> ISP >>>" >> ~/.bashrc
echo "# >>> ISP >>>" >> ~/.bash_profile
echo "# >>> ISP >>>" >> ~/.zshrc
echo "# Integrated Seismic Program variables and alias." >> ~/.bashrc
echo "# Integrated Seismic Program variables and alias." >> ~/.bash_profile
echo "# Integrated Seismic Program variables and alias." >> ~/.zshrc
echo "# Never remove these commented marks!" >> ~/.bashrc
echo "# Never remove these commented marks!" >> ~/.bash_profile
echo "# Never remove these commented marks!" >> ~/.zshrc
echo "export ISP_ENV=${ISP_ENV}" >> ~/.bashrc
echo "export ISP_ENV=${ISP_ENV}" >> ~/.bash_profile
echo "export ISP_ENV=${ISP_ENV}" >> ~/.zshrc
echo "export ISP_VER=${version}" >> ~/.bashrc
echo "export ISP_VER=${version}" >> ~/.bash_profile
echo "export ISP_VER=${version}" >> ~/.zshrc
echo "export ISP_ROOT=${ISP_ROOT}" >> ~/.bashrc
echo "export ISP_ROOT=${ISP_ROOT}" >> ~/.bash_profile
echo "export ISP_ROOT=${ISP_ROOT}" >> ~/.zshrc
# Create an alias
if [[ $option =~ ^[Yy]$ ]]
then
     printf "\n> Creating an alias...\n"

     echo "alias isp=${ISP_ROOT}/start_isp.sh" >> ~/.bashrc
     echo "alias isp=${ISP_ROOT}/start_isp.sh" >> ~/.bash_profile
     echo "alias isp=${ISP_ROOT}/start_isp.sh" >> ~/.zshrc
fi
# End marks.
echo "# <<< ISP <<<" >> ~/.bashrc
echo "# <<< ISP <<<" >> ~/.bash_profile
echo "# <<< ISP <<<" >> ~/.zshrc

# All done. 
printf "\n> ISP installation finished! \n"
printf "\n> You can now delete the folder < $(realpath "${ISP_DIR}") > \n"
printf "\n> For launch ISP follow these steps: \n"
printf "    -> Open a new terminal or refresh this one using < exec bash >.\n"
printf "    -> Go to the folder < $ISP_ROOT >\n"
printf "    -> Type in the terminal < ./start_isp.sh > and and get to work now! \n"
printf "\n> If you created an alias, you can also: \n"
printf "    -> Open a new terminal or refresh this one using < exec bash >.\n"
printf "    -> Type in the terminal < isp > directly. \n\n"

# Refresh bashrc.
source ~/.bashrc
source ~/.bash_profile
source ~/.zshrc
exec bash

# Exit
exit 0

# ******************************************************************************


