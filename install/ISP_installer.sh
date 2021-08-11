#!/bin/bash

ISP_DIR="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )/.."

# Create/update the isp conda environment from file.yml

conda env list | grep '^isp\s' > /dev/null
if (( $? )); then
  echo "No isp environment found. Creating one"

if [[ `uname -s` == "Darwin" ]]; then

export OS="MacOSX"  
conda env create -f ./mac_installer/mac_environment.yml

else
export OS="Linux"
conda env create -f ./linux_installer/linux_environment.yml
fi
else
echo "isp environment found"
fi

# Compile packages
pushd ${ISP_DIR} > /dev/null
python ${ISP_DIR}/setup.py build_ext --inplace 
popd > /dev/null

# Creat an alias
read -p "Create alias at .bashrc for ISP?[Y/n] " -r
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]
then
     echo "# ISP Installation" >> ~/.bashrc
     echo "# ISP Installation" >> ~/.bash_profile
     echo "# ISP Installation" >> ~/.zshrc

     echo "alias isp=${ISP_DIR}/isp.sh" >> ~/.bashrc
     echo "alias isp=${ISP_DIR}/isp.sh" >> ~/.bash_profile
     echo "alias isp=${ISP_DIR}/isp.sh" >> ~/.zshrc

     echo "# ISP Installation end" >> ~/.bashrc
     echo "# ISP Installation end" >> ~/.bash_profile
     echo "# ISP Installation end" >> ~/.zshrc
     echo "Run ISP by typing isp at terminal"
else
    echo "To run ISP execute isp.sh at ${ISP_DIR}"
fi