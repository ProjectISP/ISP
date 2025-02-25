#!/bin/bash

ISP_DIR="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh

if [[ `uname -s` == "Darwin" ]]; then
    export OS="MacOSX"
    source activate isp
else
    export OS="Linux"
    conda activate isp
fi

# Detectar si estamos en una sesión Wayland
if [[ "$XDG_SESSION_TYPE" == "wayland" ]]; then
    export QT_QPA_PLATFORM=wayland
fi

pushd ${ISP_DIR} > /dev/null
python start_isp.py
popd > /dev/null
