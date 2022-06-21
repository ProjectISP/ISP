#!/bin/bash

ISP_DIR="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"


# Activate the Conda environment.
printf "\n> Activating ISP environment < $ISP_ENV > using Conda...\n"
eval "$(conda shell.bash hook)"
conda activate $ISP_ENV

printf "\n> Launching ISP (version < $ISP_VER >)...\n"
pushd ${ISP_DIR} > /dev/null
python isp-"$ISP_VER".py
popd > /dev/null
