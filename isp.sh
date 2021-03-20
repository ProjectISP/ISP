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

pushd ${ISP_DIR} > /dev/null
python start_isp.py
popd > /dev/null
