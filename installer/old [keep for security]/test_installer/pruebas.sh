#!/bin/sh


# Check for root privileges. 
eval "$(conda shell.bash hook)"
conda activate isp
conda install pillow << -y
