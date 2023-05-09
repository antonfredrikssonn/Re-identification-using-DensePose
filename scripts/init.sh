#!/bin/bash
# Source python
# eval "$(pyenv init -)"

# Install dependencies for Densepose and resnet
# PIP_CONSTRAINT=/ReID_Using_DensePose/scripts/constraints.txt
# export PIP_CONSTRAINT
python3 -m pip install pip
python3 -m pip install \
  -r /ReID_Using_DensePose/scripts/requirements.in

# python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
python3 -m pip install git+https://github.com/facebookresearch/detectron2@main#subdirectory=projects/DensePose

# Step into working directory
# cd /kpi_workspace

# Display prompt at startup
echo ""
echo "Welcome to the DensePose re-ID docker image"

/bin/bash