#!/bin/bash
source /Users/shaokaiye/miniforge3/bin/activate
conda env create -f conda/amadesuGPT-gpu.yml
conda activate amadeusgpt-gpu
conda install pytorch cudatoolkit=11.8 -c pytorch
pip install "git+https://github.com/DeepLabCut/DeepLabCut.git@pytorch_dlc#egg=deeplabcut[gui,modelzoo,wandb]"


pip install -e .[streamlit]
