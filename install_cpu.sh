#!/bin/bash
source /Users/shaokaiye/miniforge3/bin/activate
conda env create -f conda/amadesuGPT-cpu.yml
conda activate amadeusgpt-cpu
conda install pytorch torchvision cpuonly -c pytorch
pip install "git+https://github.com/DeepLabCut/DeepLabCut.git@pytorch_dlc#egg=deeplabcut"
pip install pycocotools
pip install -e .[streamlit]