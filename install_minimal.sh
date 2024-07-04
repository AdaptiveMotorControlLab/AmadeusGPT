#!/bin/bash
# change this to your own miniconda / miniforge path
source /Users/shaokaiye/miniforge3/bin/activate
conda env create -f conda/amadesuGPT-minimal.yml
conda activate amadeusgpt-minimal
pip install pycocotools
pip install -e .[streamlit]
