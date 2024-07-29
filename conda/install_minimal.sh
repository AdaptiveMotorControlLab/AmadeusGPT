#!/bin/bash
# change this to your own miniconda / miniforge path
source /Users/shaokaiye/miniforge3/bin/activate
conda env create -f conda/amadesuGPT.yml
conda activate amadeusgpt
pip install pycocotools
pip install -e .[streamlit]
python -m ipykernel install --user --name amadeusgpt --display-name "amadeusgpt"