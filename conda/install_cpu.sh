#!/bin/bash
source /Users/shaokaiye/miniforge3/bin/activate
conda env create -f conda/amadesuGPT.yml
conda activate amadeusgpt
conda install pytorch torchvision cpuonly -c pytorch
pip install "git+https://github.com/DeepLabCut/DeepLabCut.git@pytorch_dlc#egg=deeplabcut"
pip install pycocotools
pip install -e .[streamlit]
# install the python kernel
python -m ipykernel install --user --name amadeusgpt --display-name "amadeusgpt"
