#!/bin/bash
source /mnt/md0/shaokai/miniconda3/bin/activate
conda env create -f conda/amadesuGPT-gpu.yml
conda activate amadeusgpt-gpu
# adjust this line according to your cuda version
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install "git+https://github.com/DeepLabCut/DeepLabCut.git@pytorch_dlc#egg=deeplabcut"
pip install pycocotools
pip install -e .[streamlit]