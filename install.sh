#!/bin/bash

source /Users/shaokaiye/miniforge3/bin/activate

conda env create -f conda/amadesuGPT.yml

conda activate amadeusgpt

pip install -e . 'streamlit' 
