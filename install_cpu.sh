#!/bin/bash
source /Users/shaokaiye/miniforge3/bin/activate
conda env create -f conda/amadesuGPT-cpu.yml
conda activate amadeusgpt-cpu
conda install pytorch cpuonly -c pytorch

pip install -e .[streamlit]
