<p align="center">
<img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/8555eac6-6af0-4538-bda4-c1a8a2c7bed8/amadeusgpt_logo.png?format=1500w" width="95%">
</p>

🪄 **We turn natural language descriptions of behaviors into machine-executable code.**

- We use large language models (LLMs) to bridge natural language and behavior analysis.
- This work is accepted to **NeuRIPS2023!** Read the paper, [AmadeusGPT: a natural language interface for interactive animal behavioral analysis](https://www.google.com/search?q=amadeusgpt+openreview&sca_esv=590699485&rlz=1C5CHFA_enCH1059CH1060&ei=K1N6ZaHdKvmrur8PosOOkAo&ved=0ahUKEwjhnta83I2DAxX5le4BHaKhA6IQ4dUDCBE&uact=5&oq=amadeusgpt+openreview&gs_lp=Egxnd3Mtd2l6LXNlcnAiFWFtYWRldXNncHQgb3BlbnJldmlldzIHECEYoAEYCjIHECEYoAEYCki2HVDfAliOHHACeACQAQGYAYMDoAHaGaoBCDEuMTEuMS40uAEDyAEA-AEBwgIFECEYqwLCAggQABiABBiiBMICCBAAGIkFGKIE4gMEGAEgQYgGAQ&sclient=gws-wiz-serp#:~:text=AmadeusGPT%3A%20a%20natural,net%20%E2%80%BA%20pdf)
- Like this project? Please consider giving us a star ⭐️!
  
## Install & Run AmadeusGPT🎻

- AmadeusGPT is a Python package hosted on `pypi`. You can create a virtual env (conda, etc, see below*) or Docker and run:
```python
pip install 'amadeusgpt[streamlit]'
```
 - Please note that you need an [openAI API key](https://platform.openai.com/account/api-keys), which you can easily create [here](https://platform.openai.com/account/api-keys).
- If you want the **Streamlit Demo on your computer**, you will also need demo files that are supplied in our repo (see below**), so please git clone the repo and navigate into the `AmadeusGPT` directory. Then in your conda env/terminal run  `pip install '.[streamlit]'`. Then, to launch the Demo App execute in the terminal:
```python
make app
```

## Citation

  If you use ideas or code from this project in your work, please cite us  using the following BibTeX entry. 🙏

 ```
@article{ye2023amadeusGPT,
      title={AmadeusGPT: a natural language interface for interactive animal behavioral analysis}, 
      author={Shaokai Ye and Jessy Lauer and Mu Zhou and Alexander Mathis and Mackenzie Weygandt Mathis},
      journal={Thirty-seventh Conference on Neural Information Processing Systems},
      year={2023},
      url={https://openreview.net/forum?id=9AcG3Tsyoq},
```
- arXiv preprint version **[AmadeusGPT: a natural language interface for interactive animal behavioral analysis](https://arxiv.org/abs/2307.04858)** by [Shaokai Ye](https://github.com/yeshaokai), [Jessy Lauer](https://github.com/jeylau), [Mu Zhou](https://github.com/zhoumu53), [Alexander Mathis](https://github.com/AlexEMG) & [Mackenzie W. Mathis](https://github.com/MMathisLab).

### Install tips
- *Make a new conda env: `conda create --name amadeusGPT python=3.9` then run `conda activate amadeusGPT` or you can also use our supplied conda if you git cloned the repo (navigate into the conda directory): `conda env create -f amadesuGPT.yml` then pip install amadeusGPT once created/launched.
- **Git clone this repo: so please open a terminal, we recommend to download into Documents (so type `cd Documents`) and run `git clone https://github.com/AdaptiveMotorControlLab/AmadeusGPT.git` Then go into the dir (`cd AmadeusGPT`)
- If you want to use SAM, you need to download the weights. Otherwise you will see the following message in the app: `Cannot find SAM checkpoints. Skipping SAM`. Download them and add to "static" directory: wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
  
###  Install trouble shooting:
- If you hit an error during installing on an M1/M2 Macbook with installing HDF5, run `conda install h5py` in your conda env.
- If you launch the app and get an ffmpeg error, `RuntimeError: No ffmpeg exe could be found. Install ffmpeg on your system, or set the IMAGEIO_FFMPEG_EXE environment variable.` try running `conda install ffmpeg`.
- If you have an M1/M2 chip and use CEBRA within AmadeusGPT, and you get this error: `RuntimeError: Device type MPS is not supported for torch.Generator() api` run `pip install --upgrade torch`.

## License 

AmadeusGPT is license under the Apache-2.0 license.
  -  🚨 Please note several key dependencies have their own licensing. Please carefully check the license information for [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) (LGPL-3.0 license), [SAM](https://github.com/facebookresearch/segment-anything) (Apache-2.0 license), [CEBRA](https://github.com/AdaptiveMotorControlLab/CEBRA) (Non-Commercial), etc...


## News
- 🤩 Dec 2023, code released!
- 🔥 Our work was accepted to NeuRIPS2023
- 🧙‍♀️ Open-source code coming in the fall of 2023
- 🔮 arXiv paper and demo released July 2023
- 🪄[Contact us](http://www.mackenziemathislab.org/)
