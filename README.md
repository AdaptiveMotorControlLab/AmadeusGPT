<div align="center">
  
<p align="center">
<img src="https://images.squarespace-cdn.com/content/v1/57f6d51c9f74566f55ecf271/8555eac6-6af0-4538-bda4-c1a8a2c7bed8/amadeusgpt_logo.png?format=1500w" width="95%">
</p>

[![Downloads](https://pepy.tech/badge/amadeusgpt)](https://pepy.tech/project/amadeusgpt)
[![Downloads](https://pepy.tech/badge/amadeusgpt/month)](https://pepy.tech/project/amadeusgpt)
[![PyPI version](https://badge.fury.io/py/amadeusgpt.svg)](https://badge.fury.io/py/amadeusgpt)
[![GitHub stars](https://img.shields.io/github/stars/AdaptiveMotorControlLab/AmadeusGPT.svg?style=social&label=Star)](https://github.com/AdaptiveMotorControlLab/AmadeusGPT)

## 🪄  We turn natural language descriptions of behaviors into machine-executable code.

[🛠️ Installation](https://github.com/AdaptiveMotorControlLab/AmadeusGPT?tab=readme-ov-file#install--run-amadeusgpt) |
[🌎 Home Page](http://www.mackenziemathislab.org/amadeusgpt) |
[🚀 Demos & Docs](https://adaptivemotorcontrollab.github.io/AmadeusGPT/README.html)
[🚨 News](https://github.com/AdaptiveMotorControlLab/AmadeusGPT?tab=readme-ov-file#news) |
[🪲 Reporting Issues](https://github.com/AdaptiveMotorControlLab/AmadeusGPT/issues) |
[💬 Discussions!](https://github.com/AdaptiveMotorControlLab/AmadeusGPT/discussions)

</div>

- We use large language models (LLMs) to bridge natural language and behavior analysis.
- This work is published at **NeurIPS2023!** Read the paper, [AmadeusGPT: a natural language interface for interactive animal behavioral analysis]([https://www.google.com/search?q=amadeusgpt+openreview&sca_esv=590699485&rlz=1C5CHFA_enCH1059CH1060&ei=K1N6ZaHdKvmrur8PosOOkAo&ved=0ahUKEwjhnta83I2DAxX5le4BHaKhA6IQ4dUDCBE&uact=5&oq=amadeusgpt+openreview&gs_lp=Egxnd3Mtd2l6LXNlcnAiFWFtYWRldXNncHQgb3BlbnJldmlldzIHECEYoAEYCjIHECEYoAEYCki2HVDfAliOHHACeACQAQGYAYMDoAHaGaoBCDEuMTEuMS40uAEDyAEA-AEBwgIFECEYqwLCAggQABiABBiiBMICCBAAGIkFGKIE4gMEGAEgQYgGAQ&sclient=gws-wiz-serp#:~:text=AmadeusGPT%3A%20a%20natural,net%20%E2%80%BA%20pdf](https://proceedings.neurips.cc/paper_files/paper/2023/file/1456560769bbc38e4f8c5055048ea712-Paper-Conference.pdf)) by [Shaokai Ye](https://github.com/yeshaokai), [Jessy Lauer](https://github.com/jeylau), [Mu Zhou](https://github.com/zhoumu53), [Alexander Mathis](https://github.com/AlexEMG) & [Mackenzie W. Mathis](https://github.com/MMathisLab).
- Like this project? Please consider giving us a star ⭐️!

## What is AmadeusGPT? 

**Developed by part of the same team that brought you [DeepLabCut](https://www.mackenziemathislab.org/deeplabcut), AmadeusGPT is a natural language interface that turns natural language descriptions of behaviors into machine-executable code.** The process of quantifying and analyzing animal behavior involves translating the naturally occurring descriptive language of their actions into machine-readable code. Yet, codifying behavior analysis is often challenging without deep understanding of animal behavior and technical machine learning knowledge, so we wanted to ease this jump.
In short, we provide a "code-free" interface for you to analysis video data of animals. If you are a [DeepLabCut](https://www.mackenziemathislab.org/deeplabcut) user, this means you could upload your videos and .h5 keypoint files and then ask questions, such as "How much time does the mouse spend in the middle of the open field?". 
In our original work (NeurIPS 2023) we used GPT3.5 and GPT4 as part of our agent. We continue to support the latest OpenAI models, and are continuing to actively develop this project. 
  
## Get started: install AmadeusGPT🎻

### [1] You will need an API key (OpenAI or OpenRouter):

**Why an API Key is needed** AmadeusGPT relies on API calls to language models for understanding natural language and generating code. You can use either OpenAI's models directly or access a wider variety of models through OpenRouter.

#### Option A: OpenAI API Key
Sign up for an [OpenAI API key](https://platform.openai.com/account/api-keys) to use GPT-4, GPT-4o, and other OpenAI models.

#### Option B: OpenRouter API Key  
Sign up for an [OpenRouter API key](https://openrouter.ai/keys) to access a wide variety of models from different providers. OpenRouter offers:
- **Pricing flexibility**: Choose from free models or pay-per-use options. See [OpenRouter pricing](https://openrouter.ai/pricing) for model costs.
- **Rate limits**: Check [OpenRouter rate limits](https://openrouter.ai/docs/limits) for usage restrictions.
- **Model variety**: Access models from OpenAI, Anthropic, Google, Meta, and more.

#### Setting up your API key:

**Option 1: .env file (recommended)**
Create a `.env` file in the repository root and add:
```
OPENAI_API_KEY=your_openai_api_key
# OR
OPENROUTER_API_KEY=your_openrouter_api_key
```

For Jupyter Notebooks, use the .env file approach and load it:
```python
from dotenv import load_dotenv
load_dotenv()  # This loads the .env file automatically
```

**Option 2: Environment variables**
```bash
# For OpenAI
export OPENAI_API_KEY='your_openai_api_key'

# For OpenRouter  
export OPENROUTER_API_KEY='your_openrouter_api_key'
```

**Option 3: Python script inline**
```python
import os
# For OpenAI
os.environ["OPENAI_API_KEY"] = 'your_openai_api_key'

# For OpenRouter
os.environ["OPENROUTER_API_KEY"] = 'your_openrouter_api_key'
```

#### Configuring models:
OpenRouter models can be specified in the configuration files located at `amadeusgpt/configs/<example_type>.yaml` under the `llm_info` section:
```yaml
llm_info:
  gpt_model: "qwen/qwen3-coder:free"  # Example OpenRouter model
  max_tokens: 20000
```

Supported models and their pricing are defined in the [`LLM` class](amadeusgpt/analysis_objects/llm.py#L23-L27). To add a new model, update the `prices` dictionary with the model name and its input/output costs per token.

### [2] Set up your Python environment:

[uv](https://github.com/astral-sh/uv) is a fast Python package manager by Astral. Install uv first:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or on Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then install AmadeusGPT:
```bash
# Clone the repository and install
git clone https://github.com/AdaptiveMotorControlLab/AmadeusGPT.git
cd AmadeusGPT
uv sync
```

**Note:** Some modules AmadeusGPT can use benefit from GPU support, therefore we recommend also having an NVIDIA GPU available and installing CUDA. 


### [3] 🪄 That's it! Now you have AmadeusGPT installed! 

See below on how to get started!


## Get started: run AmadeusGPT🎻

We provide a StreamLit App, or you can use AmadeusGPT in any python interface, such as Jupyter notebooks. For this we suggest getting started from our demos:

### Try AmadeusGPT with an example Jupyter Notebook
You can git clone (or download) this repo to grab a copy and go. We provide example notebooks [here](notebooks)!

### Here are a few demos that could fuel your own work, so please check them out!

1) [Draw a region of interest (ROI) and ask, "when is the animal in the ROI?"](https://github.com/AdaptiveMotorControlLab/AmadeusGPT/tree/main/notebooks/EPM_demo.ipynb)
2) [Use your own data](https://github.com/AdaptiveMotorControlLab/AmadeusGPT/tree/main/notebooks/YourData.ipynb) - (make sure you use a GPU to run SuperAnimal if you don't have corresponding DeepLabCut keypoint files already!)
3) [Write you own integration modules and use them](https://github.com/AdaptiveMotorControlLab/AmadeusGPT/tree/main/notebooks/Horse_demo.ipynb). Bonus: [source code](amadeusgpt/integration_modules). Make sure you delete the cached modules_embedding.pickle if you add new modules!
4) [Multi-Animal social interactions](https://github.com/AdaptiveMotorControlLab/AmadeusGPT/tree/main/notebooks/MABe_demo.ipynb)
5) [Reuse the task program generated by LLM and run it on different videos](https://github.com/AdaptiveMotorControlLab/AmadeusGPT/tree/main/notebooks/Use_Task_Program.ipynb)
7) You can ask one query across multiple videos. Put your keypoint files and video files (pairs) in the same folder and specify the `data_folder` as shown in this [Demo](https://github.com/AdaptiveMotorControlLab/AmadeusGPT/tree/main/notebooks/YourData.ipynb). Make sure your video file and keypoint file follows the normal DeepLabCut convention, i.e., `prefix.mp4` `prefix*.h5`.

### Minimal example

```python
import os
from amadeusgpt import create_project
from amadeusgpt import AMADEUS
from amadeusgpt.utils import parse_result

if 'OPENAI_API_KEY' not in os.environ:  
     os.environ['OPENAI_API_KEY'] = 'your key'

# data folder contains video files and optionally keypoint files
# please pay attention to the naming convention as described above
data_folder = "temp_data_folder"
# where the results are saved 
result_folder = "temp_result_folder"
# Create a project
config = create_project(data_folder, result_folder, video_suffix = ".mp4")

# Create an AMADEUS instance
amadeus = AMADEUS(config)

query = "Plot the trajectory of the animal using the animal center and color it by time"
qa_message = amadeus.step(query)
# we made it easier to parse the result
parse_result(amadeus, qa_message)
```

### Try AmadeusGPT with a local WebApp
- You will need to git clone this repo and have a copy locally. Then in your env run `pip install 'amadeusGPT[streamlit]'`
- Then you can open the terminal and within the directory run:
```python
make app
```


## [BONUS - installation via a customized conda env] 
If you want to set up your own env, 

```bash
conda create -n amadeusGPT python=3.10
```
the key dependencies that need installed are:
```python
pip install notebook
conda install hdf5
conda install pytables==3.8
pip install -U --pre deeplabcut
pip install amadeusgpt
```
## Citation

  If you use ideas or code from this project in your work, please cite us using the following BibTeX entry. 🙏

```
@article{ye2023amadeusGPT,
      title={AmadeusGPT: a natural language interface for interactive animal behavioral analysis}, 
      author={Shaokai Ye and Jessy Lauer and Mu Zhou and Alexander Mathis and Mackenzie Weygandt Mathis},
      journal={Thirty-seventh Conference on Neural Information Processing Systems},
      year={2023},
      url={https://openreview.net/forum?id=9AcG3Tsyoq},
```
- arXiv preprint version **[AmadeusGPT: a natural language interface for interactive animal behavioral analysis](https://arxiv.org/abs/2307.04858)** by [Shaokai Ye](https://github.com/yeshaokai), [Jessy Lauer](https://github.com/jeylau), [Mu Zhou](https://github.com/zhoumu53), [Alexander Mathis](https://github.com/AlexEMG) & [Mackenzie W. Mathis](https://github.com/MMathisLab).


## License 

AmadeusGPT is license under the Apache-2.0 license.
  -  🚨 Please note several key dependencies have their own licensing. Please carefully check the license information for [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) (LGPL-3.0 license), [SAM](https://github.com/facebookresearch/segment-anything) (Apache-2.0 license), etc.

## FAQ:

### Do I need to provide keypoint files or is video-only enough to get starte?
- If you only provide a video file, we use SuperAnimal models [SuperAnimal models](https://www.nature.com/articles/s41467-024-48792-2) to predict which animal is in your video. While we highly recommend GPU installation, we are working on faster, light-weight SuperAnimal models to work on your CPU.
- If you already have keypoint file corresponding to the video file, look how we set-up the config file in the Notebooks.  Right now we only support keypoint output from DeepLabCut.

## News
- June 2025 [v0.1.3](https://pypi.org/project/amadeusgpt/0.1.3/) is out, and we introduce new demo docs!
- July 2024 [v0.1.1](https://github.com/AdaptiveMotorControlLab/AmadeusGPT/releases/tag/v0.1.1) is released! This is a major code update ...
- June 2024 as part of the CZI EOSS, The Kavli Foundation now supports this work! ✨
- 🤩 Dec 2023, code released!
- 🔥 Our work was accepted to NeuRIPS2023
- 🧙‍♀️ Open-source code coming in the fall of 2023
- 🔮 arXiv paper and demo released July 2023
- 🪄[Contact us](http://www.mackenziemathislab.org/)
