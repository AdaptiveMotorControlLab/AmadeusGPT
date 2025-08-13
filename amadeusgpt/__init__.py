#
"""AmadeusGPT: We turn natural language descriptions of behaviors into machine-executable code.
Written by Shaokai Ye and contributors | see Ye et al. 2023 NeurIPS for details.
SOURCE CODE: https://github.com/AdaptiveMotorControlLab/AmadeusGPT
Apache-2.0 license
"""
import streamlit as st

from matplotlib import pyplot as plt

from amadeusgpt.integration_modules import *
from amadeusgpt.main import AMADEUS
from amadeusgpt.project import create_project
from amadeusgpt.version import VERSION, __version__

params = {
    "axes.labelsize": 10,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "text.usetex": False,
    "figure.figsize": [8, 8],
    "font.size": 10,
}
plt.rcParams.update(params)
plt.rcParams["figure.subplot.hspace"] = 0.5  # Horizontal space between subplots
plt.rcParams["figure.subplot.wspace"] = 0.5  # Vertical space between subplots
plt.style.use("dark_background")
