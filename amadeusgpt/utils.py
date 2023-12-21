import copy
import re
from itertools import groupby
from operator import itemgetter
from typing import Sequence
import inspect
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import uniform_filter1d
import time
from matplotlib.ticker import FuncFormatter
from amadeusgpt.logger import AmadeusLogger
import sys
import traceback
import cv2

def moving_average(x: Sequence, window_size: int, pos: str = "centered"):
    """
    Compute the moving average of a time series.
    :param x: array_like, 1D input array
    :param window_size: int
        Must be odd positive, and less than or equal to the size of *x*.
    :param pos: str, optional (default="centered")
        Averaging window position.
        By default, the window is centered on the current data point,
        thus averaging over *window_size* // 2 past and future observations;
        no delay is introduced in the averaging process.
        Other options are "backward", where the average is taken
        from the past *window_size* observations; and "forward",
        where the average is taken from the future *window_size* observations.
    :return: ndarray
        Filtered time series with same length as input
    """
    # This function is not only very fast (unlike convolution),
    # but also numerically stable (unlike the one based on cumulative sum).
    # https://stackoverflow.com/questions/13728392/moving-average-or-running-mean/27681394#27681394
    x = np.asarray(x, dtype=float)
    window_size = int(window_size)

    if window_size > x.size:
        raise ValueError("Window size must be less than or equal to the size of x.")

    if window_size < 1 or not window_size % 2:
        raise ValueError("Window size must be a positive odd integer.")

    middle = window_size // 2
    if pos == "centered":
        origin = 0
    elif pos == "backward":
        origin = middle
    elif pos == "forward":
        origin = -middle
    else:
        raise ValueError(f"Unrecognized window position '{pos}'.")

    return uniform_filter1d(x, window_size, mode="constant", origin=origin)


def moving_variance(x, window_size):
    """
    Blazing fast implementation of a moving variance.
    :param x: ndarray, 1D input
    :param window_size: int, window length
    :return: 1D ndarray of length len(x)-window+1
    """
    nrows = x.size - window_size + 1
    n = x.strides[0]
    mat = np.lib.stride_tricks.as_strided(
        x,
        shape=(nrows, window_size),
        strides=(n, n),
    )
    return np.var(mat, axis=1)


def smooth_boolean_mask(x: Sequence, window_size: int):
    # `window_size` should be at least twice as large as the
    # minimal number of consecutive frames to be smoothed out.
    return moving_average(x, window_size) > 0.5


def group_consecutive(x: Sequence):
    for _, g in groupby(enumerate(x), key=lambda t: t[0] - t[1]):
        yield list(map(itemgetter(1), g))


def get_fps(video_path):    
    # Load the video
    video = cv2.VideoCapture(video_path)
    # Get the FPS
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()    
    return fps


def frame_number_to_minute_seconds(frame_number, video_path):
    fps = get_fps(video_path)
    temp = frame_number / fps
    minutes = int(temp // 60)
    seconds = int(temp % 60)
    ret = f"{minutes:02d}:{seconds:02d}"
    return ret


def _plot_ethogram(masks, ax, video_path, cmap="rainbow"):
    fps = get_fps(video_path)
    n_rois = len(masks)
    cmap = plt.cm.get_cmap(cmap, n_rois)

    colors = cmap(np.linspace(0, 1, n_rois))
    pos = []
    for mask in masks.values():
        video_length = len(mask)
        pos.append(np.flatnonzero(mask) / fps)

    def format_func(value, tick_number):
        # format the value to minute:second format
        minutes = int(value // 60)
        seconds = int(value % 60)
        ret = f"{minutes:02d}:{seconds:02d}"
        return ret

    ax.set_xlim([0, video_length / fps])
    ax.eventplot(pos, colors=colors)
    ax.xaxis.set_major_formatter(FuncFormatter(format_func))
    ax.set_xlabel("Time (mm:ss)")


def search_generated_func(text):
    functions = []
    lines = text.split("\n")
    func_names = []
    i = 0

    while i < len(lines):
        line = lines[i]
        func_signature = "def task_program"
        if line.startswith(func_signature):
            start = line.index("def ") + 4
            end = line.index("(")
            func_name = line[start:end]
            func_names.append(func_name)
            function_lines = [line]
            nesting_level = 0
            i += 1

            while i < len(lines):
                line = lines[i]
                # Check for nested function definitions
                if line.lstrip().startswith("def "):
                    nesting_level += 1
                elif line.lstrip().startswith("return") and nesting_level > 0:
                    nesting_level -= 1
                elif line.lstrip().startswith("return") and nesting_level == 0:
                    function_lines.append(line)
                    break

                function_lines.append(line)
                i += 1

            functions.append("\n".join(function_lines))
        i += 1

    return functions, func_names


def search_external_module_for_context_window(text):
    """
    just include everything
    """
    functions = []
    i = 0
    lines = text.split("\n")
    func_names = []
    while i < len(lines):
        line = lines[i].strip()
        func_signature = "def "
        if line.strip() == "":
            i += 1
            continue
        if line.startswith(func_signature):
            start = line.index("def ") + 4
            end = line.index("(")
            func_name = line[start:end]
            func_names.append(func_name)
            function_lines = [line]
            in_function = True
            while in_function:
                i += 1
                if i == len(lines):
                    break
                next_line = lines[i].rstrip()
                if not next_line.startswith((" ", "\t")):
                    in_function = False
                    continue
                function_lines.append(next_line)
            functions.append("\n".join(function_lines))
        else:
            i += 1
    return functions, func_names


def search_external_module_for_task_program_table(text):
    """
    in this case, just include everything
    """
    functions = []
    i = 0
    lines = text.split("\n")
    lines_copy = copy.deepcopy(lines)
    func_names = []
    example_indentation = " " * 4
    while i < len(lines):
        if lines[i].startswith("def "):
            start = lines[i].index("def ") + 4
            end = lines[i].index("(")
            func_name = lines[i][start:end]
            func_names.append(func_name)
        if ">>>" not in lines[i]:
            i += 1
            continue
        else:
            lines[i] = lines[i].replace(">>>", "").strip()
        line = lines[i].strip()
        func_signature = "def "
        if line.strip() == "":
            i += 1
            continue
        if line.startswith("def "):
            function_lines = [line]
            in_function = True
            while in_function:
                i += 1
                if i == len(lines):
                    break
                next_line = lines[i].rstrip()
                if ">>>" not in lines_copy[i]:
                    in_function = False
                    continue
                function_lines.append(
                    next_line.replace(">>>", "").replace(example_indentation, "", 1)
                )
            functions.append("\n".join(function_lines))
        else:
            i += 1

    return functions, func_names


def filter_kwargs_for_function(func, kwargs):
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # before calling the function
        result = func(*args, **kwargs)  # call the function
        end_time = time.time()  # after calling the function
        AmadeusLogger.debug(
            f"The function {func.__name__} took {end_time - start_time} seconds to execute."
        )
        return result

    return wrapper


def parse_error_message_from_python():
    exc_type, exc_value, exc_traceback = sys.exc_info()
    traceback_str = "".join(
        traceback.format_exception(exc_type, exc_value, exc_traceback)
    )
    return traceback_str

def validate_openai_api_key(key):
    import openai

    openai.api_key = key
    try:
        openai.Model.list()
        return True
    except openai.error.AuthenticationError:
        return False

def flatten_tuple(t):
    """
    Used to handle function returns
    """
    flattened = []
    for item in t:
        if isinstance(item, tuple):
            flattened.extend(flatten_tuple(item))
        else:
            flattened.append(item)
    return tuple(flattened)
