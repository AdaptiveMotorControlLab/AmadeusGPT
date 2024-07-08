import ast
import inspect
import sys
import time
import traceback
from itertools import groupby
from operator import itemgetter
from typing import Any, Dict, Sequence
import cv2
import numpy as np
from scipy.ndimage.filters import uniform_filter1d
from amadeusgpt.logger import AmadeusLogger


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
    x = np.squeeze(x)

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


def smooth_boolean_mask(x: Sequence, window_size: int):
    # `window_size` should be at least twice as large as the
    # minimal number of consecutive frames to be smoothed out.
    if window_size % 2 == 0:
        window_size += 1
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


def get_video_length(video_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    n_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    return int(n_frames)


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
        print(
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
        openai.models.list()
        return True
    except openai.AuthenticationError:
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


# the function func2json takes a function object as inputs
# and returns a json object that contains the function name,
# input type, output type, function body, and function description

import inspect
import textwrap


def func2json(func):
    if isinstance(func, str):
        func_str = textwrap.dedent(func)

        # Parse the function string to an AST
        parsed = ast.parse(func_str)
        func_def = parsed.body[0]
        # Use the AST to extract the function's name
        func_name = func_def.name
        # Extract the docstring directly from the AST
        docstring = ast.get_docstring(func_def)
        # Remove the docstring node if present
        if (
            func_def.body
            and isinstance(func_def.body[0], ast.Expr)
            and isinstance(func_def.body[0].value, (ast.Str, ast.Constant))
        ):
            func_def.body.pop(0)

        # Remove decorators from the function definition
        func_def.decorator_list = []

        # Convert the modified AST back to source code
        if hasattr(ast, "unparse"):
            source_without_docstring_or_decorators = ast.unparse(func_def)
        else:
            # Placeholder for actual conversion in older Python versions
            source_without_docstring_or_decorators = None  # Consider using `astor` here

        # Attempt to evaluate return annotation, if any
        return_annotation = "No return annotation"
        if func_def.returns:
            return_annotation = ast.unparse(func_def.returns)

        json_obj = {
            "name": func_name,
            "inputs": "",
            "source_code": source_without_docstring_or_decorators,
            "docstring": docstring,
            "return": return_annotation,
        }
        return json_obj
    else:

        # Capture the function's signature for input arguments
        sig = inspect.signature(func)
        inputs = {name: str(param.annotation) for name, param in sig.parameters.items()}

        # Extract the docstring
        docstring = inspect.getdoc(func)
        if docstring:
            docstring = textwrap.dedent(docstring)

        # Capture the function's full source and parse it to an AST
        full_source = inspect.getsource(func)
        parsed = ast.parse(textwrap.dedent(full_source))
        func_def = parsed.body[0]

        # Remove the docstring node if present
        if (
            func_def.body
            and isinstance(func_def.body[0], ast.Expr)
            and isinstance(func_def.body[0].value, (ast.Str, ast.Constant))
        ):
            func_def.body.pop(0)

        # Remove decorators from the function definition
        func_def.decorator_list = []

        # Convert the modified AST back to source code
        if hasattr(ast, "unparse"):
            source_without_docstring_or_decorators = ast.unparse(func_def)
        else:
            # Placeholder for actual conversion in older Python versions
            source_without_docstring_or_decorators = None  # Consider using `astor` here

        json_obj = {
            "name": func.__name__,
            "inputs": inputs,
            "source_code": textwrap.dedent(source_without_docstring_or_decorators),
            "docstring": docstring,
            "return": str(sig.return_annotation),
        }
        return json_obj

