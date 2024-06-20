import ast
import copy
import inspect
import re
import sys
import time
import traceback
from itertools import groupby
from operator import itemgetter
from pydoc import doc
from typing import Any, Dict, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
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


def frame_number_to_minute_seconds(frame_number, video_path):
    fps = get_fps(video_path)
    temp = frame_number / fps
    minutes = int(temp // 60)
    seconds = int(temp % 60)
    ret = f"{minutes:02d}:{seconds:02d}"
    return ret


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
        if "" not in lines[i]:
            i += 1
            continue
        else:
            lines[i] = lines[i].replace("", "").strip()
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
                if "" not in lines_copy[i]:
                    in_function = False
                    continue
                function_lines.append(
                    next_line.replace("", "").replace(example_indentation, "", 1)
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


def get_func_name_from_func_string(function_string: str):
    import ast

    # Parse the string into an AST
    parsed_ast = ast.parse(function_string)

    # Initialize a variable to hold the function name
    function_name = None

    # Traverse the AST
    for node in ast.walk(parsed_ast):
        if isinstance(node, ast.FunctionDef):
            function_name = node.name
            break

    return function_name
