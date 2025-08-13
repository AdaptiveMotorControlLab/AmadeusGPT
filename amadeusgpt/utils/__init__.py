import ast
import inspect
import sys
import time
import traceback
from collections import defaultdict
import textwrap
import numpy as np
from amadeusgpt.analysis_objects.event import Event
from amadeusgpt.logger import AmadeusLogger
from IPython.display import Markdown, Video, display, HTML
from amadeusgpt.utils.openai_adapter import OpenAIAdapter

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
    client = OpenAIAdapter(key)
    return client.validate()

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

def func2json(func):
    if isinstance(func, str):
        func_str = textwrap.dedent(func)
        parsed = ast.parse(func_str)
        func_def = parsed.body[0]
        func_name = func_def.name
        docstring = ast.get_docstring(func_def)
        if (
            func_def.body
            and isinstance(func_def.body[0], ast.Expr)
            and isinstance(func_def.body[0].value, (ast.Str, ast.Constant))
        ):
            func_def.body.pop(0)
        func_def.decorator_list = []
        if hasattr(ast, "unparse"):
            source_without_docstring_or_decorators = ast.unparse(func_def)
        else:
            source_without_docstring_or_decorators = None
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
        sig = inspect.signature(func)
        inputs = {name: str(param.annotation) for name, param in sig.parameters.items()}
        docstring = inspect.getdoc(func)
        if docstring:
            docstring = textwrap.dedent(docstring)
        full_source = inspect.getsource(func)
        parsed = ast.parse(textwrap.dedent(full_source))
        func_def = parsed.body[0]
        if (
            func_def.body
            and isinstance(func_def.body[0], ast.Expr)
            and isinstance(func_def.body[0].value, (ast.Str, ast.Constant))
        ):
            func_def.body.pop(0)
        func_def.decorator_list = []
        if hasattr(ast, "unparse"):
            source_without_docstring_or_decorators = ast.unparse(func_def)
        else:
            source_without_docstring_or_decorators = None
        json_obj = {
            "name": func.__name__,
            "inputs": inputs,
            "source_code": textwrap.dedent(source_without_docstring_or_decorators),
            "docstring": docstring,
            "return": str(sig.return_annotation),
        }
        return json_obj

class QA_Message:
    def __init__(self, query: str, video_file_paths: list[str]):
        self.query = query
        self.video_file_paths = video_file_paths
        self.code = None
        self.chain_of_thought = None
        self.error_message = defaultdict(list)
        self.plots = defaultdict(list)
        self.out_videos = defaultdict(list)
        self.pose_video = defaultdict(list)
        self.function_rets = defaultdict(list)
        self.meta_info = {}
    def get_masks(self) -> dict[str, np.ndarray]:
        ret = {}
        function_rets = self.function_rets
        for video_path, rets in function_rets.items():
            if isinstance(rets, list) and len(rets) > 0 and isinstance(rets[0], Event):
                events = rets
                masks = []
                for event in events:
                    masks.append(event.generate_mask())
                ret[video_path] = np.array(masks)
            else:
                ret[video_path] = None
        return ret
    def serialize_qa_message(self):
        return {
            "query": self.query,
            "video_file_paths": self.video_file_paths,
            "code": self.code,
            "chain_of_thought": self.chain_of_thought,
            "error_message": self.error_message,
            "plots": None,
            "out_videos": self.out_videos,
            "pose_video": self.pose_video,
            "function_rets": self.function_rets,
            "meta_info": self.meta_info,
        }
def create_qa_message(query: str, video_file_paths: list[str]) -> QA_Message:
    return QA_Message(query, video_file_paths)
def parse_result(amadeus, qa_message, use_ipython=True, skip_code_execution=False):
    if use_ipython:
        display(Markdown(qa_message.chain_of_thought))
    else:
        print(qa_message.chain_of_thought)
    sandbox = amadeus.sandbox
    if not skip_code_execution:
        qa_message = sandbox.code_execution(qa_message)
    qa_message = sandbox.render_qa_message(qa_message)
    if len(qa_message.out_videos) > 0:
        print(f"videos generated to {qa_message.out_videos}")
        print(
            "Open it with media player if it does not properly display in the notebook"
        )
        if use_ipython:
            if len(qa_message.out_videos) > 0:
                for identifier, event_videos in qa_message.out_videos.items():
                    for event_video in event_videos:
                        display(Video(event_video, embed=True))
    if use_ipython:
        from matplotlib.animation import FuncAnimation
        if len(qa_message.function_rets) > 0:
            for identifier, rets in qa_message.function_rets.items():
                if not isinstance(rets, (tuple, list)):
                    rets = [rets]
                for ret in rets:
                    if isinstance(ret, FuncAnimation):
                        display(HTML(ret.to_jshtml()))
                    else:
                        display(Markdown(str(qa_message.function_rets[identifier])))
    return qa_message

def patch_pytorch_weights_only():
    """
    Patch for PyTorch 2.6 weights_only issue with DeepLabCut SuperAnimal models.
    This adds safe globals to allow loading of ruamel.yaml.scalarfloat.ScalarFloat objects.
    Only applies the patch if torch.serialization.add_safe_globals exists (PyTorch >=2.6).
    """
    try:
        import torch
        from ruamel.yaml.scalarfloat import ScalarFloat
        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([ScalarFloat])
    except ImportError:
        pass  # If ruamel.yaml is not available, continue without the patch 