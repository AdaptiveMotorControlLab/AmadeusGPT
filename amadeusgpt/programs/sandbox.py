import ast
import inspect
import json
import os
import re
import traceback
import typing
from functools import wraps
import matplotlib.pyplot as plt
import numpy as np
from amadeusgpt.analysis_objects.analysis_factory import create_analysis
from amadeusgpt.analysis_objects.event import BaseEvent
from amadeusgpt.analysis_objects.relationship import Orientation
from amadeusgpt.config import Config
from amadeusgpt.programs.api_registry import (CORE_API_REGISTRY,
                                              INTEGRATION_API_REGISTRY)
from amadeusgpt.programs.task_program_registry import (TaskProgram,
                                                       TaskProgramLibrary)
from pathlib import Path
from collections import defaultdict

class QA_Message(dict):
    def __init__(self, *args, **kwargs):
        super(QA_Message, self).__init__(*args, **kwargs)

    def get_masks(self):
        function_rets = self["function_rets"]
        # if function_ret is a list of events
        if (
            isinstance(function_rets, list)
            and len(function_rets) > 0
            and isinstance(function_rets[0], BaseEvent)
        ):
            events = function_rets
            masks = []
            for event in events:
                masks.append(event.generate_mask())
            return np.array(masks)
        else:
            raise ValueError("No events found in the function_rets")
        
    def get_serializable(self):
        """
        Only part of qa messages are serializable.
        """
        selected_keys = ['query', 'code', 'chain_of_thought', 'function_rets', 'meta_info']
        ret = {}
        for key in selected_keys:
            ret[key] = self[key]
        return ret


def create_message(query, sandbox):
    return QA_Message({
        "query": query,
        "code": None,
        "chain_of_thought": None,
        "plots": [],
        "error_message": None,
        "function_rets": None,
        "sandbox": sandbox,
        "out_videos": None,
        "pose_video": None,
        "meta_info": None,
    })


class SandboxBase:
    """
    This class takes task program library, api registry.
    It's responsible for maintaining the states of the ongoing execution
    of the task program.
    It's also responsible for formatting apis, task programs to a format that
    GPT can understand better.

    Following are examples

    '''coreapidocs
    # this function gives the speed of the animal
    BaseEvent: A class that represents an event
    get_speed() -> np.ndarray
    get_animal_state_events() -> List[BaseEvent]
    '''

    '''optionalapidocs
    # these functions are loaded dynamically for the current query
    '''

    '''taskprograms
    # available task programs
    '''

    '''variables
    # variables from previous runs
    '''

    '''helperfunction
    def helper_function():
        do something else
    '''

    '''maincode
    def main()
        a = helper_function()
        b = api_1()
        return a + b
    '''
    """

    def __init__(self):
        # In the future we might want to create a copy for these two
        self.api_registry = CORE_API_REGISTRY
        self.integration_api_registry = INTEGRATION_API_REGISTRY

    def _map_type(self, type_string):
        """
        example param_dict
        {'self': "<class 'inspect._empty'>",
         'object_name': "<class 'str'>",
         'relation_query': "<class 'str'>",
         'comparison': 'typing.Optional[str]',
         'negate': "<class 'inspect._empty'>",
         'bodypart_names': 'typing.Optional[typing.List[str]]',
         'min_window': "<class 'int'>",
         'max_window': "<class 'int'>",
         'smooth_window_size': "<class 'inspect._empty'>"}

        We want it to be
        (object_name: str, relation_query: str, comparison: typing.Optional[str], negate: bool, bodypart_names: typing.Optional[typing.List[str]], min_window: int, max_window: int, smooth_window_size: int)
        """
        class_pattern = re.compile(r"<class '(.+)'>")
        # Check if the type string matches the "<class '...'>" pattern
        class_match = class_pattern.match(type_string)
        if class_match:
            type_name = class_match.group(1)
            # Special handling for 'inspect._empty', which maps to None
            if type_name == "inspect._empty":
                return "None"
            return type_name
        else:
            # Attempt to strip the 'typing.' prefix from type annotations
            modified_string = re.sub(r"typing\.(\w+)", r"\1", type_string)
            return modified_string

    def enforce_indentation(self, text, spaces_per_indent=4):
        """
        Adjusts the indentation of the given text to a specific number of spaces per indent level.
        Assumes the text uses spaces for indentation.

        :param text: The input text.
        :param spaces_per_indent: Number of spaces for each indentation level.
        :return: Text with adjusted indentation.
        """
        lines = text.split("\n")
        adjusted_lines = []

        for line in lines:
            # Count the leading spaces
            leading_spaces = len(line) - len(line.lstrip(" "))
            # Calculate the new indentation level, assuming the smallest non-zero
            # indentation level in the original text is one level.
            new_indentation = (leading_spaces // spaces_per_indent) * spaces_per_indent
            adjusted_line = " " * new_indentation + line.lstrip(" ")
            adjusted_lines.append(adjusted_line)

        return "\n".join(adjusted_lines)

    def _fill_parameters(self, param_dict):

        ret = ""
        for i, (name, type) in enumerate(param_dict.items()):
            if name == "self":
                continue
            if type == "<class 'inspect._empty'>":
                ret += f"{name}: {self._map_type(type)}"
            else:
                ret += f"{name}: {self._map_type(type)}"
            if i < len(param_dict) - 2:
                ret += ", "
        return ret

    def get_helper_functions(self):
        return ""

    def get_variables(self):
        # not sure if this needs to be implemented
        return ""

    def get_main_code(self):
        # the code that should actually be executed
        # we instruct LLM to write this kind of block
        # we let the code generation LLM write this part of code
        return ""


def wrap_instance_method(instance, method_name):
    # Fetch the method directly from the instance
    method = getattr(instance, method_name)
    if not callable(method):
        raise ValueError(f"{method_name} is not callable.")
    from inspect import signature

    sig = signature(method)
    # Note: No need to extract the instance, as it's already provided

    @wraps(method)
    def wrapper(*args, **kwargs):
        # Delegate the call to the instance method
        return method(*args, **kwargs)

    # Update the wrapper to have the same signature as the original method
    sig = inspect.signature(method)
    wrapper.__signature__ = sig
    wrapper.__doc__ = method.__doc__

    return wrapper


class Sandbox(SandboxBase):
    def __init__(self, config):
        super().__init__()
        self.task_program_library = TaskProgramLibrary().get_task_programs()
        self.config = config
        self.messages = []
        # initialize the code execution namespace with builtins
        self.exec_namespace = {"__builtins__": __builtins__}
        # update_namespace initializes behavior analysis
        self.update_namespace()
        # then we can configure behavior analysis using vlm
        self.meta_info = None
        # where llms are stored
        self.llms = {}
        # just easier to pass this around
        self.query = None
        self.matched_modules = []
        # example result_cahe
        """
        {'query' :  
            {
                'file1.mp4':  QA_Message(),
                'file2.mp4':  QA_Message(),
            }
        }         
        """

        self.result_cache = defaultdict(dict)
        # configure how to save the results to a result folder
        self.result_folder = Path(self.config["result_info"].get("result_folder", "./results"))

    def configure_using_vlm(self):
        # example meta_info:
        """
        {
        "description": "Top-down view of a laboratory setting with a small animal marked with colored dots on a white surface. Various laboratory equipment and objects are visible in the background.",
        "individuals": 1,
        "species": "topview_mouse",
        "background_objects": ["laboratory equipment", "white surface", "colored dots"]
        }
        """
        json_obj = self.llms["visual_llm"].speak(self)

        self.meta_info = json_obj
        # configure meta info on the analysis managers
        analysis = self.exec_namespace["behavior_analysis"]
        analysis.animal_manager.configure_animal_from_meta(json_obj)

    def get_core_api_docs(self):
        """
        Turn the core api docs into a format that GPT can understand

        We optionally also mix the integration api docs
        """
        ret = """```coreapidocs\n
All following functions are part of class AnimalBehaviorAnalysis:
The usage and the parameters of the functions are provided."""
        for name, api in self.api_registry.items():
            description = api["description"]
            # parameters is a dictionary that might contain self
            parameters = self._fill_parameters(api["parameters"])
            description = self.enforce_indentation(description)
            ret += f"{name}({parameters}): \n{description}\n"

        for name, api in self.integration_api_registry.items():

            if name not in self.matched_modules:
                continue
            description = api["description"]
            # parameters is a dictionary that might contain self
            parameters = self._fill_parameters(api["parameters"])
            description = self.enforce_indentation(description)
            ret += f"{name}({parameters}): \n{description}\n"

        ret += "\n```"
        return ret

    def get_task_program_docs(self):
        ret = "```taskprograms\n"
        for name, task_program in self.task_program_library.items():
            description = task_program.json_obj["docstring"]
            ret += f"{name}(config: Config): \n{description}\n"
        ret += "\n```"

        return ret

    def get_query_block(self):
        query = self.query
        ret = f"```query\n {query}\n```"
        return ret

    def update_config(self, config):
        self.config = config
        self.update_namespace()

    def copy(self):
        return Sandbox(self.config, self.api_registry)

    def update_matched_integration_modules(self, matched_modules):
        self.matched_modules = matched_modules

    def update_namespace(self):
        # we need to manage the scope of the session
        # there are potentially new variables, new task programs, new apis
        analysis = create_analysis(self.config)
        for api in self.api_registry.values():
            f = wrap_instance_method(analysis, api["name"])
            self.exec_namespace[api["name"]] = f

        for name, task_program in self.task_program_library.items():
            self.exec_namespace[name] = task_program

        current_scope = globals()
        for name, value in current_scope.items():
            if callable(value) or isinstance(
                value, (int, float, str, list, dict, tuple)
            ):
                self.exec_namespace[name] = value

        # the namespace needs to access the config
        from amadeusgpt.config import Config

        self.exec_namespace["config"] = self.config
        self.exec_namespace["Config"] = Config
        # the namespace needs to access AnimalBehaviorAnalysis for API

        self.exec_namespace["plt"] = plt
        # instance linked to class?
        self.exec_namespace["create_analysis"] = create_analysis
        self.exec_namespace["behavior_analysis"] = analysis

        # useful classes needed the APIs

        self.exec_namespace["Orientation"] = Orientation
        self.exec_namespace["List"] = typing.List
        self.exec_namespace["BaseEvent"] = BaseEvent
        # numpy might be needed for raw kinematics
        self.exec_namespace["np"] = np
        # to allow the program to access existing task programs
        self.exec_namespace["task_programs"] = TaskProgramLibrary.get_task_programs()

    def code_execution(self, qa_message):
        # update the namespace in the beginning of code execution makes sure that 
        # if there is a change in the config, we always use the newest config
        self.update_namespace()
        code = qa_message["code"]
        # not need to do further if there was no code found
        if code is None:
            return qa_message
        exec(code, self.exec_namespace)
        # call the main function
        function_name = self.get_function_name_from_string(code)
        call_str = f"{function_name}(config)"
        try:
            exec(f"result = {call_str}", self.exec_namespace)
            qa_message["error_message"] = None
        except Exception as e:
            # use traceback to get full error
            full_traceback = traceback.format_exc()
            print(full_traceback)
            qa_message["error_message"] = str(full_traceback)
            return qa_message
        result = self.exec_namespace["result"]
        qa_message["function_rets"] = result

        return qa_message

    def get_function_name_from_string(self, code):
        # Parse the string into an AST
        parsed_ast = ast.parse(code)
        # Initialize a variable to hold the function name
        function_name = None

        # Traverse the AST
        for node in ast.walk(parsed_ast):
            if isinstance(node, ast.FunctionDef):
                function_name = node.name
                break

        return function_name

    def register_task_program(self, code, parents=None, mutation_from=None):
        self.update_namespace()

        if isinstance(code, str):
            TaskProgramLibrary.register_task_program(
                creator="llm", parents=parents, mutation_from=mutation_from
            )(code)

        elif isinstance(code, TaskProgram):
            TaskProgramLibrary.register_task_program(
                creator="llm", parents=parents, mutation_from=mutation_from
            )(code)

        elif isinstance(code, dict):
            TaskProgramLibrary.register_task_program(
                creator="llm", parents=parents, mutation_from=mutation_from
            )(code)

    def register_llm(self, name, llm):
        self.llms[name] = llm

    def events_to_videos(self, events, function_name):
        behavior_analysis = self.exec_namespace["behavior_analysis"]
        visual_manager = behavior_analysis.visual_manager
        # save video clips to the result folder
        out_folder = str(self.result_folder)
        os.makedirs(out_folder, exist_ok=True)
        behavior_name = "_".join(function_name.split(" "))
        return visual_manager.generate_video_clips_from_events(
            out_folder, events, behavior_name
        )

    def render_qa_message(self, qa_message):
        """
        To be called after code execution.
        If the function returns a list of events, we visualize those events to keypoint plot, ethogram plot and videos
        if the function returns is a tuple of axe and figure, we put them into the plots filed 
        """
        function_rets = qa_message["function_rets"]
        behavior_analysis = self.exec_namespace["behavior_analysis"]
        bodypart_names = behavior_analysis.animal_manager.get_keypoint_names()
        qa_message["pose_video"] = (
            behavior_analysis.animal_manager.superanimal_predicted_video
        )
        visual_manager = behavior_analysis.visual_manager
        plots = []
        if isinstance(function_rets, tuple):
            # could be plotting tuple
            if isinstance(function_rets[0], plt.Figure):
                # this is for "return fig, ax"
                plots.append(function_rets)

            else:
                for e in function_rets:
                    if (
                        isinstance(e, list)
                        and len(e) > 0
                        and isinstance(e[0], BaseEvent)
                    ):
                        # here we need to understand what we do with the events
                        # we have ethogram plot, keypoint plot, head orientation plot, scene plot
                        # and animal interaction plot
                        # of course we can show all of them at the same time except for animal interaction if there are multi animals
                        # keypoint visualization
                        plots.append(
                            visual_manager.get_keypoint_visualization(
                                bodypart_names=bodypart_names, events=e
                            )
                        )
                        qa_message["out_videos"] = self.events_to_videos(
                            e, self.get_function_name_from_string(qa_message["code"])
                        )

        elif (
            isinstance(function_rets, list)
            and len(function_rets) > 0
            and isinstance(function_rets[0], BaseEvent)
        ):
            # this is for "return events"
            plots.append(
                visual_manager.get_keypoint_visualization(
                    bodypart_names=bodypart_names, events=function_rets
                )
            )
            plots.append(
                visual_manager.get_ethogram_visualization(events=function_rets)
            )
            qa_message["out_videos"] = self.events_to_videos(
                function_rets, self.get_function_name_from_string(qa_message["code"])
            )
     
        qa_message["plots"].extend(plots)
        return qa_message

    def llm_step(self, user_query):
        """
        1) We first use gpt-4o to create meta_info describing the scene
        2) We then ask LLM to generate code based on the query
        3) We also cache the qa_message for future reference
        """
        qa_message = create_message(user_query, self)

        # so that the frontend can display it too
        if self.meta_info is not None:
            qa_message["meta_info"] = self.meta_info

        self.messages.append(qa_message)
        # there might be better way to set this
        self.query = user_query
        self.llms["code_generator"].speak(self)
        # cache the resulted qa message for future use
        self.result_cache[user_query][self.config['video_info']['video_file_path']] = qa_message

        # task program that is written by llm is automatically registered to be used in the future
        
        if qa_message['code'] is not None and qa_message['error_message'] is None:
            TaskProgramLibrary.register_task_program(creator="llm")(qa_message['code'])

        return qa_message

    def run_task_program(self, config: Config, task_program_name: str):
        """
        1) sandbox is also responsible for running task program
        2) self.task_program_library references to a singleton so a different sandbox still has reference to the task program
        """
        # update the config 
        self.config = config

        task_program = self.task_program_library[task_program_name]
        # there might be better way to set this
        self.query = task_program_name
        qa_message = create_message(self.query, self)
        qa_message["code"] = task_program["source_code"]
        self.messages.append(qa_message)

        # code execution will use the latest config, if updated
        self.code_execution(qa_message)

        qa_message = self.render_qa_message(qa_message)
        self.result_cache[task_program_name][config['video_info']['video_file_path']] = qa_message
        return qa_message

    # def step(self, user_query, number_of_debugs=1):
    #     """
    #     Currently not used. We tried to seperate LLM inference and code execution
    #     """            
    #     qa_message = create_message(user_query, self)

    #     if self.meta_info is not None:
    #         qa_message["meta_info"] = self.meta_info

    #     self.messages.append(qa_message)

    #     self.query = user_query
    #     self.llms["code_generator"].speak(self)
    #     # all these llms collectively compose a amadeus_answer
    #     qa_message = self.code_execution(qa_message)

    #     if qa_message["error_message"] is not None:
    #         for i in range(number_of_debugs):
    #             self.llms["self_debug"].speak(self)
    #             qa_message = self.code_execution(qa_message)

    #     qa_message = self.render_qa_message(qa_message)
    #     self.result_cache[user_query][self.config['video_info']['video_file_path']] = qa_message
    #     return qa_message


def save_figure_to_tempfile(fig):
    """
    Only used for debug
    """
    import tempfile

    # save the figure
    folder_path = os.path.join("tmp_imgs")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # Generate a unique temporary filename in the specified folder
    temp_file = tempfile.NamedTemporaryFile(
        dir=folder_path, suffix=".png", delete=False
    )
    filename = temp_file.name
    temp_file.close()
    fig.savefig(
        filename,
        format="png",
        bbox_inches="tight",
        pad_inches=0.0,
        dpi=400,
        transparent=True,
    )
    return filename


def render_temp_message(query, sandbox):
    """
    Only used for debug
    """
    import streamlit as st

    qa_message = create_message("random query", sandbox)

    with open("temp_answer.json", "r") as f:
        data = json.load(f)
        qa_message["chain_of_thought"] = data["chain_of_thought"]

    text = qa_message["chain_of_thought"]
    lines = text.split("\n")
    inside_code_block = False
    code_block = []

    for line in lines:
        if line.strip().startswith("```python"):
            inside_code_block = True
            code_block = []
        elif line.strip().startswith("```") and inside_code_block:
            inside_code_block = False
            code = "\n".join(code_block)
            qa_message["code"] = code
            st.code(code, language="python")
        elif inside_code_block:
            code_block.append(line)
        else:
            st.markdown(line)

    if qa_message["code"] is not None:
        qa_message = sandbox.code_execution(qa_message)
        print("after code execution")
        print(len(qa_message["function_rets"]))
        events = qa_message["function_rets"]

    sandbox.render_qa_message(qa_message)

    if qa_message["function_rets"] is not None:
        st.markdown(qa_message["function_rets"])

    plots = qa_message["plots"]
    print("plots", plots)
    for fig, axe in plots:
        filename = save_figure_to_tempfile(fig)
        st.image(filename, width=600)

    videos = qa_message["out_videos"]
    print("videos", videos)
    for video in videos:
        st.video(video)


if __name__ == "__main__":
    # testing qa message
    import pickle

    from amadeusgpt.analysis_objects.object import ROIObject
    from amadeusgpt.main import create_amadeus

    config = Config("amadeusgpt/configs/EPM_template.yaml")

    amadeus = create_amadeus(config)
    sandbox = amadeus.sandbox
    analysis = sandbox.exec_namespace["behavior_analysis"]
    with open("temp_roi_objects.pickle", "rb") as f:
        roi_objects = pickle.load(f)

    for name, roi_object in roi_objects.items():
        analysis.object_manager.add_roi_object(ROIObject(name, roi_object["Path"]))

    render_temp_message("random query", sandbox)   