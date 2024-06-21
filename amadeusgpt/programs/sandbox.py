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
from amadeusgpt.implementation import AnimalBehaviorAnalysis
from amadeusgpt.managers import visual_manager
from amadeusgpt.programs.api_registry import (CORE_API_REGISTRY,
                                              INTEGRATION_API_REGISTRY)
from amadeusgpt.programs.task_program_registry import (TaskProgram,
                                                       TaskProgramLibrary)


def create_message(query, sandbox):
    return {
        "query": query,
        "code": None,
        "chain_of_thought": None,
        "plots": [],
        "error_message": None,
        "function_rets": None,
        "sandbox": sandbox,
        "out_videos": [],
    }


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
        self.exec_namespace = {"__builtins__": __builtins__}
        self.update_namespace()
        self.visual_cache = {}
        self.llms = {}
        self.query = None
        self.matched_modules = []

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

    def visual_validate(self, video_file, events, behavior_name):
        # change video and keypoint file
        analysis = create_analysis(self.config)
        out_folder = os.path.join(self.config["evo_info"]["data_folder"], "inspection")
        discovered_behaviors = []
        for name, task_program in self.task_program_library.items():
            if task_program["creator"] != "human":
                discovered_behaviors.append(name)
        # if behavior_name not in self.visual_cache and behavior_name in discovered_behaviors:
        #     os.makedirs(out_folder, exist_ok=True)
        #     analysis.visual_manager.generate_video_clips_from_events(
        #         out_folder,
        #         video_file,
        #         events,
        #         behavior_name)
        #     self.visual_cache[behavior_name] = 'set'

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

    def parse_function_results(self, function_rets):
        pass

    def code_execution(self, qa_message):
        # add main function into the namespace
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

    def events_to_videos(self, events, query):
        behavior_analysis = self.exec_namespace["behavior_analysis"]
        visual_manager = behavior_analysis.visual_manager
        out_folder = "event_clips"
        os.makedirs(out_folder, exist_ok=True)
        behavior_name = "_".join(query.split(" "))
        video_file = self.config["video_info"]["video_file_path"]
        return visual_manager.generate_video_clips_from_events(
            out_folder, video_file, events, behavior_name
        )

    def render_qa_message(self, qa_message):
        function_rets = qa_message["function_rets"]
        behavior_analysis = self.exec_namespace["behavior_analysis"]
        n_animals = behavior_analysis.animal_manager.get_n_individuals()
        bodypart_names = behavior_analysis.animal_manager.get_keypoint_names()
        visual_manager = behavior_analysis.visual_manager
        plots = []

        if isinstance(function_rets, tuple):
            # could be plotting tuple
            if isinstance(function_rets[0], plt.Figure):
                # this is for "return fig, ax"
                plots.append(function_rets)

            else:
                for e in function_rets:
                    if isinstance(e, list) and isinstance(e[0], BaseEvent):
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
                        qa_message["out_videos"].append(
                            self.events_to_videos(e, qa_message["query"])
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
            qa_message["out_videos"].append(
                self.events_to_videos(function_rets, qa_message["query"])
            )
        else:
            pass
        qa_message["plots"].extend(plots)
        return qa_message

    def llm_step(self, user_query):
        qa_message = create_message(user_query, self)
        self.messages.append(qa_message)
        post_process_llm = []  # ['self_debug', 'diagnosis']
        self.query = user_query
        self.llms["code_generator"].speak(self)
        return qa_message

    def step(self, user_query, number_of_debugs=1):
        qa_message = create_message(user_query, self)
        self.messages.append(qa_message)

        post_process_llm = ["self_debug"]
        self.query = user_query
        self.llms["code_generator"].speak(self)
        # all these llms collectively compose a amadeus_answer
        qa_message = self.code_execution(qa_message)

        if qa_message["error_message"] is not None:
            for i in range(number_of_debugs):
                self.llms["self_debug"].speak(self)
                qa_message = self.code_execution(qa_message)

        qa_message = self.render_qa_message(qa_message)

        return qa_message


def save_figure_to_tempfile(fig):
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
        # for event in events:
        #     print (event.start, event.end)
        # qa_message = sandbox.render_qa_message(qa_message)

    if qa_message["function_rets"] is not None:
        st.markdown(qa_message["function_rets"])

    plots = qa_message["plots"]
    for fig, axe in plots:
        filename = save_figure_to_tempfile(fig)
        st.image(filename, width=600)

    videos = qa_message["out_videos"]
    for video in videos:
        st.video(video)


if __name__ == "__main__":
    # testing qa message
    from amadeusgpt.analysis_objects.object import ROIObject
    from amadeusgpt.main import create_amadeus

    config = Config("amadeusgpt/configs/Horse_template.yaml")
    amadeus = create_amadeus(config)
    sandbox = amadeus.sandbox
    analysis = sandbox.exec_namespace["behavior_analysis"]
    analysis.add_roi_object = ""

    res = sandbox.step("plot the trajectory of the bodypart body_mouth")
