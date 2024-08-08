import warnings

from amadeusgpt.config import Config
from amadeusgpt.programs.sandbox import Sandbox
##########
# all these are providing the customized classes for the code execution
##########
from amadeusgpt.utils import *

warnings.filterwarnings("ignore")
import glob
import os
import pickle
from pathlib import Path

from amadeusgpt.analysis_objects.llm import (CodeGenerationLLM, SelfDebugLLM,
                                             VisualLLM)
from amadeusgpt.behavior_analysis.identifier import Identifier
from amadeusgpt.integration_module_hub import IntegrationModuleHub
from amadeusgpt.programs.task_program_registry import TaskProgramLibrary


class AMADEUS:
    def __init__(self, config: Config | dict, use_vlm=False, movie2keypoint_func=None):
        self.config = config
        ### fields that decide the behavior of the application
        self.use_self_debug = True
        self.use_diagnosis = False
        self.use_behavior_modules_in_context = True
        self.smart_loading = False
        self.load_module_top_k = 3
        self.module_threshold = 0.3
        ### fields that serve as important storage
        # for long-term memory
        self.integration_module_hub = IntegrationModuleHub()

        ### For the sake of multiple animal, we store multiple sandboxes
        ### the example {video_file_path : sandbox }
        self.result_folder: str = config["data_info"]["result_folder"]
        self.data_folder = config["data_info"]["data_folder"]
        self.video_suffix = config["data_info"]["video_suffix"]

        self.video_file_paths, self.keypoint_file_paths = (
            self.fetch_data_from_data_folder(movie2keypoint_func)
        )

        self.sandbox = Sandbox(config, self.video_file_paths, self.keypoint_file_paths)

        self.code_generator_llm = CodeGenerationLLM(config)
        self.self_debug_llm = SelfDebugLLM(config)
        self.visual_llm = VisualLLM(config)

        ####

        ## register the llm to the sandbox

        self.sandbox.register_llm("code_generator", self.code_generator_llm)
        self.sandbox.register_llm("visual_llm", self.visual_llm)
        if self.use_self_debug:
            self.sandbox.register_llm("self_debug", self.self_debug_llm)

        # can only do this after the register process
        if use_vlm:
            self.sandbox.configure_using_vlm()

    def fetch_data_from_data_folder(self, movie2keypoint_func):
        """
        1) video file exists, keypoint file does not exist
        2) video file and keypoint file both exist and one-to-one mapping
        3) many video files and one keypoint file (many-to-one mapping), perhaps for 3D data
        4) none of video file and keypoint file exist
        5) only keypoint file exists

        Returns
        -------
        video_file_paths: list[str]
            a list of video file paths
        keypoint_file_paths: list[str]
            a list of keypoint file paths

        """
        video_files = glob.glob(os.path.join(self.data_folder, f"*{self.video_suffix}"))
        video_files = [
            video_file for video_file in video_files if "labeled" not in video_file
        ]
        keypoint_files = glob.glob(os.path.join(self.data_folder, "*.h5"))
        # case 1
        if movie2keypoint_func is None:
            if len(video_files) > 0 and len(keypoint_files) == 0:
                return video_files, [""] * len(video_files)
            # case 2
            elif len(video_files) > 0 and len(video_files) == len(keypoint_files):
                return video_files, self.get_DLC_keypoint_files(video_files)
            # case 3
            elif len(video_files) > 1 and len(keypoint_files) == 1:
                print(
                    "We assume this is 3D data with multiple video files and one keypoint file"
                )
                return video_files, [keypoint_files[0]] * len(video_files)
            # case 4
            elif len(video_files) == 0 and len(keypoint_files) == 0:
                raise ValueError(
                    "No video files and keypoint files found in the data folder"
                )
            # case 5
            elif len(video_files) == 0 and len(keypoint_files) > 0:
                print("No video found. We proceed with the keypoint file only")
                return [""] * len(keypoint_files), keypoint_files
            else:
                return video_files, keypoint_files
        else:
            if len(video_files) > 0:
                return video_files, [
                    movie2keypoint_func(video_file) for video_file in video_files
                ]
            else:
                raise ValueError("No video files found in the data folder")

    def get_DLC_keypoint_files(self, video_file_paths: list[str]):
        ret = []
        # how to get the filename from the path file
        video_folder = Path(video_file_paths[0]).parent
        for video_file_path in video_file_paths:
            videoname = Path(video_file_path).stem
            if len(glob.glob(str(video_folder / f"{videoname}*.h5"))) > 0:
                keypoint_file_path = glob.glob(str(video_folder / f"{videoname}*.h5"))[
                    0
                ]
            else:
                keypoint_file_path = ""
            ret.append(keypoint_file_path)

        return ret

    def match_integration_module(self, user_query: str) -> list:
        """
        Return a list of matched integration modules
        """
        sorted_query_results = self.integration_module_hub.match_module(user_query)
        if len(sorted_query_results) == 0:
            return None
        modules = []
        for i in range(min(self.load_module_top_k, len(sorted_query_results))):
            query_result = sorted_query_results[i]
            query_module = query_result[0]
            query_score = query_result[1][0][0]
            if query_score > self.module_threshold:
                modules.append(query_module)

                # parse the query result by loading active loading
        return modules

    def step(self, user_query: str) -> QA_Message:
        integration_module_names = self.match_integration_module(user_query)

        self.sandbox.update_matched_integration_modules(integration_module_names)
        qa_message = self.sandbox.llm_step(user_query)

        return qa_message

    def get_video_file_paths(self) -> list[str]:
        return self.video_file_paths

    def get_keypoint_file_paths(self) -> list[str]:
        return self.keypoint_file_paths

    def get_behavior_analysis(
        self, video_file_path: str = "", keypoint_file_path: str = ""
    ):
        """
        Every sandbox stores a unique "behavior analysis" instance in its namespace
        Therefore, get analysis gets the current sandbox's analysis.
        """
        identifier = Identifier(self.config, video_file_path, keypoint_file_path)
        analysis = self.sandbox.namespace_dict[identifier]["behavior_analysis"]

        return analysis

    def run_task_program(self, task_program_name: str):
        """
        Execute the task program on the currently holding sandbox
        Parameters
        -----------
        config: a config specifies the movie file and the keypoint file to run task program
        task_program_name: the name of the task program to run

        """
        return self.sandbox.run_task_program(task_program_name)

    def register_task_program(self, task_program, creator="human"):
        TaskProgramLibrary.register_task_program(creator=creator)(task_program)

    def get_messages(self):
        return self.sandbox.message_cache

    def get_task_programs(self):
        return TaskProgramLibrary.get_task_programs()


if __name__ == "__main__":
    from amadeusgpt.analysis_objects.llm import VisualLLM
    from amadeusgpt.config import Config

    config = Config("amadeusgpt/configs/EPM_template.yaml")
    amadeus = AMADEUS(config)
