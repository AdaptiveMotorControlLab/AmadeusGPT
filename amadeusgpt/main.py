import warnings
from amadeusgpt.config import Config
from amadeusgpt.programs.sandbox import Sandbox
##########
# all these are providing the customized classes for the code execution
##########
from amadeusgpt.utils import *
warnings.filterwarnings("ignore")
from amadeusgpt.analysis_objects.llm import (CodeGenerationLLM,
                                             SelfDebugLLM, VisualLLM)
from amadeusgpt.integration_module_hub import IntegrationModuleHub

from amadeusgpt.integration_module_hub import IntegrationModuleHub
from amadeusgpt.programs.task_program_registry import TaskProgramLibrary
from pathlib import Path
import glob 
class AMADEUS:
    def __init__(self, config: Config):
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
        data_info = config['data_info']
        self.result_folder: str = data_info['result_folder']

        data_folder = Path(data_info['data_folder'])
        video_suffix = data_info['video_suffix']
        video_file_paths = glob.glob(str(data_folder / f'*{video_suffix}'))

        # optionally get the corresponding keypoint files
        keypoint_file_paths = self.get_superanimal_keypoint_files(video_file_paths)
        if len(keypoint_file_paths) == 0:
            keypoint_file_paths = [""] * len(video_file_paths)
        else:
            assert len(video_file_paths) == len(keypoint_file_paths), "The number of video files and keypoint files should be the same"
        

        self.sandbox = Sandbox(config, 
                               video_file_paths, 
                               keypoint_file_paths)
        
        self.code_generator_llm = CodeGenerationLLM(config.get("llm_info", {}))
        self.self_debug_llm = SelfDebugLLM(config.get("llm_info", {}))
        self.visual_llm = VisualLLM(config.get("llm_info", {}))

        ####

        ## register the llm to the sandbox
        
        self.sandbox.register_llm("code_generator", self.code_generator_llm)
        self.sandbox.register_llm("visual_llm", self.visual_llm)
        if self.use_self_debug:
            self.sandbox.register_llm("self_debug", self.self_debug_llm)
        

        # can only do this after the register process
        # self.sandbox.configure_using_vlm()

    def get_superanimal_keypoint_files(self, video_file_paths: list[str]):
        ret = []
        # how to get the filename from the path file

        for video_file_path in video_file_paths:
            keypoint_file_path = video_file_path.replace(".mp4", "_superanimal_topviewmouse_hrnetw32.h5")
            ret.append(keypoint_file_path)
        
        return ret

    def match_integration_module(self, user_query: str):
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
       

    def step(self, user_query:str)-> QA_Message:
        integration_module_names = self.match_integration_module(user_query)

        self.sandbox.update_matched_integration_modules(integration_module_names)
        qa_message = self.sandbox.llm_step(user_query)

        return qa_message

    def get_analysis(self, video_file_path: str):
        """
        Every sandbox stores a unique "behavior analysis" instance in its namespace
        Therefore, get analysis gets the current sandbox's analysis.
        """
        analysis = self.sandbox.namespace_dict[video_file_path]

        return analysis

    def run_task_program(self, 
                         config: Config, 
                         task_program_name: str):
        """
        Execute the task program on the currently holding sandbox
        Parameters
        -----------
        config: a config specifies the movie file and the keypoint file to run task program
        task_program_name: the name of the task program to run

        """
        return self.sandbox.run_task_program(task_program_name)
    
    # def save_results(self, out_folder: str| None = None):
    #     """
    #     Save the results of the qa message (since it has all the information needed)
    #     """
    #     # if out_folder is None:
    #     #     result_folder = self.sandbox.result_folder
    #     # else:
    #     #     result_folder = out_folder
    #     # # make sure it exists
    #     # os.makedirs(result_folder, exist_ok=True)
    #     # results = self.sandbox.result_cache

    #     if out_folder is None:
    #         result_folder = self.result_folder
    #     else:
    #         result_folder = out_folder
    #     # make sure it exists
    #     os.makedirs(result_folder, exist_ok=True)

    #     ret = defaultdict(dict)
    #     for video_file_path in self.sandboxes:
    #         result = self.sandboxes[video_file_path].result_cache
    #         for query in result:
    #             ret[video_file_path][query] = result[query].get_serializable()                

    #     # save results to a pickle file
    #     with open (os.path.join(result_folder, "results.pickle"), "wb") as f:
    #         pickle.dump(ret, f)

    # def load_results(self, result_folder: str | None = None ):
    #     if result_folder is None:
    #         result_folder = self.result_folder
    #     else:
    #         result_folder = result_folder
        
    #     with open (os.path.join(result_folder, "results.pickle"), "rb") as f:
    #         results = pickle.load(f)

    #     for video_file_path in results:
    #         self.sandboxes[video_file_path].result_cache = results[video_file_path]

    def register_task_program(self, task_program, creator = "human"):
        TaskProgramLibrary.register_task_program(creator = creator)(task_program)

    # def get_results(self, video_file_path: str):
        
    #     return self.sandboxes[video_file_path].result_cache
    
    def get_task_programs(self):
        return TaskProgramLibrary.get_task_programs()

if __name__ == "__main__":
    from amadeusgpt.analysis_objects.llm import VisualLLM
    from amadeusgpt.config import Config
    config = Config("amadeusgpt/configs/EPM_template.yaml")
    amadeus = AMADEUS(config)   
