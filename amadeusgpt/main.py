import warnings

from amadeusgpt.config import Config
##########
# all these are providing the customized classes for the code execution
from amadeusgpt.programs.sandbox import Sandbox
##########
from amadeusgpt.utils import *

warnings.filterwarnings("ignore")
import os

from amadeusgpt.analysis_objects.llm import (CodeGenerationLLM, DiagnosisLLM,
                                             SelfDebugLLM)
from amadeusgpt.integration_module_hub import IntegrationModuleHub

amadeus_fac = {}


# using the config file to cache the amadeus instance
# not sure if this is the best practice
def create_amadeus(config: Config):
    if str(config) not in amadeus_fac:
        amadeus_fac[str(config)] = AMADEUS(config)
    return amadeus_fac[str(config)]


class AMADEUS:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.code_generator_llm = CodeGenerationLLM(config.get("llm_info", {}))
        self.self_debug_llm = SelfDebugLLM(config.get("llm_info", {}))
        self.diagnosis_llm = DiagnosisLLM(config.get("llm_info", {}))
        ### fields that decide the behavior of the application
        self.use_self_debug = True
        self.use_diagnosis = False
        self.use_behavior_modules_in_context = True
        self.smart_loading = False
        self.load_module_top_k = 3
        self.module_threshold = 0.3
        ### fileds that serve as important storage
        # for long-term memory
        self.integration_module_hub = IntegrationModuleHub(config)
        #### sanbox that actually takes query and executes the code
        self.sandbox = Sandbox(config)
        ####

        ## register the llm to the sandbox
        self.sandbox.register_llm("code_generator", self.code_generator_llm)
        if self.use_self_debug:
            self.sandbox.register_llm("self_debug", self.self_debug_llm)
        if self.use_diagnosis:
            self.sandbox.register_llm("diagnosis", self.diagnosis_llm)

    def match_integration_module(self, user_query: str):
        """
        Return a list of matched integration modules
        """
        sorted_query_results = self.integration_module_hub.match_module(user_query)
        if len(sorted_query_results) == 0:
            return None
        modules = []
        print("sorted_query_results", sorted_query_results)
        for i in range(min(self.load_module_top_k, len(sorted_query_results))):
            query_result = sorted_query_results[i]
            query_module = query_result[0]
            query_score = query_result[1][0][0]
            if query_score > self.module_threshold:
                modules.append(query_module)

                # parse the query result by loading active loading
        return modules

    def chat_iteration(self, user_query):
        qa_message = self.sandbox.llm_step(user_query)
        return qa_message

    def step(self, user_query):
        integration_module_names = self.match_integration_module(user_query)
        self.sandbox.update_matched_integration_modules(integration_module_names)
        result = self.sandbox.llm_step(user_query)
        return result


if __name__ == "__main__":
    config = Config("amadeusgpt/configs/MausHaus_template.yaml")

    # amadeus = AMADEUS(config)
    # query = "Give me events when mice are close"
    # amadeus.step(query)

    query = "Plot the trajectory with the keypoint butt"
    amadeus = create_amadeus(config)
    sandbox = amadeus.sandbox
    analysis = sandbox.exec_namespace["behavior_analysis"]

    analysis.object_manager.load_roi_objects("temp_roi_objects.pickle")

    from amadeusgpt.programs.sandbox import render_temp_message

    render_temp_message(query, sandbox)
