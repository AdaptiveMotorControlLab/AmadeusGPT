import warnings
import json
from amadeusgpt.programs import task_program_registry
from amadeusgpt.config import Config
from amadeusgpt.sandbox import Sandbox
import openai
import amadeusgpt

from amadeusgpt.middle_end import AmadeusAnswer
from amadeusgpt.utils import parse_error_message_from_python


##########
# all these are providing the customized classes for the code execution
from amadeusgpt.implementation import (
    AnimalBehaviorAnalysis,  
)
#from amadeusgpt.modules.implementation import *

##########

from amadeusgpt.utils import *
from amadeusgpt.module_matching import match_module
from amadeusgpt.logger import AmadeusLogger
#from amadeusgpt.middle_end import AmadeusAnswer

warnings.filterwarnings("ignore")
from amadeusgpt.analysis_objects.llm import CodeGenerationLLM, SelfDebugLLM, DiagnosisLLM 

        
class AMADEUS:
    def __init__(self, config: Dict[str, Any]):
        self.config = config    
        # functionally different llms
        self.code_generator_llm = CodeGenerationLLM(config)
        self.self_debug_llm = SelfDebugLLM(config)
        self.diagnosis_llm = DiagnosisLLM(config)  
        ### fields that decide the behavior of the application
        self.use_self_debug = False
        self.use_diagnosis = False        
        self.behavior_modules_in_context = True
        self.smart_loading = False        
        self.load_module_top_k = 3
        self.module_threshold = 0.7
        self.enforce_prompt = "#"
        self.code_generator_llm.enforce_prompt = ""  
        ### fileds that serve as important storage    
        self.context_window_dict = {}
        self.behavior_modules_str = ""
        ####
        from amadeusgpt.sandbox import Sandbox
        from amadeusgpt.programs.task_program_registry import TaskProgramLibrary
        from amadeusgpt.api_registry import CORE_API_REGISTRY
        task_programs = TaskProgramLibrary().get_task_programs()
                
        self.sandbox = Sandbox(
            config,
            task_programs,
            CORE_API_REGISTRY)
        ####

        ## register the llm to the mediator
        self.sandbox.register_llm(self.code_generator_llm)    
        if self.use_self_debug:
            self.sandbox.register_llm(self.self_debug_llm)        
        if self.use_diagnosis:
            self.sandbox.register_llm(self.diagnosis_llm)
                            
    def step(self, user_query):       
        self.sandbox.step(user_query)        
         
   

if __name__ == "__main__":
    from amadeusgpt.programs.task_program_registry import TaskProgramLibrary
    from amadeusgpt.api_registry import CORE_API_REGISTRY
    from amadeusgpt.implementation import AnimalBehaviorAnalysis

    @TaskProgramLibrary.register_task_program(creator="human")
    def get_speeding_events(config):
        """
        Get events where animals are speeding
        """
        analysis = AnimalBehaviorAnalysis(config)
        speed_events = analysis.event_manager.get_animals_state_events("speed", ">=5")        
        return speed_events

    config = Config('../tests/nishant.yaml')
    amadeus = AMADEUS(config)
    amadeus.step("I want to get behavior of the animal watching the other animals while speeding ")