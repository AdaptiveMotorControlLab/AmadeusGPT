import warnings
from amadeusgpt.config import Config

##########
# all these are providing the customized classes for the code execution
from amadeusgpt.programs.sandbox import Sandbox
##########
from amadeusgpt.utils import *
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
                
        self.sandbox = Sandbox(
            config)
          
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
    from amadeusgpt.programs.api_registry import CORE_API_REGISTRY
   
    config = Config('../tests/nishant.yaml')
    amadeus = AMADEUS(config)
    amadeus.step("I want to get behavior of the animal watching the other animals while speeding ")