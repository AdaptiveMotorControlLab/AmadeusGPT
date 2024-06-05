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
        ### fileds that serve as important storage    
        self.context_window_dict = {}
        self.behavior_modules_str = ""
        ####        
                
        self.sandbox = Sandbox(
            config)
          
        ####

        ## register the llm to the mediator
        self.sandbox.register_llm('code_generator', self.code_generator_llm)    
        if self.use_self_debug:
            self.sandbox.register_llm('self_debug', self.self_debug_llm)        
        if self.use_diagnosis:
            self.sandbox.register_llm('diagnosis', self.diagnosis_llm)

    def chat_iteration(self, user_query):
        qa_message = self.sandbox.llm_step(user_query)
        return qa_message

    def step(self, user_query):       
        result = self.sandbox.llm_step(user_query)        
        return result
         
   

if __name__ == "__main__":
    from amadeusgpt.programs.task_program_registry import TaskProgramLibrary
    from amadeusgpt.programs.api_registry import CORE_API_REGISTRY
   
    config = Config('../tests/nishant.yaml')
    amadeus = AMADEUS(config)
    amadeus.step("I want to get behavior of the animal watching the other animals while speeding ")