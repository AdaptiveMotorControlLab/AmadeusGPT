from amadeusgpt import task_program_registry
from amadeusgpt.analysis_factory import create_analysis
from amadeusgpt.system_prompts import mutation
from amadeusgpt.task_program_registry import TaskProgram, TaskProgramLibrary
from functools import wraps
import inspect
import re



class ChatChannel:
    """
    All llms are supposed to check and update the chat channel
    """

    def __init__(self):
        self.reflection = [None]
        self.error_signal = [None]
        self.code_history = [None]
        self.chain_of_thought = [None]
        self.analysis_result = [None]
        self.user_query = [None]
        self.error_diagnosis = [None]
        self.task_program_description = [None]
        self.terminate = False

    def add_user_query(self, user_query):
        self.user_query.append(user_query)
    
    def get_last_message(self):      
        return {
            "reflection": self.reflection[-1],
            "error_signal": self.error_signal[-1],
            "code_history": self.code_history[-1],
            "chain_of_thought": self.chain_of_thought[-1],
            "analysis_result": self.analysis_result[-1],
            "user_query": self.user_query[-1],
            "error_diagnosis": self.error_diagnosis[-1],
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
    def __init__(self,
                 api_registry):      

        self.api_registry = api_registry

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
            if type_name == 'inspect._empty':
                return 'None'
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
        lines = text.split('\n')
        adjusted_lines = []

        for line in lines:
            # Count the leading spaces
            leading_spaces = len(line) - len(line.lstrip(' '))
            # Calculate the new indentation level, assuming the smallest non-zero
            # indentation level in the original text is one level.
            new_indentation = (leading_spaces // spaces_per_indent) * spaces_per_indent
            adjusted_line = ' ' * new_indentation + line.lstrip(' ')
            adjusted_lines.append(adjusted_line)

        return '\n'.join(adjusted_lines)

    def _fill_parameters(self, param_dict):                   

        ret = ""
        for i, (name, type) in enumerate(param_dict.items()):
            if name == 'self':
                continue
            if type == "<class 'inspect._empty'>":
                ret += f"{name}: {self._map_type(type)}"
            else:
                ret += f"{name}: {self._map_type(type)}"
            if i < len(param_dict) - 2:
                ret += ", "
        return ret 

    def get_core_api_docs(self):
        """
        Turn the core api docs into a format that GPT can understand
        """
        ret = "```coreapidocs\n"
        for name, api in self.api_registry.items():
            description = api['description']
            # parameters is a dictionary that might contain self
            parameters = self._fill_parameters(api['parameters'])
            description = self.enforce_indentation(description)
            ret += f"{name}({parameters}): \n{description}\n"

        ret += "\n```"  

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
    def __init__(self, 
                config, 
                api_registry):                
        super().__init__(api_registry)
        self.task_program_library = TaskProgramLibrary().get_task_programs()
        self.config = config
        self.chat_channel = ChatChannel()
        self.exec_namespace = {'__builtins__': __builtins__}
        self.update_namespace()
    

    def get_task_program_docs(self):
        ret = "```taskprograms\n"
        for name, task_program in self.task_program_library.items():
            description = task_program.json_obj['docstring']
            ret +=f"{name}(config: Config): \n{description}\n"
        ret += "\n```"

        return ret

    def get_user_query(self):
        return self.chat_channel.get_last_message()['user_query']
  
    
    def update_namespace(self):
        # we need to manage the scope of the session
        # there are potentially new variables, new task programs, new apis
        analysis = create_analysis(self.config)
        for api in self.api_registry.values():    
            f = wrap_instance_method(analysis, api['name']) 
            self.exec_namespace[api['name']] = f

        for name, task_program in self.task_program_library.items():
            self.exec_namespace[name] = task_program

        current_scope = globals()
        
        for name, value in current_scope.items():
            if callable(value) or isinstance(value, (int, float, str, list, dict, tuple)):
                self.exec_namespace[name] = value
        self.exec_namespace['config'] = self.config
        from amadeusgpt.config import Config
        self.exec_namespace['Config'] = Config

        from amadeusgpt.implementation import AnimalBehaviorAnalysis
        self.exec_namespace['AnimalBehaviorAnalysis'] = AnimalBehaviorAnalysis
        from amadeusgpt.analysis_objects.relationship import Orientation
        self.exec_namespace['Orientation'] = Orientation        

    def code_execution(self):
        code = self.chat_channel.get_last_message()['code_history']

        # add main function into the namespace        
        self.update_namespace()
        exec(code, self.exec_namespace)
        # call the main function
        call_str = f"main(config)"

        exec(f"result = {call_str}",self.exec_namespace)
        result = self.exec_namespace['result']

    
    def register_task_program(self,
                            code, 
                            parents = None, 
                            mutation_from = None):
        self.update_namespace()

        if isinstance(code, str):
            exec(code,  globals())
            TaskProgramLibrary.register_task_program(creator="llm", 
                                                     parents = parents,
                                                     mutation_from = mutation_from)(code)

        elif isinstance(code, TaskProgram):
            TaskProgramLibrary.register_task_program(creator="llm", 
                                                     parents = parents,
                                                     mutation_from = mutation_from)(code)

        elif isinstance(code, dict):
            TaskProgramLibrary.register_task_program(creator="llm", 
                                                     parents = parents,
                                                     mutation_from = mutation_from)(code)



    def step(self, user_query):    
        self.chat_channel.add_user_query(user_query)
        for llm_name, llm in self.llms.items():
            if llm.whether_speak(self.chat_channel):                
                llm.speak(self)
        self.code_execution()

class EvoSandbox(Sandbox):
    def __init__(self,
                config, 
                api_registry): 
        super().__init__(config, api_registry)
        self.task_program_library = TaskProgramLibrary().get_task_programs()
        # name of the program
        self.program_to_mutate = None
        self.breed_info = None
  
    def update_program_to_mutate(self, program_name):
        self.program_to_mutate = program_name

    def update_breed_info(self, task_program1_docs, task_program2_docs, composition_type):

        self.breed_info = (task_program1_docs, task_program2_docs, composition_type)

    def get_breed_info(self):
        return self.breed_info


    def get_task_program_docs(self):
        ret = "```taskprograms\n"
        for name, task_program in self.task_program_library.items():            
            description = task_program.json_obj['docstring']
            ret +=f"{name}(config): \n{description}\n"
        ret += "\n```"
        return ret
    
    def get_mutation_task_program(self):
        ret = "```task_program_to_be_changed\n"
        task_program = self.task_program_library[self.program_to_mutate]
        description = task_program.json_obj['docstring']
        ret +=f"{self.program_to_mutate}(config): \n{description}\n"
        ret += "\n```"
        return ret


if __name__ == '__main__':
    from amadeusgpt.config import Config
    config = Config('../tests/MABe.yaml')
    analysis = create_analysis(config)
    get_animals_animals_events = wrap_instance_method(analysis, "get_animals_animals_events")
    
    def main(config):               
        # Define a threshold for the relative head angle that indicates "watching"
        watching_angle_threshold = 30  # degrees
        
        # Get events where the relative head angle between two animals is less than the threshold
        watching_events = get_animals_animals_events(
            ['relative_head_angle'],
            [f'<={watching_angle_threshold}'],
            bodypart_names=['head'],  # Assuming 'head' is a valid bodypart name
            otheranimal_bodypart_names=None,  # We are interested in the head of the first animal only
            min_window=1,  # Minimum duration for an event to be considered
            max_window=100,  # Maximum duration for an event to be considered
            smooth_window_size=1  # No smoothing needed
        )
        
        # Get events where the animals are moving at a speed of at least 2 units per second        
        return watching_events
    main(config)