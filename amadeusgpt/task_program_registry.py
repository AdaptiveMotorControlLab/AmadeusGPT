
from typing import List, Dict, Any
import os
from amadeusgpt.analysis_objects import BaseEvent
from typing import List, Any
import ast
from amadeusgpt.utils import func2json
import json
from inspect import signature
import copy 
from numpy import ndarray
import typing
import traceback
from amadeusgpt.implementation import AnimalBehaviorAnalysis
from amadeusgpt.config import Config


required_classes = {
    'AnimalBehaviorAnalysis': AnimalBehaviorAnalysis,
    'Config': Config  
}
required_types = {name: getattr(typing, name) for name in dir(typing) if not name.startswith('_')}
required_types.update({'ndarray': ndarray})
class TaskProgram:
    """
    The task program in the system should be uniquely tracked by the id
    Attributes

    All the function body in the json_obj should look like following:

    The task program usually involve with a series of API calls and return a list of events
    The LLM agent should be able to craft, reuse task program for abitrary tasks

    def task_program_name(config) -> List[BaseEvent]: 
        analysis = AnimalBehaviorAnalysis(config)
        ... api callings
        return result

    The class should validate a few things:

    1) The function body should be a valid python function
    2) The function body should have the correct signature
    4) The function body should have and only have one input parameter which is the config    

    ----------
    the program attribute should keep the json obj describing the program
    Methods
    -------
    __call__(): should take the context and run the program in a sandbox.
    In the future we use docker container to run it 
    
    """    
    def __init__(self, json_obj: dict,
                 config: dict,
                 id: int,
                 creator: str = 'human'
                 ):
        """
        The task program requires the json obj for the code logic
        and the config file for the system so it knows which video and model to run stuff
        """

        self.json_obj = json_obj
        self.id = id
        # should somehow cache the result
        self.result_buffer = ''  
        self.creator = creator
        self.validate_function_body()
        # there are only two creators for the task program
        assert creator in ['human', 'llm_agent']
    


    def display(self):
        json_obj = copy.deepcopy(self.json_obj)
        json_obj['id'] = self.id
        json_obj['creator'] = self.creator
        temp = json.dumps(json_obj, indent=4)
        print (temp)

    def validate_function_body(self):
        function_body = self.json_obj['source_code']
        try:
            # Parse the function body
            tree = ast.parse(function_body)

            # Check if the body consists of a single function definition
            if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
                raise ValueError("Function body should contain a single function definition")

            # Get the function definition node
            function_def = tree.body[0]

            # Check if the function has the correct signature
            if len(function_def.args.args) != 1 or function_def.args.vararg or function_def.args.kwarg:
                raise ValueError("Function should have exactly one input parameter")
            
            # Check if the function takes a config parameter
            if function_def.args.args[0].arg != "config":
                raise ValueError("Function should take a config parameter")

        except SyntaxError as e:
            raise ValueError("Invalid function body syntax") from e

    def serialize(self):
        pass
    def deserialize(self):
        return super().deserialize()

    def validate(self):
        pass

    def __call__(self, *args, **kwargs) -> Any:
        """
        This is to execute the task program
        """
        # execute the code in a scope that has access to the AnimalBehaviorAnalysis class
                
        from amadeusgpt.implementation import AnimalBehaviorAnalysis
        from amadeusgpt.config import Config

        exec_namespace = {'__builtins__': __builtins__}
        exec_namespace.update(required_types)
        exec_namespace.update(required_classes)
        exec(self.json_obj['source_code'], exec_namespace)
                
        arguments = [repr(arg) for arg in args] + [f"{k}={repr(v)}" for k, v in kwargs.items()]
        arguments_str = ", ".join(arguments)

        # Construct the call string safely
        call_str = f"{self.json_obj['name']}({arguments_str})"
        try:
            exec(f"result = {call_str}", exec_namespace)
        except Exception as e:
            error_message = traceback.format_exc()
            print(error_message)  # Or log/store the error message as needed
        result = exec_namespace['result']
        self.result_buffer = result       
        return result        


class TaskProgramLibrary:
    """
    Keep track of the task programs
    There are following types of task programs:
    1) Custom task programs that are created by the user (can be loaded from disk)     
    2) Task programs that are created by LLMs
    

    """
    LIBRARY = {}    

    @classmethod
    def register_task_program(cls, config, creator='human'):
        def decorator(func):
            json_obj = func2json(func)
            id = len(cls.LIBRARY)
            task_program = TaskProgram(json_obj, 
                                        config,
                                         id,  creator=creator)
            cls.LIBRARY[func.__name__] = task_program
            return func  # It's common to return the original function unmodified
        return decorator
              
    @classmethod
    def get_task_programs(cls):
        """
        Get the task programs
        """
        return cls.LIBRARY

    # def save_task_programs(self):
    #     """
    #     Save the task programs to the disk
    #     """
    #     ret = []
    #     for task_program in self.task_programs:
    #         ret.append(task_program.json_obj)
    #     save_path = self.config['task_programs']['save_path']
    #     with open(save_path, 'w') as f:
    #         f.write(ret)
        

    # def save_human_task_programs(self):
    #     """
    #     Save to to dict, according to the config file
    #     """
    #     pass

    # def save_llm_task_programs(self): 
    #     """
        
    #     """
    #     pass

    # def load_human_task_programs(self):
    #     pass

    # def load_llm_task_programs(self):
    #     pass       
        
