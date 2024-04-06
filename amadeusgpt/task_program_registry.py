
from typing import Callable, List, Dict, Any
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
    def __init__(self, 
                 json_obj: dict,
                 id: int,
                 creator: str = 'human',
                 parents = None,
                 mutation_from = None,
                 generation = 0
                 ):
        """
        The task program requires the json obj for the code logic
        and the config file for the system so it knows which video and model to run stuff
        """

        self.json_obj = json_obj 
        if self.json_obj['source_code'] is not None:
            self.validate_function_body()
        # there are only two creators for the task program
        assert creator in ['human', 'llm']
       
        self.json_obj['id'] = id
        self.json_obj['creator'] = creator
        self.json_obj['parents'] = parents
        self.json_obj['mutation_from'] = mutation_from
        self.json_obj['generation'] = generation

    
    def __setitem__(self, key, value):
        self.json_obj[key] = value

    def __getitem__(self, key):
        '''
        {
            'name': ''
            'inputs': '',
            'source_code': ''
            'docstring': ''
            'return': ''
        }
        '''
        return self.json_obj[key]

  

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

    def __call__(self, config, namespace) -> Any:        
        function_name = self.json_obj['name']
        if self.json_obj['source_code'] is not None:
            exec(self.json_obj['source_code'], namespace)
            
            function = namespace[function_name]
        else:
            assert self.json_obj['func_pointer'] is not None
            function = self.json_obj['func_pointer']
            namespace[function_name] = function
        call_str = f"{function_name}(config)"
        exec(f"result = {call_str}", namespace)
        result = namespace['result']

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
    def register_task_program(cls, 
                              creator='human', 
                              parents = None, 
                              mutation_from = None):
        # we need to add the relationship for the created
        # task program
        def decorator(func):
            if isinstance(func, Callable) and not isinstance(func, TaskProgram):
                json_obj = func2json(func)
                id = len(cls.LIBRARY)
                task_program = TaskProgram(json_obj, 
                                            id,  
                                            creator=creator,
                                            parents=parents,
                                            mutation_from=mutation_from)              
                cls.LIBRARY[json_obj['name']] = task_program
                return func  # It's common to return the original function unmodified
           
            elif isinstance(func, dict):               
                data_json = func 
                task_program = TaskProgram(data_json, 
                                            len(cls.LIBRARY),
                                            creator=creator,
                                            parents=parents,
                                            mutation_from=mutation_from)
                
                cls.LIBRARY[data_json['name']] = task_program
                return task_program
            elif isinstance(func, str):
                json_obj = func2json(func)
                id = len(cls.LIBRARY)
                task_program = TaskProgram(json_obj, 
                                            id,  
                                            creator=creator,
                                            parents=parents,
                                            mutation_from=mutation_from)
                cls.LIBRARY[json_obj['name']] = task_program
        return decorator
              
    @classmethod
    def get_task_programs(cls):
        """
        Get the task programs
        """
        return cls.LIBRARY
    
    @classmethod
    def show_all(cls):
        for name, task_program in cls.LIBRARY.items():
            print(task_program['name'])
            print(task_program['docstring'])
    
    @classmethod
    def save(cls, out_path):
        ret = []
        for name, task_program in cls.LIBRARY.items():
            if 'func_pointer' in task_program.json_obj:
                task_program.json_obj['func_pointer'] = None

            ret.append(task_program.json_obj)

        with open(out_path, 'w') as f:
            json.dump(ret, f, indent = 4)