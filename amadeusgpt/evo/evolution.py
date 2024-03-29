
from amadeusgpt.system_prompts import mutation
from amadeusgpt.task_program_registry import TaskProgram, TaskProgramLibrary
from amadeusgpt.sandbox import EvoSandbox
from amadeusgpt.api_registry import CORE_API_REGISTRY
from amadeusgpt.analysis_objects.llm import MutationLLM, CodeGenerationLLM, BreedLLM
import re
import random
import ast
from itertools import combinations
import functools
from functools import partial
import time
import os
random.seed(time.time() * os.getpid())

def preset_args(**preset_kwargs):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Combine preset arguments with any additional arguments provided
            combined_kwargs = {**preset_kwargs, **kwargs}
            return func(*args, **combined_kwargs)
        return wrapper
    return decorator


def breed_program(config, events1, events2, composition_type='logical_and'):
    analysis = AnimalBehaviorAnalysis(config)
    ret_events = analysis.event_manager.get_composite_events([events1, events2], composition_type = composition_type)
    return ret_events

class BaseEvolution:
    """
    Keep a reference to task program library

    Keep an internal look up table for task programs.
     
    evaluate: evaluate the fitness of each individuals

    select: Select individuals for breeding

    mutate: Mutate an existing task program using LLM

    breed: breed two task programs. LLM is used for new naming and description

    """


    def __init__(self, config):
        self.config = config

    def evaluate(self):
        pass

    def select_for_breed(self):
        pass

    def select_for_mutation(self):
        pass

    def breed(self):
        pass    

    def mutate(self, task_program):
        # ask a mutation LLM to mutate a task program description        
        pass

    def rank(self):
        pass

    def step(self):
        
        # rank

        # select for mutation

        # mutate

        # select for breeding

        # breed

        # evaluation

        pass
    def train(self):
        pass


class EasyEvolution(BaseEvolution):
    def __init__(self, config):
        super().__init__(config)
        self.mutation_llm = MutationLLM(config)
        self.code_generation_llm = CodeGenerationLLM(config)
        self.breed_llm = BreedLLM(config)
        self.task_program_library = TaskProgramLibrary().get_task_programs()
        # keep track of the fitness scores of each task program
        self.scores = {}
        # cache the results of task programs in scores
        self.cache = {}
        self.sandbox = EvoSandbox(
            config,
            CORE_API_REGISTRY)

    def events_duration(self, events):
        temp = [event.duration_in_seconds for event in events]
        return sum(temp)

    def mutate(self):        
        mutation_response = self.mutation_llm.speak(self.sandbox)
        # get the mutated task program
        pattern = r"```python(.*?)```"
        function_code = re.findall(pattern, mutation_response, re.DOTALL)[0]
        print ('mutated function')
        print (function_code)
        parent_generation = TaskProgramLibrary[self.sandbox.program_to_mutate]['generation']
        self.sandbox.register_task_program(function_code,
                                           mutation_from = self.sandbox.program_to_mutate,
                                           generation = parent_generation+1)
    
    def get_breed_function(self):
        breed_program.__globals__.update(self.sandbox.exec_namespace)
        return breed_program
        #return breed_program(self.config, events1, events2, composition_type)

    def breed(self, participans):
        pairs = list(combinations(participans, 2))
        composition_types = ['logical_and', 'sequential']
        for pair in pairs:
            for composition_type in composition_types:
                name1, name2  = pair[0], pair[1]
                events1, events2 = self.cache[name1], self.cache[name2]
                docs1, docs2 = self.task_program_library[name1]['docstring'], self.task_program_library[name2]['docstring']
                breed_func = self.get_breed_function()#events1, events2, composition_type=composition_type)
                breed_func = preset_args(events1 = events1, events2 = events2, composition_type = composition_type)(breed_func)
                new_events = breed_func(self.config)
                # inspect the source code for the breed function

                # get the new description from the llm
                self.sandbox.update_breed_info(docs1, docs2, composition_type)
                template_function = self.breed_llm.speak(self.sandbox)
                template_function = template_function.replace('<template>', 'return []')                
                pattern = r"```python(.*?)```"
                function_code = re.findall(pattern, template_function, re.DOTALL)[0]
                ast_info = ast.parse(function_code)
                # get function name from ast_info
                function_name = ast_info.body[0].name
                docstring = ast.get_docstring(ast_info.body[0])

                json_obj = {
                    'name': function_name,
                    'source_code': None,
                    'docstring': docstring,
                    'func_pointer': partial(breed_func)
                }

                self.sandbox.register_task_program(json_obj,
                                                   parents=[name1, name2])

       
    def select_for_mutation(self):
        survivals = list(self.scores.keys())
        # generate a random number from 0 to len(survivals)
        # return the task program name
        index = random.randint(0, len(survivals)-1)     
        return survivals[index]

    def select_for_breed(self, num_participants = 2):
        survivals = list(self.scores.keys())
        # select num_participants from survivals
        indices = random.sample(range(0, len(survivals)), num_participants)
        return [survivals[i] for i in indices]    


    def train(self):
        self.evaluate() 
        for i in range(2):
            print (f'training iteration {i}')
            mutate_candidate = self.select_for_mutation()
            print (f'selecting {mutate_candidate} {self.scores[mutate_candidate]} for mutation')
            self.sandbox.update_program_to_mutate(mutate_candidate)
            self.mutate()            
            breed_participants = self.select_for_breed()
            print (f'selecting {breed_participants} for breeding')
            self.breed(breed_participants)
            self.evaluate()                        

    def evaluate(self):
        # run each task program and assign the fitness score
        # let's check whether this works first
        for name, task_program in self.task_program_library.items():           
            if name not in self.scores:
                # call the function
                events = task_program(self.config, self.sandbox.exec_namespace)
                if self.events_duration(events) > 1:
                    task_program['duration'] = round(self.events_duration(events), 2)
                    self.scores[name] = round(self.events_duration(events), 2)
                    self.cache[name] = events
                    print (task_program.json_obj)

            print ('scores', self.scores)
if __name__ == "__main__":
    from amadeusgpt.config import Config
    from amadeusgpt.common_task_programs import register_common_task_programs
    config = Config('../../tests/MABe.yaml')
    register_common_task_programs()
    evo = EasyEvolution(config)
    evo.train()
    TaskProgramLibrary.save('task_program_checkpoint.json')
    