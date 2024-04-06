
from dis import disco
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
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob
from collections import defaultdict
import numpy as np

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
        self.train_folder = config['evo_info']['train_folder']
        self.video_type = config['evo_info']['video_type']
        self.keypoint_type = config['evo_info']['keypoint_type']

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



# def single_evaluate(config, keypoint_file, task_program, sandbox):
#     print(f'evaluating {task_program["name"]} on {keypoint_file}')
#     # Adjusting the config for the current file
#     config['keypoint_info']['keypoint_file_path'] = keypoint_file
#     config['video_info']['video_file_path'] = keypoint_file.replace('.json', '.avi')
#     sandbox.update_config(config)
#     events = task_program(config, sandbox.exec_namespace)
#     return keypoint_file, task_program['name'], len(events)
    
class EasyEvolution(BaseEvolution):
    def __init__(self, config):
        super().__init__(config)
        self.mutation_llm = MutationLLM(config)
        self.code_generation_llm = CodeGenerationLLM(config)
        self.breed_llm = BreedLLM(config)
        self.task_program_library = TaskProgramLibrary().get_task_programs()
        # keep track of the fitness scores of each task program
        # the key should be the video file name -> task program name -> score
        self.scores = defaultdict(dict)
        # cache the results of task programs in scores
        # the key should be the video file name -> task program name -> events
        self.cache = defaultdict(dict)
        self.sandbox = EvoSandbox(
            config,
            CORE_API_REGISTRY)
        self.video2keypointfile = {}

    def events_duration(self, events):
        temp = [event.duration_in_seconds for event in events]
        return sum(temp)

    def calculate_fitness(self):
        # the fitness score of the behavior is determined by many factors
        # 1) total duration of the behavior
        # 2) number of videos it occurs
        total_duration = defaultdict(int)
        number_of_videos = defaultdict(int)
        for video in self.cache:
            for behavior_name in self.cache[video]:               
                total_duration[behavior_name] += self.events_duration(self.cache[video][behavior_name])
                if total_duration[behavior_name] > 0:
                    number_of_videos[behavior_name] += 1
        behaviors = set()
        for video in self.cache:
            for behavior_name in self.cache[video]:
                behaviors.add(behavior_name)
        self.debug = {}
        for behavior_name in behaviors:
            self.scores[behavior_name] = total_duration[behavior_name] * number_of_videos[behavior_name]
            self.debug[behavior_name] = f'''
total:{total_duration[behavior_name]} 
#video:{number_of_videos[behavior_name]}
score:{self.scores[behavior_name]}
'''       
        

    def mutate(self):        
        mutation_response = self.mutation_llm.speak(self.sandbox)
        print (mutation_response)
        # get the mutated task program
        pattern = r"```python(.*?)```"
        function_code = re.findall(pattern, mutation_response, re.DOTALL)[0]
       
        #parent_generation = self.task_program_library[self.sandbox.program_to_mutate]['generation']
        self.sandbox.scores = self.scores
        self.sandbox.register_task_program(function_code)
                                           #mutation_from = self.sandbox.program_to_mutate)
                                           
    
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
      
        ranked_survivals = sorted(self.scores.keys(), key=lambda x: self.scores[x], reverse=True)
        ranked_survivals = ranked_survivals[:3]
        index = random.randint(0, len(ranked_survivals)-1)     
        return ranked_survivals[index]

    def select_for_breed(self, num_participants = 2):
        ranked_survivals = sorted(self.scores.keys(), key=lambda x: self.scores[x], reverse=True)
        ranked_survivals = ranked_survivals[:3]

        indices = random.sample(range(0, len(ranked_survivals)), num_participants)
        return [ranked_survivals[i] for i in indices]    


    def train(self):
        self.evaluate() 
        for i in range(5):
            try:
                print (f'training iteration {i}')
                #mutate_candidate = self.select_for_mutation()
                #print (f'selecting {mutate_candidate} for mutation')
                #self.sandbox.update_program_to_mutate(mutate_candidate)
                self.mutate()            
                breed_participants = self.select_for_breed()
                print (f'selecting {breed_participants} for breeding')
                self.breed(breed_participants)
                self.evaluate()
                TaskProgramLibrary.save('task_program_checkpoint.json')
            except Exception as e:
                import traceback
                traceback.print_exc()
                continue              
   

    
    # def parallel_evaluate(self):
    #     train_files = glob.glob('../../tests/mabe_samples/train/*.json')
    #     test_files = glob.glob('../../tests/mabe_samples/test/*.json')
    #     config = Config('../../tests/MABe_evo.yaml')

    #     # Assuming self.task_program_library is a dictionary of task programs
    #     # and self.sandbox is properly initialized and can be deep copied for thread safety

    #     # Use a ProcessPoolExecutor to run tasks in parallel
    #     with ProcessPoolExecutor() as executor:
    #         # Creating a list to hold futures
    #         futures = []
    #         for name, task_program in self.task_program_library.items():
    #             for keypoint_file in train_files:
    #                 # Submit tasks to the executor
    #                 futures.append(executor.submit(single_evaluate, config.copy(), keypoint_file, task_program, self.sandbox.copy()))

    #         # Processing results as they complete
    #         for future in as_completed(futures):
    #             keypoint_file, task_name, events_len = future.result()
    #             print(keypoint_file, task_name, events_len)

    def info_gain(self):
        discovered_behaviors = []
        for name, task_program in self.task_program_library.items():                
            if task_program['creator'] != 'human':
                discovered_behaviors.append(name)
        scores = np.array([self.scores[name] for name in discovered_behaviors])

        for behavior_name in discovered_behaviors:
            print (behavior_name, self.debug[behavior_name])

        print (scores)        
        print ('success rate', np.sum(~np.isnan(scores)) / len(scores)) 

        # we only look at created ones
        for video in self.cache:
            for task_program_name in self.cache[video]:
                task_program = self.task_program_library[task_program_name]
                # if task_program['creator'] == 'human':
                #     continue
                if not np.isnan(self.scores[task_program_name]):
                    video_file_path = os.path.join(self.train_folder,video + self.video_type)
                    config['keypoint_info']['keypoint_file_path'] = self.video2keypointfile[video]
                    config['video_info']['video_file_path'] = video_file_path
                    self.sandbox.update_config(config)
                    self.sandbox.visual_validate(video_file_path, self.cache[video][task_program_name], task_program_name)   
        

    def evaluate(self):
        # this is to evaluate many videos in different processes
        import glob
        train_files = glob.glob(self.train_folder + f'/*{self.keypoint_type}')[:1]
        print (train_files)
        for name, task_program in self.task_program_library.items():
            for keypoint_file in train_files:
                videoname = keypoint_file.split('/')[-1].replace(self.keypoint_type, '')
                if videoname in self.cache and name in self.cache[videoname]:
                    continue
                # just replacing stuff in the config
                config['keypoint_info']['keypoint_file_path'] = keypoint_file
                config['video_info']['video_file_path'] = os.path.join(self.train_folder,keypoint_file.replace(self.keypoint_type, self.video_type))
                self.video2keypointfile[videoname] = keypoint_file
                self.sandbox.update_config(config)
                try:
                    events = task_program(config, self.sandbox.exec_namespace)
                    assert events is not None
                    self.cache[videoname][name] = events                    
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print ('something is wrong with task program', name)
                    print ('deleting it')
                    del self.task_program_library[name]
                
        self.calculate_fitness()
        self.info_gain()
        
       
    

if __name__ == "__main__":
    from amadeusgpt.config import Config
    from amadeusgpt.common_task_programs import register_common_task_programs
    config = Config('../../tests/MABe_evo.yaml')
    register_common_task_programs()
    evo = EasyEvolution(config)
   
    evo.train()
    
    