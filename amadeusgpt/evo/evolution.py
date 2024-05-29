from amadeusgpt.programs.task_program_registry import TaskProgram, TaskProgramLibrary
from amadeusgpt.programs.sandbox import EvoSandbox
from amadeusgpt.programs.api_registry import CORE_API_REGISTRY
from amadeusgpt.analysis_objects.analysis_factory import create_analysis
from amadeusgpt.analysis_objects.llm import MutationLLM, CodeGenerationLLM, BreedLLM
import re
import random
import functools
import time
import os
from collections import defaultdict
import numpy as np
import glob
import json 
from tqdm import tqdm


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
    ret_events = analysis.event_manager.get_composite_events(events1,
                                                              events2, composition_type = composition_type)
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
        self.sandbox = EvoSandbox(
            config)            
        self.video2keypointfile = {}

        TaskProgramLibrary.bind_exec_namespace(self.sandbox.exec_namespace)

    def events_duration(self, events):
        temp = [event.duration_in_seconds for event in events]
        if len(temp) == 0:
            return 0

        return sum(temp)

    def calculate_fitness(self):
        # the fitness score of the behavior is determined by many factors
        # 1) total duration of the behavior
        # 2) number of videos it occurs
        total_duration = defaultdict(int)
        number_of_videos = defaultdict(int)
        for video in TaskProgram.cache:
            for behavior_name in TaskProgram.cache[video]:               
                total_duration[behavior_name] += self.events_duration(TaskProgram.cache[video][behavior_name])
                if total_duration[behavior_name] > 0:
                    number_of_videos[behavior_name] += 1
        behaviors = set()
        for video in TaskProgram.cache:
            for behavior_name in TaskProgram.cache[video]:
                behaviors.add(behavior_name)
        detailed_scores = {}
        for behavior_name in behaviors:
            self.scores[behavior_name] = round(total_duration[behavior_name] * number_of_videos[behavior_name],1)
            detailed_scores[behavior_name] = f'''
total duration in seconds:{round(total_duration[behavior_name],2)} 
number of video occured:{number_of_videos[behavior_name]}
'''            
        # update the sandbox with the new scores  
        self.sandbox.scores = self.scores       
        self.sandbox.detailed_scores = detailed_scores
    
    def mutate(self, autonomous = False):        
        mutation_response = self.mutation_llm.speak(self.sandbox)
        print (mutation_response)
        # get the mutated task program
        pattern = r"```python(.*?)```"
        function_code = re.findall(pattern, mutation_response, re.DOTALL)[0]
        if autonomous:
            json_pattern = r"```json(.*?)```"
            json_string = re.findall(json_pattern, mutation_response, re.DOTALL)[0]
            json_obj = json.loads(json_string)
            mutate_from = json_obj['mutate_from']
            combine_with = json_obj['combine_with']
            print ('LLM autonomously selecting')
            print ('mutate from', mutate_from)
            print ('combine with', combine_with)

        self.sandbox.register_task_program(function_code)
                                                                                      
               
    def select_for_mutation(self) -> str:
        """
        This returns the name of the task program that is selected for mutation
        """
        keys, scores = [],[]
        for key, value in self.scores.items():
            keys.append(key)
            scores.append(value)

        keys = np.array(keys)
        scores = np.array(scores)       
        total_score = sum(scores)
        probabilities = [score / total_score for score in scores]
        selected_keys = np.random.choice(keys, size=2, replace=False, p=probabilities)
        return selected_keys[0], selected_keys[1]

    def select_mutation_strategy(self, autonomous = False):

        is_high_risk =  random.random() < 0.0
        if is_high_risk:
            return {}
        else:
            mutation_candidate, breed_candidate = self.select_for_mutation()            
            if autonomous:
                return {
                    'autonomous': True
                }
            else:
                return {
                    'to_mutate': mutation_candidate,
                    'to_breed': breed_candidate}
    
    def log_checkopint(self):
        TaskProgramLibrary.save(os.path.join(self.config['evo_info']['data_folder'],
                                                    'inspection',
                                                    'task_program_checkpoint.json'))
        
        with open (os.path.join(self.config['evo_info']['data_folder'],
                                'inspection',
                                'api_docs.json'), 'w') as f:
            json.dump(CORE_API_REGISTRY, f, indent=4)

    def load_checkpoint(self):
        TaskProgramLibrary.load(os.path.join(self.config['evo_info']['data_folder'],
                                                    'inspection',
                                                    'task_program_checkpoint.json'))
        
       


    def train(self, autonomous = False):
        # initial evaluation to provide initial scores
        self.evaluate()
        for i in range(10):
            try:
                print (f'training iteration {i}')
                # select the mutation strategy
                self.sandbox.mutation_strategy_info = self.select_mutation_strategy(autonomous=autonomous)
                self.mutate(autonomous=autonomous)
            
                self.evaluate()
                self.log_checkopint()
               
            except Exception as e:
                import traceback
                traceback.print_exc()
                continue                                     

    def info_gain(self):
        discovered_behaviors = []
        for name, task_program in self.task_program_library.items():                
            if task_program['creator'] != 'human':
                discovered_behaviors.append(name)
        scores = np.array([self.scores[name] for name in discovered_behaviors])       

        for name in discovered_behaviors:
            print (name, self.scores[name])

        print ('success rate', np.sum(scores!=0) / len(scores))

        train_folder = os.path.join(self.config['evo_info']['data_folder'],
                                    'train')
        video_type = self.config['evo_info']['video_type']
        # we only look at created ones
        for video in TaskProgram.cache:
            for task_program_name in TaskProgram.cache[video]:
                task_program = self.task_program_library[task_program_name]
                # if task_program['creator'] == 'human':
                #     continue
                if not np.isnan(self.scores[task_program_name]):
                    video_file_path = os.path.join(train_folder,video + video_type)
                    config['keypoint_info']['keypoint_file_path'] = self.video2keypointfile[video]
                    config['video_info']['video_file_path'] = video_file_path
                    self.sandbox.update_config(config)
                    # need to optimize if there are many videos
                    self.sandbox.visual_validate(video_file_path, TaskProgram.cache[video][task_program_name], task_program_name)   
        

    def evaluate(self):
        # this is to evaluate many videos in different processes
        train_folder = os.path.join(self.config['evo_info']['data_folder'],
                                    'train')
        keypoint_type = self.config['evo_info']['keypoint_type']
        video_type = self.config['evo_info']['video_type']
        train_files = glob.glob(train_folder + f'/*{keypoint_type}')
        total_iterations = len(train_files) * len(self.task_program_library)
        with tqdm(total=total_iterations, desc="Total progress") as pbar:
            for name, task_program in list(self.task_program_library.items()):
                for keypoint_file in train_files:
                    videoname = keypoint_file.split('/')[-1].replace(keypoint_type, '')
                    if videoname in TaskProgram.cache and name in TaskProgram.cache[videoname]:
                        pbar.update(1)
                        continue
                    # just replacing stuff in the config
                    config['keypoint_info']['keypoint_file_path'] = keypoint_file
                    config['video_info']['video_file_path'] = os.path.join(train_folder,keypoint_file.replace(keypoint_type, video_type))
                    self.video2keypointfile[videoname] = keypoint_file
                    self.sandbox.update_config(config)
                    try:
                        events = task_program(config)
                        assert events is not None
                        TaskProgram.cache[videoname][name] = events                    
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        print ('something is wrong with task program', name)                    
                        self.task_program_library.pop(name)
                    pbar.update(1)
        self.calculate_fitness()
        self.info_gain()
        
       
    

if __name__ == "__main__":
    from amadeusgpt.config import Config
    from amadeusgpt.programs.mabe_social_mabe import register_common_task_programs
    config = Config('../../experiments/mabe_evo_mabe.yaml')    
    analysis = create_analysis(config)
    # video_folder = os.path.join(config['evo_info']['data_folder'], 'train')
    # video_type = config['evo_info']['video_type']
    # keypoint_type = config['evo_info']['keypoint_type']
    # analysis.visual_manager.sanity_check_files(video_folder, video_type, keypoint_type)
    register_common_task_programs()
    evo = EasyEvolution(config)   
    evo.train()
    
    