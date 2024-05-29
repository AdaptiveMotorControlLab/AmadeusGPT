from os import system

def generate_core_api_block(core_api_docs):
    return f"""coreapidocs: this block contains information about the core apis that can help capture behaviors. Make sure you don't access any functions or attributes not defined in the core api docs.
{core_api_docs}\n"""

def generate_task_program_block(task_program_docs):
    return f"""taskprograms: this block contains the description of the behaivors we alreay know how to capture. 
{task_program_docs}\n"""

def generate_useful_info_block(useful_info):
    return f"""useful_info: this block contains information that can help you understand the context of the problem. 
    When you write task program about behaviors, e.g, fast vs. slow, far vs. close,  make sure you use the information in this block.
{useful_info}\n"""

def generate_grid_info_block(grid_labels, occupation_heatmap):
    ret =  f"""
    This block describes the spatial information of the environment. Each grid is a suqare area with a name. 
    The animal can be in one of the grids. The occupation heatmap shows the occupation of the grids (in terms of percentage of time).
    You can create task programs that the activities of animals in different grids.
    grid_labels:\n {grid_labels}
    occupation_heatmap:\n {occupation_heatmap}
    """
    print ('grid block')
    print (ret)
    return ret

def get_action_1_example():
    return """
Action 1: 
    Slightly change an existing task program '{to_mutate}' and make a new task program. Then you combine it with another existing task program '{to_breed}' by directly reusing the program from the variable task_programs. 

Example for Action 1:

This task program "get_approaches_then_contact_events" modifies the distance threshold of task program "get_approach_events" and 
combine it with an existing task program "get_contact_events" from a global variable 'task_programs'. You don't need to describe the existing task programs in the taskprograms block.

```python
def get_approaches_then_contact_events(config):
    '''
    This behavior is called "get_approach_and_contact". 
    This behavior describes animals moving from at least 50 pixels away to less than 8 pixels away and than make contact that is described in "get_contact_events".
    '''
    analysis = AnimalBehaviorAnalysis(config)
    distance_events = analysis.get_animals_animals_events(['distance>50'])

    close_distance_events = analysis.get_animals_animals_events(['distance<8'])

    # max_interval_between_sequential_events is the maximum interval between two events. Setting it higher might link two remote events together and likely give you a behavior that is not natural.
    approaches_events = analysis.get_composite_events(distance_events,
                                                                    close_distance_events,
                                                                    composition_type="sequential", 
                                                                    max_interval_between_sequential_events = 15)
    # retrieve an existing task program get_contact_events
    global task_programs
    contact_events = task_programs['get_contact_events'](config) 

    approach_then_contact_events = analysis.get_composite_events(approaches_events,
                                                                    contact_events,
                                                                    composition_type="sequential") 

    return approach_then_contact_events
```
"""


def modify_behavior(to_mutate = '', to_breed = '', autonomous = False):
    header = ""
    #     if autonomous:
    #         header = """
    # Your task is

    # 1) pick one existing task program from taskprograms block as the candidate to mutate. So you change it to make a new task program. You should make larger changes if task programs created by you are too similar to the original task programs.
    # 2) pick one existing task program from taskprograms block as the candidate to combine with the one above. So you combine the two task programs to make a new task program. You must directly reuse the to-combine program from the variable task_programs.

    # """

    header = f"""Your action space for you to write task programs has following actions: 
    Action:
        Since grids are special objects, we can use get_animals_object_events() to capture behaviors related to grids. 
        You will be given information about the available grids and the heatmaps. Use heatmaps as your guideline when you propose a new task program.
        
    """

    strategy = f"""
    {header}
    Make sure your task program has the same input and return types of other task programs. Make sure it also includes the description of the behavior.

    Example for the action:

    This task program "get_B2_to_C2_events" describes the behavior that animals move from grid B2 to grid C3.

    def get_B2_to_C3_events(config):
        analysis = AnimalBehaviorAnalysis(config)

        B2_events = analysis.get_animals_object_events('B2', 'overlap == True')
        C3_events = analysis.get_animals_object_events('C3', 'overlap == True')
        B2_to_C3_events = analysis.get_composite_events(B2_events, C3_events, composition_type="sequential")
        return B2_to_C3_events


    """
    return strategy

def get_social_interaction_example():
    return 
"""
An example of task program looks like following. Note it takes one and only config as input and it returns a list of BaseEvent.
```python
def get_approaches_events(config)->List[BaseEvent]:
    '''
    This behavior is called "approach". This behavior describes animals moving from at least 40 pixels away to less than 8 pixels away.
    '''
    analysis = AnimalBehaviorAnalysis(config)
    distance_events = analysis.get_animals_animals_events(['distance>40'])

    close_distance_events = analysis.get_animals_animals_events(['distance<8'])

    # max_interval_between_sequential_events is set to 40 to allow slightly longter interval between distance events and close distance events to ensure we capture long enough approach events.    
    approaches_events = analysis.get_composite_events(distance_events,
                                                                    close_distance_events,
                                                                    composition_type="sequential",
                                                                    max_interval_between_sequential_events = 40)


    return approaches_events

Query about orientation should use the following class:
class Orientation(IntEnum):
    FRONT = 1 
    BACK = 2 
    LEFT = 3 
    RIGHT = 4 
    
Note that the orientation is egocentric to the initiating animal.
For example, if we are describing a behavior that says 'this behavior involves animals watching animals in front of them', the orientation (animals being watched) is set to be FRONT.
If the animals are watching other animals in their left, the orientation is set to LEFT.    
"""


def select_strategy(strategy_info):
    ret = ''
    if 'to_mutate' in strategy_info:
        ret =  modify_behavior(to_mutate = strategy_info['to_mutate'],
                               to_breed = strategy_info['to_breed'])
    elif 'autonomous' in strategy_info:
        ret =  modify_behavior(autonomous = True)  
    # print ('selection strategy')
    # print (ret)

    return ret

def _get_system_prompt(sandbox
                       ):
    
    core_api_docs = sandbox.get_core_api_docs()
    task_program_docs = sandbox.get_task_program_docs()
    
    grid_labels, occupation_heatmap = sandbox.get_grid_info()    

    useful_info = sandbox.get_useful_info()
    mutation_strategy_info = sandbox.mutation_strategy_info
    

    system_prompt = f"""
You are an expert in animal behavior and are also good at python coding.
You have task programs that are functions that can capture behaviors from videos.
You treat every task program as an individual in evolution process. Each task program has its own fitness that is positively correlated with the number of videos it occured and the total duration in seconds.
Your GOAL is to 'evolve' task programs by using mutation and combination, similar to sexual production in biology. Over time, you should create diverse task programs that occur in many videos and have long durations and they are different from original ones.
You are provided with primitive task programs, as the seed of the evolution. You have access to task programs you created in the past.

You are provided with information that are organized in following blocks:
{generate_core_api_block(core_api_docs)}
{generate_task_program_block(task_program_docs)}
{generate_grid_info_block(grid_labels, occupation_heatmap)}


There are 3 and only 3 ways to combine events using get_composite_events.
'logical_and': to combine events if they happen at the same time. e.g., chase and follow can be combined using logical_and to be chase_and_follow. Get_animals_animals_events() is by default doing this.
'logical_or':  to combine events if one of the behaviors happen, e.g., chase and follow can be combined using logical_or to be chase_or_follow.
'sequential': One event happens after the other.  e.g., approach and leave can be combined using sequential to be approach_then_leave.

Coding style:
Don't import any new modules. Don't use apis under AnimalBehaviorAnalysis if they are not defined in the api docs.

Below is actions you can take:
{select_strategy(mutation_strategy_info)}

Format your answers as following:

1) Observation
    Describe your observations on the outcomes of task programs. Make sure you distinguish primitive task programs and those created by you.
    Analyze their total number of videos and the total duration in seconds and insights you drawed.

2) Reflection    
    Reflect whether you are making progress in achieving your goal that is to create novel, diverse task programs that occur in many videos and have long durations.
    Based on your reflection, you should:
    - comment why those task programs you created didn't succeed if they didn't and adjust your strategy accordingly.
    - comment whether your mutation is too conversavetiive. If they are, you should make larger changes to the task programs by larger changes or combining them with more complicate task programs.

3) Plans
Comment how you take your reflection into actionable plans

Write down the name of the task program you proposed.

```json  
    proposed : '' # filled if you 
```

4) Actions 
Here you turn your plans to actions by writing task program code
Make sure you write and only write one function for the task program.

```python
   # Your code starts here
```

"""
    return system_prompt