"""
Provide the seed task programs for the evolutionary algorithm
"""
from amadeusgpt.implementation import AnimalBehaviorAnalysis

def get_chase_events(config):
    """
    This behavior is called "chasing". Chasing describe when animals are less
    than 40 pixels and the chased animal is in front of the chasing animal and the chasing animal has to
    travel faster than 3.4 pixels per frame. The smooth window size is 25 and min window is 30. 
    """
    analysis = AnimalBehaviorAnalysis(config)
    chase_events = analysis.event_manager.get_animals_animals_events(['closest_distance',
                                                                      "orientation"],
                                                                       ['<40', "==Orientation.FRONT"],
                                                                       smooth_window_size=25,
                                                                       min_window = 30)
    speeding_events = analysis.event_manager.get_animals_state_events("speed", ">=3.4")

    chase_events = analysis.event_manager.get_composite_events(events_list = [chase_events,
                                                                speeding_events],
                                                                composition_type="logical_and",
                                                                min_window=30,
                                                                smooth_window_size = 25)    

    return chase_events
    


def get_oral_genital_contact_events(config):
    """
    This behavior called is "oral genital contact". This behavior describes animals' closest distance between 
    one animal’s "nose" and other animal’s bodyparts "tail base" is less than 15.   
    """

    analysis = AnimalBehaviorAnalysis(config)
    get_oral_genital_contact_events = analysis.event_manager.get_animals_animals_events(['closest_distance'],
                                                                                         ['<15'],
                                                                                         bodypart_names=['nose'],
                                                                                         otheranimal_bodypart_names=['tail base'])
    return get_oral_genital_contact_events
    
                                                                                        
def get_oral_oral_contact_events(config):
    """
    This behavior is called "oral oral contact". This behavior describes animals' closest distance between 
    one animal’s "nose" and other animal’s bodyparts "nose" is less than 15.
    """
    analysis = AnimalBehaviorAnalysis(config)
    get_oral_oral_contact_events = analysis.event_manager.get_animals_animals_events(['closest_distance'],
                                                                                         ['<15'],
                                                                                         bodypart_names=['nose'],
                                                                                         otheranimal_bodypart_names=['nose'])
    return get_oral_oral_contact_events


def get_approaches_events(config):
    """
    This behavior is called "approach". This behavior describes animals moving from at least 40 pixels away to less than 8 pixels away.
    """
    analysis = AnimalBehaviorAnalysis(config)
    distance_events = analysis.event_manager.get_animals_animals_events(['distance'],
                                                                               ['>40'])
    close_distance_events = analysis.event_manager.get_animals_animals_events(['distance'],
                                                                               ['<8'])
    approaches_events = analysis.event_manager.get_composite_events(events_list = [distance_events,
                                                                    close_distance_events],
                                                                    composition_type="sequential")


    return approaches_events


def get_huddles_events(config):
    """
    This behavior is called "huddle". 
    This behavior describes that multiple animals have distance less than 50 pixels and relative speed less than 4 pixels per frame.
    The smooth window for the events is 61 and min window is 75.
    """
    analysis = AnimalBehaviorAnalysis(config)
    huddles_events = analysis.event_manager.get_animals_animals_events(['distance', 'relative_speed'],
                                                                               ['<50', '<4'],
                                                                                min_window=75,
                                                                                smooth_window_size=61)
       
    return huddles_events

def get_contact_events(config):
    """
    This behavior is called "contact". This behavior describes animals have distance less than 12 pixels.
    The smooth window for the events is 11 and min window is 5.
    """
    analysis = AnimalBehaviorAnalysis(config)
    contact_events = analysis.event_manager.get_animals_animals_events(['closest_distance'],
                                                                               ['<12'],
                                                                               smooth_window_size = 11,
                                                                               min_window = 5)
    return contact_events 

def get_watching_events(config):
    """
    This behavior is called "watching". This behavior describes animals have distance less than 260 pixels,
    distance greater than 50 pixels, and relative head angle less than 15 degrees. The smooth window for the events is 15 and min window is 100.
    """
    analysis = AnimalBehaviorAnalysis(config)
    watching_events = analysis.event_manager.get_animals_animals_events(['distance', 'distance', 'relative_head_angle'],
                                                                        ['<260', '>50', '<15'],
                                                                        min_window=100,
                                                                        smooth_window_size=15)
    
    return watching_events

def get_oral_ear_contact_events(config):
    """      
    This behavior is called "oral ear contact". This behavior describes animals' closest distance between
    one animal’s "nose" and other animal’s bodyparts "left ear" and "right ear" is less than 10. The smooth window for the events is 5 and min window is 15.  
    """

    analysis = AnimalBehaviorAnalysis(config)
    get_oral_ear_contact_events = analysis.event_manager.get_animals_animals_events(['closest_distance', 'closest_distance'],
                                                                                         ['<10', '<10'],
                                                                                         bodypart_names=['nose'],
                                                                                         otheranimal_bodypart_names=['left ear', 'right ear'],
                                                                                        min_window = 15,
                                                                                        smooth_window_size= 5)
    return get_oral_ear_contact_events


def register_common_task_programs():
    # get all the functions in the current file and register them
    import inspect
    import sys
    from amadeusgpt.task_program_registry import TaskProgramLibrary
    current_module = sys.modules[__name__]
    all_functions = inspect.getmembers(current_module, inspect.isfunction)
    for function_name, function in all_functions:
        if function_name == "register_common_task_programs":
            continue
        TaskProgramLibrary.register_task_program(creator="human")(function)

if __name__ == "__main__":
    register_common_task_programs()
    from amadeusgpt.task_program_registry import TaskProgramLibrary
    task_programs = TaskProgramLibrary.get_task_programs()
    for name, task in task_programs.items():
        task.display()
