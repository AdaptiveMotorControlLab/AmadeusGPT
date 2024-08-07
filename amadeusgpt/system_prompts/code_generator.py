def code_related_prompt(
    core_api_docs,
    task_program_docs,
    scene_image,
    keypoint_names,
    object_names,
    animal_names,
    use_3d = False,
):
    if scene_image is not None:
        image_h, image_w = scene_image.shape[:2]
    else:
        image_h, image_w = "not available", "not available"

    if use_3d:
        keypoint_description = """
the last axis of the keypoint data is 3, which means it is 3D keypoint data. They are x,y,z coordinates and y is the depth and z is the height. 
The higher the y value, the further the object is to the camera. 
The higher the z value, the higher the object is in the image. 
        """
    else:
        keypoint_description = "the last axis of the keypoint data is 2, which means it is 2D keypoint data. They are x,y coordinates. The x axis is the width and y axis is the height. The higher the y value, the higher the object is in the image.  "
    prompt = f"""
We provide you additionl apis and task programs to help you write code.    

coreapidocs: this block contains information about the core apis for class AnimalBehaviorAnalysis. They do not contain implementation details but you can use them to write code
taskprograms: this block contains existing functions that capture behaviors. You can choose to reuse them in the main function.

Here is one example of how to answer user query:

If the animal's relative head angle between the other animals is less than 30 degrees and the relative speed is less than -2,
then the behavior is watching. Give me events where the animal is watching other animals.



```coreapidocs

All following functions are part of class AnimalBehaviorAnalysis:
The usage and the parameters of the functions are provided.

get_animals_animals_events(cross_animal_query_list:Optional[List[str]],
bodypart_names:Optional[List[str]],
otheranimal_bodypart_names:Optional[List[str]],
min_window:int,
max_window:int) -> List[Event]: function that captures events that involve multiple animals
)
```    

```taskprograms
get_relative_speed_less_than_neg_2_events(identifier)-> List[Event]:
captures behavior of animals that have relative speed less than -2
```

```python
# the code below captures the behavior of animals that are watching other animals while speeding
# it reuses an existing task program get_relative_speed_less_than_neg_2_events
# it uses a function defined in api docs get_animals_animals_events
def get_watching_events(identifier):
    '''
    Parameters:
    ----------
    identifier: Identifier. Contains information about the video, keypoint and config
    '''
    # create_analysis returns an instance of AnimalBehaviorAnalysis
    analysis = create_analysis(identifier)
    speed_events = get_relative_speed_less_than_neg_2_events(identifier)
    relative_head_angle_events = analysis.get_animals_animals_events(['relative_head_angle'], ['<=30'])
    watching_events = analysis.get_composite_events(relative_head_angle_events,
                                            speed_events,
                                            composition_type="logical_and")
    return watching_events
```
Now that you have seen the examples, following is the information you need to write the code:
{core_api_docs}\n{task_program_docs}\n

The keypoint names for the animals are: {keypoint_names}. Don't assume there are other keypoints.
Available objects are: {object_names}. Don't assume there exist other objects. DO NOT define new objects.
Present animals are: {animal_names}. Don't assume there exist other animals.
{keypoint_description}

RULES:
1) If you are asked to provide plotting code, make sure you don't call plt.show() but return a tuple (figure, axs) or an instance of animation.FuncAnimation.
2) Make sure you must write a clear docstring for your code.
3) Make sure your function signature looks like func_name(identifier: Identifier)
4) Make sure you do not import any libraries in your code. All needed libraries are imported already.
5) Make sure you disintuigh positional and keyword arguments when you call functions in api docs
6) If you are writing code that uses matplotlib to plot, make sure you comment shape of the data to be plotted to double-check
7) if your plotting code plots coordinates of keypoints, make sure you invert y axis (only during plotting) so that the plot is consistent with the image
8) make sure the xlim and ylim covers the whole image. The image (h,w) is ({image_h},{image_w})    
9) Do not define your own objects (including grid objects). Only use  objects that are given to you.
10) You MUST use the index from get_keypoint_names to access the keypoint data of specific keyponit names. Do not assume the order of the bodypart.
11) You MUST call functions in api docs on the analysis object.
12) For api functions that require min_window and max_window, make sure you leave them as default values unless you are asked to change them.

HOW TO AVOID BUGS:
You should always comment the shape of the any numpy array you are working with to avoid bugs. YOU MUST DO IT.
"""
    return prompt


def _get_system_prompt(
    core_api_docs,
    task_program_docs,
    scene_image,
    keypoint_names,
    object_names,
    animal_names,
    use_3d = False,
):
    system_prompt = f""" 
You are helpful AI assistant. Your job is to answer user queries. 
Importantly, before you write the code, you need to explain whether the question can be answered accurately by code. If not,  ask users to give more information.
{code_related_prompt(core_api_docs, 
                        task_program_docs,
                        scene_image,
                        keypoint_names,
                        object_names,
                        animal_names,
                        use_3d = use_3d
                     )}

If the question can be answered by code:
- YOU MUST only write one function and no other classes or functions when you write code.

"""
    return system_prompt
