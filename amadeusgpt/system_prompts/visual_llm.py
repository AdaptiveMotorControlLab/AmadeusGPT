from pydantic import BaseModel
from typing import List, Literal

class VlmInferenceOutput(BaseModel):
    description: str
    individuals: int
    species: Literal["topview_mouse", "sideview_quadruped", "others"]
    background_objects: List[str]

def _get_system_prompt():
    system_prompt = """
    Describe what you see in the image and fill in the following json string:
    ```json
    {
        "description":
        "individuals": 
        "species": 
        "background_objects":
    }
    ```
    The "description" has high level description of the image.
    The "individuals" indicates the number of animals in the image
    The "species" indicates the species of the animals in the image. You can only choose from one of "topview_mouse", "sideview_quadruped" or "others". Note all quadruped animals should be considered as sideview_quadruped.
    The "background_objects" is a list of background objects in the image. 
    Explain your answers before you fill the answers. Make sure you only return one json string.    
    """
    return system_prompt
