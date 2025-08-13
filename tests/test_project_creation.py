import os
from amadeusgpt import create_project
from amadeusgpt import AMADEUS
import pytest
from dotenv import load_dotenv

try:
    load_dotenv()
except:
    pass

# if 'OPENAI_API_KEY' not in os.environ:  
#      os.environ['OPENAI_API_KEY'] = 'your key'

# Create a project

@pytest.mark.parametrize("example_name", ["MausHaus", "MausHaus3D"])
def test_project_creation(example_name):

    data_folder = f"examples/{example_name}"
    result_folder = "temp_result_folder"

    kwargs = {
        "llm_info.max_tokens": 20000,
        "llm_info.temperature": 0.0,
        'llm_info.gpt_model': "moonshotai/kimi-dev-72b:free",
        "llm_info.keep_last_n_messages": 2,
        "object_info.load_objects_from_disk": False,
        "object_info.use_grid_objects": False,
        "keypoint_info.use_3d": False,
        "keypoint_info.include_confidence": False
    }

    config = create_project(data_folder, result_folder, **kwargs)

    # Create an AMADEUS instance
    amadeus = AMADEUS(config, use_vlm=False)

    # let's start testing a simple query using openai api
    query = "Plot the trajectory of the animal using the animal center and color it by time"
    qa_message = amadeus.step(query)
