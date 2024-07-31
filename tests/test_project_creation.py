import os
from amadeusgpt import create_project
from amadeusgpt import AMADEUS
from amadeusgpt.utils import parse_result

if 'OPENAI_API_KEY' not in os.environ:  
     os.environ['OPENAI_API_KEY'] = 'your key'

# Create a project

data_folder = "examples/EPM/"
result_folder = "temp_result_folder"

config = create_project(data_folder, result_folder)

# Create an AMADEUS instance
amadeus = AMADEUS(config, use_vlm=True)

# query = "Plot the trajectory of the animal using the animal center and color it by time"
# qa_message = amadeus.step(query)
# parse_result(amadeus, qa_message)