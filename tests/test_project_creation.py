import os
if 'OPENAI_API_KEY' not in os.environ:  
     os.environ['OPENAI_API_KEY'] = 'your key'

from amadeusgpt import create_project
from amadeusgpt import AMADEUS
# Create a project

data_folder = "temp_data_folder"
result_folder = "temp_result_folder"

config = create_project(data_folder, result_folder)

# Create an AMADEUS instance
amadeus = AMADEUS(config)