from amadeusgpt.config import Config
import os
import pprint
import yaml

def create_project(data_folder, 
                result_folder, 
                video_suffix=".mp4"):
    """
    Create a project config file. Save the config file to the result folder
    """
    config = {
        "data_info": {
            "data_folder": data_folder,
            "result_folder": result_folder,
            "video_suffix": video_suffix,
        },
        "llm_info": {
            "max_tokens": 4096,
            "temperature": 0.0,
            "keep_last_n_messages": 2
        },
    }
    # save the dictionary config to yaml

    os.makedirs(result_folder, exist_ok=True)

    file_path = os.path.join(result_folder, "config.yaml")

    with open(file_path, "w") as f:
        yaml.dump(config, f)

    print (f"Project created at {result_folder}. Results will be saved to {result_folder}")
    print (f"The project will load video files (*.{video_suffix}) and optionally keypoint files from {data_folder}")        
    print (f"A copy of the project config file is saved at {file_path}")
    pprint.pprint(config)

    return config