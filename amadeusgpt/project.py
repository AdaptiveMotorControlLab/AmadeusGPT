import os
import pprint

import yaml


def create_project(data_folder, result_folder, **kwargs):
    """
    Create a project config file. Save the config file to the result folder
    """
    config = {
        "data_info": {
            "data_folder": data_folder,
            "result_folder": result_folder,
            "video_suffix": ".mp4",
        },
        "llm_info": {"max_tokens": 4096, "temperature": 0.0, "keep_last_n_messages": 2},
        "object_info": {"load_objects_from_disk": False, "use_grid_objects": False},
        "keypoint_info": {
            "use_3d": False,
            "include_confidence": False,
        },
        "video_info": {"scene_frame_number": 1},
    }
    # save the dictionary config to yaml

    def set_nested_value(d, keys, value):
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value

    for key, value in kwargs.items():
        keys = key.split(".")
        set_nested_value(config, keys, value)

    os.makedirs(result_folder, exist_ok=True)

    file_path = os.path.join(result_folder, "config.yaml")

    with open(file_path, "w") as f:
        yaml.dump(config, f)

    print(
        f"Project created at {result_folder}. Results will be saved to {result_folder}"
    )
    print(
        f"The project will load video files (*{config['data_info']['video_suffix']}) and optionally keypoint files from {data_folder}"
    )
    print(f"A copy of the project config file is saved at {file_path}")
    pprint.pprint(config)

    return config
