

from amadeusgpt import create_project
from amadeusgpt import AMADEUS
from amadeusgpt.utils import parse_result


def test_3d_maushaus():

    kwargs = {
        'keypoint_info.use_3d': True,
        'llm_info.gpt_model': "moonshotai/kimi-dev-72b:free"
    }

    config = create_project(data_folder="examples/MausHaus3D", 
                            result_folder="3d_results",
                            **kwargs)


    amadeus = AMADEUS(config, use_vlm=False)

    behavior_analysis = amadeus.get_behavior_analysis(keypoint_file_path=amadeus.get_keypoint_file_paths()[0])

    assert behavior_analysis.get_keypoints().shape == (1000, 1, 30, 3)
    assert behavior_analysis.get_velocity().shape == (1000, 1, 30, 3)
    assert behavior_analysis.get_speed().shape == (1000, 1, 30, 1)
    assert behavior_analysis.get_acceleration_mag().shape == (1000, 1, 30, 1)
    
    query = "plot the 3D trajectory of the animal."

    qa_message = amadeus.step(query)

    parse_result(amadeus, qa_message, use_ipython=False)
