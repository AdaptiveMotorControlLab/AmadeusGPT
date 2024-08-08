from amadeusgpt import AMADEUS
from amadeusgpt import create_project
from amadeusgpt.utils import parse_result


def test_name_plotting():
    # the dummy video only contains 6 frames.
    kwargs = {
        'video_info.scene_frame_number': 1,
        'llm_info.gpt_model': "gpt-4o"
    }
    data_folder = "examples/DummyVideo"
    result_folder = "temp_result_folder"

    config = create_project(data_folder, result_folder, **kwargs)
    amadeus = AMADEUS(config, use_vlm=True)

    query = """ plot the keypoint names next to the keypoints """

    qa_message = amadeus.step(query)
    
    parse_result(amadeus, qa_message, use_ipython=False)

    #import matplotlib.pyplot as plt
    #plt.show()

if __name__ == "__main__":
    test_name_plotting()
