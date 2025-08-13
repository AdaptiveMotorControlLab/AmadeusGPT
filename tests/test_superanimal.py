from amadeusgpt import AMADEUS
from amadeusgpt import create_project
from amadeusgpt.utils import parse_result


def test_superanimal():
    # the dummy video only contains 6 frames.
    kwargs = {
        'video_info.scene_frame_number': 1,
        'llm_info.gpt_model': "qwen/qwen2.5-vl-72b-instruct:free"
    }
    data_folder = "examples/DummyVideo"
    result_folder = "temp_result_folder"

    config = create_project(data_folder, result_folder, **kwargs)
    amadeus = AMADEUS(config, use_vlm=True)
    behavior_analysis = amadeus.get_behavior_analysis(video_file_path=amadeus.get_video_file_paths()[0])
    keypoints = behavior_analysis.get_keypoints()
    assert keypoints.shape == (5, 1, 27, 2)


if __name__ == "__main__":
    test_superanimal()
