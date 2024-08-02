import os
import amadeusgpt
from pathlib import Path
from amadeusgpt.config import Config
from amadeusgpt import AMADEUS
import pytest


@pytest.mark.parametrize("template_name", ["EPM_template.yaml", "MABe_template.yaml", "Horse_template.yaml", "MausHaus_template.yaml"])
def test_demo_data(template_name):
    # test the completeness of the demo data

    config = Config(os.path.join("amadeusgpt", "configs", template_name))
    amadeus = AMADEUS(config, use_vlm = False)
    video_file_paths = amadeus.get_video_file_paths()
    keypoint_file_paths = amadeus.get_keypoint_file_paths()
    assert len(video_file_paths) == 1
    assert len(keypoint_file_paths) == 1
    assert os.path.exists(os.path.join(config['data_info']['data_folder'], 'example.json'))
    query = "plot the trajectory of the animal"
    qa_message = amadeus.step(query)



