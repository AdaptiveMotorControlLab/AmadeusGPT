from amadeusgpt.managers.animal_manager import AnimalManager
from amadeusgpt.config import Config
from amadeusgpt.behavior_analysis.identifier import Identifier
import pytest

def test_animal_manager():
    
    config = Config("amadeusgpt/configs/EPM_template.yaml")
    video_file_path = "examples/EPM/EPM_11.mp4"
    keypoint_file_path = "examples/EPM/EPM_11DLC_snapshot-1000.h5"
    identifier = Identifier(config, video_file_path, keypoint_file_path)
    animal_manager = AnimalManager(identifier)
   
    assert animal_manager.get_animal_names() == ['animal_0']
    assert animal_manager.get_keypoint_names() == ['nose', 'left_ear', 'right_ear', 'left_ear_tip', 'right_ear_tip', 'left_eye', 'right_eye', 'neck', 'mid_back', 'mouse_center', 'mid_backend', 'mid_backend2', 'mid_backend3', 'tail_base', 'tail1', 'tail2', 'tail3', 'tail4', 'tail5', 'left_shoulder', 'left_midside', 'left_hip', 'right_shoulder', 'right_midside', 'right_hip', 'tail_end', 'head_midpoint']
    assert animal_manager.get_n_individuals() == 1

if __name__ == "__main__":
    pytest.main()