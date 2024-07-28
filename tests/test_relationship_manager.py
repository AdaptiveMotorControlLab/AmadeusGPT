from amadeusgpt.managers.relationship_manager import AnimalManager, ObjectManager, RelationshipManager
from amadeusgpt.config import Config
from amadeusgpt.behavior_analysis.identifier import Identifier


def test_relationship_manager():

    config = Config("amadeusgpt/configs/EPM_template.yaml")
    video_file_path = "examples/EPM/EPM_11.mp4"
    keypoint_file_path = "examples/EPM/EPM_11DLC_snapshot-1000.h5"
    identifier = Identifier(config, video_file_path, keypoint_file_path)
    
    animal_manager = AnimalManager(identifier)
    object_manager = ObjectManager(identifier, animal_manager)

    relationship_manager = RelationshipManager(identifier, animal_manager, object_manager)
   
    
    assert relationship_manager.get_animals_animals_relationships() == []
    # this assertion might change if grid objects are by default added
    assert relationship_manager.get_animals_objects_relationships() == []

if __name__ == "__main__":
    pytest.main()