import os

from amadeusgpt.config import Config


class Identifier:
    """
    config contains project level meta
    other parameters help identify the unique params needed for animal behavior analysis
    Currently video_file_path and keypoint_file_path.
    Can be more in the future
    """

    def __init__(self, config: Config, video_file_path: str, keypoint_file_path: str):

        self.config = config
        self.video_file_path = video_file_path
        self.keypoint_file_path = keypoint_file_path

    def __str__(self):
        return os.path.abspath(self.video_file_path)

    def __eq__(self, other):
        return self.video_file_path == other.video_file_path

    def __hash__(self):
        return hash(self.video_file_path)
