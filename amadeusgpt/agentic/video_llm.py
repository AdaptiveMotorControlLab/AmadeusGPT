from amadeusgpt.analysis_objects.llm import LLM
import cv2
import numpy as np 
import time 
import openai
from openai import OpenAI
import base64
import io 

class VideoSampler:
    """
    Sample a video into segments and then sample the segments into multiple frames using OpenCV.
    
    Attributes:
        video_path (str): Path to the video file.
        segment_duration (float): Duration of each segment in seconds.
        frames_per_segment (int): Number of frames to sample from each segment.
    """
    def __init__(self, video_path, segment_duration, frames_per_segment):
        """
        Initializes the VideoSampler with the specified video path, segment duration, and frames per segment.
        
        Args:
            video_path (str): Path to the video file.
            segment_duration (float): Duration of each video segment in seconds.
            frames_per_segment (int): Number of frames to extract per segment.
        """
        self.video_path = video_path
        self.segment_duration = segment_duration
        self.frames_per_segment = frames_per_segment

    def process_video(self):
        """
        Processes the video by segmenting it and then extracting frames from each segment.
        
        Returns:
            dict: A dictionary containing segment indices and their corresponding frames.
        """
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        segment_frame_count = int(self.segment_duration * fps)
        frames_to_capture = np.linspace(0, segment_frame_count - 1, self.frames_per_segment, dtype=int)

        video_data = {}
        segment_index = 0
        start = time.time()
        while cap.isOpened():
            segment_frames = []
            base_frame_index = segment_index * segment_frame_count
            for i in range(segment_frame_count):
                ret, frame = cap.read()
                if not ret:
                    break
                if i in frames_to_capture:
                    segment_frames.append(frame)

            if segment_frames:
                video_data[segment_index] = segment_frames
                segment_index += 1
            else:
                break

            # Skip to the next segment
            cap.set(cv2.CAP_PROP_POS_FRAMES, base_frame_index + segment_frame_count)
        end = time.time()
        print(f'Time taken for {total_frames} frames:', end - start)
        cap.release()
        return video_data


class VideoLLM:
    """
    Run GPT-4o on concatenated frames of a video
    """
    def __init__(self, config):
        super().__init__()

    def encode_image(self, image):
        result, buffer = cv2.imencode(".jpeg", image)
        image_bytes = io.BytesIO(buffer)
        base64_image = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
        return base64_image

    def images_to_video(self, image_list, output_path, fps, size=None):
        """
        Convert a list of images (numpy arrays) to a video.

        Args:
        image_list (list of numpy.ndarray): List containing image frames.
        output_path (str): Path where the output video will be saved.
        fps (int): Frames per second in the output video.
        size (tuple): Optional. Size of the video as (width, height).

        Raises:
        ValueError: If image_list is empty or size mismatch occurs.
        """
        if not image_list:
            raise ValueError("The image list is empty")

        # Determine the size of the video
        if size is None:
            # Take size from the first image
            height, width = image_list[0].shape[:2]
            size = (width, height)
        else:
            # Check if all images are of the same size
            height, width = size
            if any((img.shape[1] != width or img.shape[0] != height) for img in image_list):
                raise ValueError("All images must be of the same size as 'size' parameter")

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
        out = cv2.VideoWriter(output_path, fourcc, fps, size)

        # Write frames to the video
        for frame in image_list:
            # Convert to BGR format for saving
            if len(frame.shape) == 2 or frame.shape[2] == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            out.write(frame)

        # Release everything when job is finished
        out.release()
        print(f"Video saved as {output_path}")

    def run(self, video_data):
        temp = [e for e in video_data.values()]
        images = []
        for segment in temp:
            images.extend(segment)
        print ('total images GPT-4 sees', len(images))
        self.images_to_video(images, 'gpt4_sees_this.mp4', 30)
        client = OpenAI()
        encoded_images = [self.encode_image(image) for image in images]
        video_content =  [{"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}"}
        } for encoded_image in encoded_images]

        messages = \
        [
        {
        "role": 'system',
        "content":
          'You are an expert in animal behavior analysis especially lab mice. You will be given a video of a mouse in a home cage. The markers on the mouse are keypoints of the animal from DeepLabCut. You will be asked to observe the video and describe the habit or behavior the mouse is exhibiting.',
        },
        {      
        "role": "user", "content":  video_content
        }
        ]
        response = client.chat.completions.create(
            model="gpt-4o", messages=messages, max_tokens=1000
        )
        print (response)


if __name__  ==  '__main__':
    video_path = '/Users/shaokaiye/AmadeusGPT-dev/examples/MausHaus/maushaus_trimmedDLC_snapshot-1000_labeled_x264.mp4'
    segment_duration = 10
    frames_per_segment = 4

    video_sampler = VideoSampler(video_path, segment_duration, frames_per_segment)
    video_data = video_sampler.process_video()

    video_llm = VideoLLM(config=None)

    video_llm.run(video_data)
