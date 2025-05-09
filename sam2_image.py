import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import os

def sam2_image_seg(image_path, prompt=None, output_path=None):
    """
    Segment an image using SAM2.
    
    Args:
        image_path (str): Path to the input image
        prompt (dict, optional): Dictionary with prompts for the model.
            Can contain 'point_coords', 'point_labels', and/or 'box'.
            Example: {'point_coords': np.array([[x, y]]), 'point_labels': np.array([1])}
        output_path (str, optional): Path to save visualization. If None, no visualization is saved.
        
    Returns:
        np.ndarray: Segmentation mask
    """    
    # Model paths
    checkpoint = "/home/ti_wang/AmadeusGPT/sam2/checkpoints/sam2.1_hiera_small.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    
    # Initialize predictor
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor.model = predictor.model.to(device)
    
    # Load image
    image = Image.open(image_path)
    image = np.array(image.convert("RGB"))
    predictor.set_image(image)
    
    # Run prediction with appropriate precision
    if device == "cuda":
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            if prompt is None:
                masks, scores, logits = predictor.predict()
            else:
                masks, scores, logits = predictor.predict(
                    point_coords=prompt.get('point_coords', None),
                    point_labels=prompt.get('point_labels', None),
                    box=prompt.get('box', None),
                    multimask_output=True
                )
    else:
        with torch.inference_mode():
            if prompt is None:
                masks, scores, logits = predictor.predict()
            else:
                masks, scores, logits = predictor.predict(
                    point_coords=prompt.get('point_coords', None),
                    point_labels=prompt.get('point_labels', None),
                    box=prompt.get('box', None),
                    multimask_output=True
                )
    
    # Save visualization if output_path is provided
    if output_path is not None:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(image)  # Show the original image
        plt.imshow(masks[0], cmap="jet", alpha=0.5)  # Overlay the first mask with transparency
        plt.axis("off")  # Remove axes for better visualization
        plt.title("Image with Predicted Mask")
        plt.savefig(output_path)
        plt.close()
    
    return masks


if __name__ == "__main__":
        
    # checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    # model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    # checkpoint = "/home/ti_wang/AmadeusGPT/sam2/checkpoints/sam2.1_hiera_small.pt"
    # model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

    # predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

    # with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    #     image_path = "./notebooks/images/cars.jpg"

    #     image = Image.open(image_path)
    #     image = np.array(image.convert("RGB"))
    #     # image = np.array(image)

    #     predictor.set_image(image)
    #     masks, _, _ = predictor.predict()
        
    #     # Plot the original image and overlay the mask
    #     plt.figure(figsize=(10, 10))
    #     plt.imshow(image)  # Show the original image
    #     plt.imshow(masks[0], cmap="jet", alpha=0.5)  # Overlay the first mask with transparency
    #     plt.axis("off")  # Remove axes for better visualization
    #     plt.title("Image with Predicted Mask")
    #     plt.savefig("./test_images/mask_overlay.png")  # Save the figure
        
    image_path = "./sam2/notebooks/images/truck.jpg"
    output_path = "./ti_test/mask_overlay_2.png"
    sam2_image_seg(image_path, output_path=output_path)