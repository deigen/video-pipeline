'''
Object Detection using HF models
'''

import torch
import numpy as np
from PIL import Image
import torch


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class HFDetector:
    def __init__(self, model_name=None, score_threshold=0.5, device=DEVICE):
        """
        Uses a Hugging Face model for object detection.
        Args:
            model_name (str): Name of the model to use. Defaults to 'facebook/detr-resnet-50'.
            device (str): Device to load the model on ('cpu' or 'cuda').
        """
        from transformers import AutoModelForObjectDetection, AutoImageProcessor

        model_name = model_name or 'facebook/detr-resnet-50'

        self.device = device
        self.score_threshold = score_threshold

        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForObjectDetection.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()


    @torch.no_grad()
    def detect(self, image):
        """
        Perform object detection on an input image.
        
        Args:
            image (PIL.Image or numpy.ndarray): Input image for detection.
        
        Returns:
            list: Detected objects with bounding boxes and scores.
        """
        inputs = self.image_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
             outputs = self.model(**inputs)

        results = self.image_processor.post_process_object_detection(outputs, target_sizes=torch.tensor([(image.height, image.width)]), threshold=0.5)

        return results



def test():
    """
    Test the Detector with a sample image.
    """
    image = Image.open('test_data/sample.jpg')
    detector = HFDetector()
    detections = detector.detect(image)
    detections = detections[0]  # remove batch dimension
    
    for score, label_id, box in zip(detections["scores"], detections["labels"], detections["boxes"]):
        score, label = score.item(), label_id.item()
        box = [round(i, 2) for i in box.tolist()]
        print(f"{detector.model.config.id2label[label]}: {score:.2f} {box}")

if __name__ == "__main__":
    test()
