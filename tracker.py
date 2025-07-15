import torch
import numpy as np
import supervision as sv
from PIL import Image

import pipeline as pl

class Tracker(pl.Component):
    def __init__(self):
        super().__init__()
        self.tracker = sv.ByteTrack()

    def update(self, detections):
        """
        Update the tracker with new detections and the current frame.
        
        Args:
            detections (list): List of detected objects with bounding boxes and scores.
            frame (PIL.Image or numpy.ndarray): Current frame for tracking.
        
        Returns:
            list: Updated tracked objects with IDs and bounding boxes.
        """
        if not isinstance(detections, sv.Detections):
            detections = sv.Detections.from_transformers(detections)
        tracked_objects = self.tracker.update_with_detections(detections)

        return tracked_objects

    def process(self, data):
        """
        Process the input data to perform tracking.
        
        Args:
            data (pl.FrameData): Frame state data containing 'detections'.
        """
        if 'detections' not in data:
            raise ValueError("Data must contain 'detections' attribute.")

        data.tracked_objects = self.update(data.detections)



def test():
    """
    Test the Tracker component with a sample image and detections.
    """
    # Create a sample image
    image = Image.new('RGB', (640, 480), color='white')

    # Create sample detections
    detections = sv.Detections(
        xyxy=np.array([[100, 100, 200, 200],
                       [300, 300, 400, 400]]),
        confidence=np.array([0.8, 0.9]),
        class_id=np.array([10, 10]),
    )

    # Initialize and process with Tracker
    tracker = Tracker()
    for frame_num in range(5):
        # move the detections slightly for each frame
        detections.xyxy += np.array([5, 5, 5, 5])

        tracked_objects = tracker.update(detections)
        print(f"Frame {frame_num}:")
        for bbox, _, conf, cls, obj, _ in tracked_objects:
            print(f"  ID: {obj}, BBox: {bbox}, Confidence: {conf:.2f}, Class: {cls}")

if __name__ == "__main__":
    test()
