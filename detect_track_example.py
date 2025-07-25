import os
import av
import numpy as np
import pipeline as pl
import supervision as sv

from detector import HFDetector
from tracker import Tracker

from components import FrameReader, Counter, Sleep, Print, Breakpoint
from multiprocess import Multiprocess

def main():
    #detector = Detector(HFDetector, model_name='facebook/detr-resnet-50', score_threshold=0.5)
    detector = Multiprocess(HFDetector, model_name='PekingU/rtdetr_v2_r18vd', score_threshold=0.5)
    detector.num_instances(2)

    tracker = Multiprocess(Tracker)
    tracker.num_instances(1)  # tracker has to be single instance to see all detections sequentially

    video_file = 'test_data/venice2.mp4'
    def video_frames_loop():
        container = av.open(video_file)
        for frame in container.decode(video=0):
            yield frame.to_image()  # Convert to PIL Image


    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    def annotate(data):
        frame = data.frame.copy()
        frame = box_annotator.annotate(frame, detections=data.tracked_objects)
        frame = label_annotator.annotate(frame, detections=data.tracked_objects)
        #data.annotated_frame = frame
        output_file = f'{output_dir}/frame_{data.pts:05d}.jpg'
        with open(output_file, 'wb') as f:
            frame.save(f, format='JPEG')

    reader = FrameReader(video_frames_loop())

    pipeline = (
        reader
        | pl.FixedRateLimiter(30)
        | Counter().fields(count='pts')
        | pl.AdaptiveRateLimiter(initial_rate=30, print_stats_interval=1.0)
        | detector
        | tracker
        | pl.Function(annotate).num_threads(4)
        #| FrameWriter(output_file).fields(frame='annotated_frame')
        | Print(lambda data: f'FRAME {data.pts}: {len(data.tracked_objects.tracker_id)} objects tracked, {len(data.detections["boxes"])} detected, latency: {pl.ts() - data.create_time:.3f}, throughput: {engine.global_meter.get():.3f} FPS')
    )

    engine = pl.PipelineEngine(pipeline)
    engine.run_until(lambda: reader.is_done)
    #engine.run()


if __name__ == "__main__":
    main()
