import os

import av
import supervision as sv

import pipeline as pl
from components import Counter, FrameReader, Print
from detector import HFDetector
from multiprocess import Multiprocess
from tracker import Tracker


def main():

    # create detector and tracker components

    detector = Multiprocess(
        HFDetector, model_name='PekingU/rtdetr_v2_r18vd', score_threshold=0.5
    )
    # use 2 detector processes running in parallel
    detector.num_instances(2)

    tracker = Multiprocess(Tracker)
    # tracker has to be single instance to see all detections sequentially
    tracker.num_instances(1)

    # input video file
    video_file = 'test_data/venice2.mp4'

    def video_frames_loop():
        container = av.open(video_file)
        for frame in container.decode(video=0):
            yield frame.to_image()  # Convert to PIL Image

    # reader component to read frames from the video file
    reader = FrameReader(video_frames_loop())

    # set up annotators for bounding boxes and labels
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    def annotate(data):
        frame = data.frame.copy()
        frame = box_annotator.annotate(frame, detections=data.tracked_objects)
        frame = label_annotator.annotate(frame, detections=data.tracked_objects)
        data.annotated_frame = frame

    # output directory for annotated frames
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    def write_frame(data):
        output_file = f'{output_dir}/frame_{data.pts:05d}.jpg'
        with open(output_file, 'wb') as f:
            data.annotated_frame.save(f, format='JPEG')

    pipeline = (
        reader
        | pl.FixedRateLimiter(30)
        | Counter().fields(count='pts')
        | pl.AdaptiveRateLimiter(initial_rate=30, print_stats_interval=1.0)
        | detector
        | tracker
        | pl.Function(annotate)
        | pl.Function(write_frame).num_threads(4)
        | Print(
            lambda data:
            f'FRAME {data.pts}: {len(data.tracked_objects.tracker_id)} objects tracked / {len(data.detections["boxes"])} detections, latency: {pl.ts() - data.create_time:.3f}, throughput: {engine.global_meter.get():.3f} FPS'
        )
    )

    engine = pl.PipelineEngine(pipeline)
    engine.run_until(lambda: reader.is_done)
    #engine.run()


if __name__ == "__main__":
    main()
