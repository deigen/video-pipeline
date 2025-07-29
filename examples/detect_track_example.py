import supervision as sv

import pipeline as pl
from pipeline.detector import HFDetector
from pipeline.tracker import Tracker


def main():

    # create detector and tracker components

    # use 2 detector processes running in parallel
    detector = pl.Multiprocess(
        HFDetector, model_name='PekingU/rtdetr_v2_r18vd', score_threshold=0.5
    ).num_instances(2)

    # tracker has to be single instance to see all detections sequentially
    tracker = pl.Multiprocess(Tracker)

    # reader component to read frames from the video file
    reader = pl.VideoReader(input_file='test_data/venice2.mp4', loop=True)

    # set up annotators for bounding boxes and labels
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    @pl.Function
    def annotate(data):
        frame = data.frame.copy()
        frame = box_annotator.annotate(frame, detections=data.tracked_objects)
        frame = label_annotator.annotate(frame, detections=data.tracked_objects)
        data.annotated_frame = frame

    # output directory for annotated frames
    output_file = 'output.mp4'
    #output_file = 'rtsp://localhost:8554/output'
    writer = pl.VideoWriter(output_file=output_file, fps=30)

    engine = pl.PipelineEngine()

    engine.add(
        reader
        | pl.FixedRateLimiter(30)
        | pl.Counter().fields(count='pts')
        | pl.AdaptiveRateLimiter(initial_rate=30, print_stats_interval=1.0)
        | detector
        | tracker
        | annotate
        | writer.fields(frame='annotated_frame')
        | pl.Print(
            lambda data:
            f'FRAME {data.pts}: {len(data.tracked_objects.tracker_id)} objects tracked / {len(data.detections["boxes"])} detections, latency: {pl.ts() - data.create_time:.3f}, throughput: {engine.global_meter.get():.3f} FPS'
        )
    )

    # add detector print component to log detection counts
    # this print runs in parallel with the main pipeline
    engine.add(
        detector | pl.Print(
            lambda data:
            f'DETECTOR {data.pts}: {len(data.detections["boxes"])} detections'
        )
    )

    engine.run_until(lambda: reader.is_done)


if __name__ == "__main__":
    main()
