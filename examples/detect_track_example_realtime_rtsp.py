'''
Example of a real-time object detection and tracking pipeline using RTSP stream output.

This example uses a Hugging Face model for object detection and a ByteTrack tracker.

To view the output, you need to have an RTSP server running, such as mediamtx
(formerly simple-rtsp-server).  You can download it from
https://github.com/bluenviron/mediamtx/releases/
and run it with the command `./mediamtx`.

Then run this script and view the output stream using a media player like vlc,
connecting to the RTSP URL `rtsp://localhost:8554/example`.
'''

import supervision as sv  # for annotators

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
    # set to loop
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

    annotate.num_threads(4)

    # output annotated frames to RTSP stream
    # make sure you have an RTSP server running, e.g. using simple-rtsp-server/mediamtx
    rtsp_url = 'rtsp://localhost:8554/example'
    writer = pl.VideoWriter(output_file=rtsp_url, fps=30)

    engine = pl.PipelineEngine()

    # use a FixedRateLimiter to simulate a realtime incoming stream at 30 FPS
    source = reader | pl.FixedRateLimiter(30)

    # add components to the pipeline
    # use an AdaptiveRateLimiter to control the processing rate: drops frames close to uniformly
    # when processing is slower than the incoming rate
    engine.add(
        source
        | pl.AdaptiveRateLimiter(initial_rate=30, print_stats_interval=1.0)
        | detector
        | tracker
        | annotate
        | writer.fields(frame='annotated_frame')
    )

    engine.run()


if __name__ == "__main__":
    main()
