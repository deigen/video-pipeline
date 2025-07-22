import av
import pipeline as pl

from detector import HFDetector
from tracker import Tracker

from components import FrameReader, Counter, Sleep, Print
from multiprocess import Multiprocess

def main():
    #detector = Detector(HFDetector, model_name='facebook/detr-resnet-50', score_threshold=0.5)
    detector = Multiprocess(HFDetector, model_name='PekingU/rtdetr_v2_r18vd', score_threshold=0.5)
    detector.num_instances(2)

    tracker = Multiprocess(Tracker)
    tracker.num_instances(1)  # tracker has to be single instance to see all detections sequentially

    video_file = 'test_data/venice2.mp4'
    def video_frames_loop():
        while True:
            container = av.open(video_file)
            for frame in container.decode(video=0):
                yield frame.to_image()  # Convert to PIL Image

    source = (FrameReader(video_frames_loop()) >> pl.FixedRateLimiter(30))

    pipeline = (
      source >> Counter()
      >> pl.AdaptiveRateLimiter(initial_rate=30, print_stats_interval=1.0)
      >> detector
      >> tracker
      >> Print(lambda data: f'{len(data.tracked_objects.tracker_id)} objects tracked, latency: {pl.ts() - data.create_time:.3f}, throughput: {engine.global_meter.get():.3f} FPS')
    )

    engine = pl.PipelineEngine()
    engine.add(pipeline)
    engine.run()

if __name__ == "__main__":
    main()
