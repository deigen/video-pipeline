import av
import pipeline as pl

from detector import HFDetector
from tracker import Tracker

from components import FrameReader, Counter, Sleep, Print, Detector

def main():
    detector = Detector(HFDetector, model_name='facebook/detr-resnet-50', score_threshold=0.5)
    tracker = Tracker()

    meter = pl.ThroughputMeter()

    video_file = 'test_data/venice2.mp4'
    def video_frames_loop():
        while True:
            container = av.open(video_file)
            for frame in container.decode(video=0):
                yield frame.to_image()  # Convert to PIL Image

    source = (FrameReader(video_frames_loop()) >> pl.FixedRateLimiter(30))

    last = (
      source >> Counter()
      >> pl.AdaptiveRateLimiter(meter, initial_rate=30, print_stats_interval=1.0)
      >> detector.num_threads(2)
      >> tracker
      >> meter
      >> Print(lambda data: f'{len(data.tracked_objects.tracker_id)} objects tracked, latency: {pl.ts() - data.create_time:.3f}, throughput: {meter.get():.3f}')
    )

    engine = pl.PipelineEngine()
    engine.add_component(last)
    engine.run()

if __name__ == "__main__":
    main()
