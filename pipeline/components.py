import threading
import time
import types
from collections.abc import Iterable

import pipeline as pl

try:
    import av
except ImportError:
    av = None

START_TIME = time.time()

__all__ = [
    'Counter', 'Sleep', 'Print', 'Breakpoint', 'VideoReader', 'VideoWriter', 'Function',
    'RunOrSkip'
]


class Counter(pl.Component):
    def __init__(self):
        super().__init__()
        self.count = 0

    def process(self, data):
        data.count = self.count
        data.create_time = pl.ts()
        self.count += 1


class Sleep(pl.Component):
    def __init__(self, seconds):
        super().__init__()
        self.seconds = seconds

    def process(self, data):
        time.sleep(self.seconds)
        data.sleeps = data.sleeps + 1 if hasattr(data, 'sleeps') else 1


class Print(pl.Component):
    def __init__(self, message=None, interval=None):
        super().__init__()
        self.message = message
        self.interval = interval
        self.last_print = 0

    def process(self, data):
        if self.interval is not None and pl.ts() - self.last_print < self.interval:
            return
        if isinstance(self.message, types.FunctionType):
            print(self.message(data))
        elif isinstance(self.message, str):
            print(self.message.format(data=data))
        elif self.message is None:
            print(data)
        self.last_print = pl.ts()
        #print(f'{data.count}  latency: {pl.ts() - data.create_time:.3f}  throughput: {meter.get():.3f}')


class Breakpoint(pl.Component):
    def __init__(self, condition=None):
        super().__init__()
        self.condition = condition

    def process(self, data):
        if self.condition and not self.condition(data):
            return
        breakpoint()
        pass


class VideoReader(pl.Component):
    def __init__(self, input_file=None, frames=None, loop=False):
        super().__init__()
        assert av is not None, 'PyAV is required for VideoReader'

        self.loop = loop

        # check only one of input_file or frames is provided
        if sum(map(lambda x: x is not None, [input_file, frames])) != 1:
            raise ValueError(
                'Either input_file or frames must be provided, but not both.'
            )

        if input_file is not None:
            # Use a video frames generator to read frames from the video file
            if self.loop:

                def frame_generator():
                    while True:
                        container = av.open(input_file)
                        for frame in container.decode(video=0):
                            yield frame
                        container.close()

                self.frames = frame_generator()
            else:
                container = av.open(input_file)
                self.frames = (frame for frame in container.decode(video=0))
        else:
            # Use the provided frames iterable or generator
            if self.loop:

                def frame_generator():
                    while True:
                        for frame in frames:
                            yield frame

                self.frames = frame_generator()
            else:
                self.frames = frames

        if isinstance(self.frames, Iterable):
            # If frames is an iterable, convert it to a generator
            self.frames = iter(self.frames)

        if not isinstance(self.frames, types.GeneratorType):
            raise ValueError('frames must be a generator or an iterable of frames.')

    def process(self, data):
        try:
            frame = next(self.frames)
            if isinstance(frame, av.VideoFrame):
                data.frame = frame.to_image()
            else:
                data.frame = frame
            if hasattr(frame, 'pts'):
                data.pts = frame.pts
        except StopIteration:
            raise pl.StreamEnd('End of video stream reached')


class VideoWriter(pl.Component):
    def __init__(self, output_file, fps=30, pix_fmt='yuv420p', codec='h264'):
        super().__init__()
        assert av is not None, 'PyAV is required for VideoWriter'
        self.output_file = output_file  # path to mp4 or other video file
        self.fps = fps
        self.pix_fmt = pix_fmt
        self.codec = codec
        self.container = None
        self.stream = None
        self.pts = 0  # pts for the video frames, from data.pts or sequential

    def num_threads(self, num_threads):
        if num_threads != 1:
            raise ValueError('VideoWriter can only run with 1 thread.')
        super().num_threads(num_threads)

    def _container_open(self, size=None):
        if self.container is not None:
            return
        if self.output_file.startswith('rtsp://'):
            # For RTSP output, use the 'rtsp' mode
            self.container = av.open(
                self.output_file, mode='w', format='rtsp', options={
                    'rtsp_transport': 'tcp', 'muxdelay': '0.1', 'stimeout': '5000000'
                }
            )
        else:
            self.container = av.open(self.output_file, mode='w')
        self.stream = self.container.add_stream(self.codec, rate=self.fps)
        if size is not None:
            self.stream.width = size[0]
            self.stream.height = size[1]
        self.stream.pix_fmt = self.pix_fmt

    def process(self, data):
        self._container_open(data.frame.size)

        frame = av.VideoFrame.from_image(data.frame)
        self.pts = data.pts if hasattr(data, 'pts') else self.pts + 1
        frame.pts = self.pts
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def end(self):
        if self.stream is not None:
            for packet in self.stream.encode(None):  # flush the stream
                self.container.mux(packet)
            self.container.close()


class Function(pl.Component):
    '''
    Component that wraps a function to be called in the pipeline.
    The function should accept a single argument, which is the FrameData object.
    '''
    def __init__(self, func):
        super().__init__()
        self.process = func


class RunOrSkip(pl.Component):
    def __init__(self, component):
        super().__init__()
        self.component = component
        self.semaphore = threading.Semaphore(1)

    def num_threads(self, num_threads):
        # use 2*num_threads or maybe num_threads+1 for this component:
        # num_threads used for hte underlying component, and 1 or more for
        # the noop pass-throughs when the semaphore is full
        self.semaphore = threading.Semaphore(num_threads)
        return super().num_threads(2 * num_threads)

    def process(self, data):
        if self.semaphore.acquire(blocking=False):
            try:
                self.component.process(data)
                data.processed = True
            finally:
                self.semaphore.release()
        else:
            data.processed = False


def test():
    #video = _teststream()

    meter = pl.ThroughputMeter()
    meter2 = pl.ThroughputMeter()
    print(meter.id, meter2.id)

    def _print_msg(data):
        #return f'SINK  t:{data.frame.time}   processed:{data.processed_frame}   latency: {pl.ts() - data.create_time:.3f}  throughput: {meter.get():.3f}'
        return f'SINK  c:{data.count}  latency: {pl.ts() - data.create_time:.3f}  throughput: {meter.get():.3f}  meter2: {meter2.get():.3f}  sleeps: {data.sleeps}'

    #source = (VideoReader(video) | pl.FixedRateLimiter(30))
    #source = VideoReader(video)

    pipeline = (
        #source | Counter()  #| pl.AdaptiveRateLimiter(meter, initial_rate=30, drop=False)
        Sleep(1 / 100.)
        | Counter()
        | pl.AdaptiveRateLimiter(meter, initial_rate=100, print_stats_interval=1.0)
        | meter2
        | Sleep(0.03)
        | meter  #| Print(_print_msg)
    )

    engine = pl.PipelineEngine(pipeline)
    engine.run()
