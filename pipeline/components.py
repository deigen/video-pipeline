import queue
import threading
import time
import types
from collections.abc import Iterable

from pipeline.engine import Component, FrameData, StreamEnd, ts

try:
    import av
except ImportError:
    av = None

START_TIME = time.time()

__all__ = [
    'Counter', 'Sleep', 'Print', 'Breakpoint', 'VideoReader', 'VideoWriter', 'Function',
    'LimitNumFrames', 'AsReady', 'FeedSource', 'IterSource', 'IterSink'
]

########### Utility and Debug Components ###########


class Counter(Component):
    '''
    Adds a count and create_time to the data object.

    data.count is set to a sequential count starting from 0.
    data.create_time is set to the current timestamp.
    '''
    def __init__(self):
        super().__init__()
        self.count = 0

    def process(self, data):
        data.count = self.count
        data.create_time = ts()
        self.count += 1


class Sleep(Component):
    '''
    Sleeps for a given number of seconds.
    '''
    def __init__(self, seconds):
        super().__init__()
        self.seconds = seconds

    def process(self, data):
        time.sleep(self.seconds)
        data.sleeps = data.sleeps + 1 if hasattr(data, 'sleeps') else 1


class Print(Component):
    '''
    Prints the data object or a formatted message to the console.

    If message is a function, it will be called with the data object.
    If message is a string, it will be formatted with the data object.
    If message is None, the data object will be printed directly.

    If interval is set, it will only print if the specified time has passed since the last print.
    '''
    def __init__(self, message=None, interval=None):
        super().__init__()
        self.message = message
        self.interval = interval
        self.last_print = 0

    def process(self, data):
        if self.interval is not None and ts() - self.last_print < self.interval:
            return
        if isinstance(self.message, types.FunctionType):
            print(self.message(data))
        elif isinstance(self.message, str):
            print(self.message.format(data=data))
        elif self.message is None:
            print(data)
        self.last_print = ts()


class LimitNumFrames(Component):
    '''
    Limits the number of frames processed by the pipeline.
    Raises StreamEnd when the limit is reached.
    '''
    def __init__(self, num_frames):
        super().__init__()
        self.num_frames = num_frames
        self.count = 0

    def process(self, data):
        if self.count >= self.num_frames:
            raise StreamEnd('Limit reached')
        self.count += 1


class Breakpoint(Component):
    '''
    Triggers a breakpoint in the code when processing data.
    If a condition is provided, it will only trigger the breakpoint if the condition is met.
    Useful for debugging to inspect the data object at a specific point in the pipeline.
    '''
    def __init__(self, condition=None):
        super().__init__()
        self.condition = condition

    def process(self, data):
        if self.condition and not self.condition(data):
            return
        breakpoint()
        pass


############# Video Reader/Writer Components ###########


class VideoReader(Component):
    '''
    Reads video frames from a file or an iterable of frames.

    If input_file is provided, it will read frames from the video file.
    If frames is provided, it will read frames from the iterable or generator.

    If loop is True, it will loop over the frames indefinitely.

    If add_pts is True, it will add a pts (presentation timestamp) to the data object.
    The pts is either taken from the video frame's pts or incremented sequentially.
    '''
    def __init__(self, input_file=None, frames=None, loop=False, add_pts=True):
        super().__init__()
        assert av is not None, 'PyAV is required for VideoReader'

        self.loop = loop
        self.add_pts = add_pts

        # check only one of input_file or frames is provided
        if sum(map(lambda x: x is not None, [input_file, frames])) != 1:
            raise ValueError(
                'Either input_file or frames must be provided, but not both.'
            )

        if input_file is not None:
            # Use a video frames generator to read frames from the video file
            if self.loop:

                def frame_generator():
                    last_d = 0
                    while True:
                        self._start_pts = self._curr_pts + last_d
                        container = av.open(input_file)
                        for frame in container.decode(video=0):
                            if hasattr(frame, 'pts'):
                                last_d = frame.pts - self._curr_pts
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

        self._start_pts = 0
        self._curr_pts = 0  # current pts for the video frames, from video pts or sequential

    def process(self, data):
        try:
            frame = next(self.frames)

            if isinstance(frame, av.VideoFrame):
                data.frame = frame.to_image()
            else:
                data.frame = frame

            if self.add_pts:
                if hasattr(frame, 'pts'):
                    self._curr_pts = self._start_pts + frame.pts
                else:
                    self._curr_pts += 1
                data.pts = self._curr_pts
        except StopIteration:
            raise StreamEnd('End of video stream reached')


class VideoWriter(Component):
    '''
    Writes video frames to a file or an RTSP stream.
    
    If output_file is a path to a video file, it will write frames to that file.
    If output_file is an RTSP URL, it will write frames to the RTSP stream.
    '''
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


############ General Purpose Components ###########


class Function(Component):
    '''
    Component that wraps a function to be called in the pipeline.
    The function should accept a single argument, which is the FrameData object.
    '''
    def __init__(self, func):
        super().__init__()
        self.process = func


class FeedSource(Component):
    '''
    Component that feeds data into the pipeline when 
    '''
    def __init__(self):
        super().__init__()
        self.stopped = False
        self.queue = queue.Queue()

    def feed_data(self, data):
        '''
        Feed data into the component's queue.
        '''
        if self.stopped:
            raise RuntimeError('FeedSource has been stopped and cannot accept new data.')
        if isinstance(data, dict):
            self.queue.put(data)
        elif isinstance(data, FrameData):
            self.queue.put(data._data)
        else:
            raise TypeError(
                f'Unsupported data type: {type(data)}. Expected dict or FrameData.'
            )

    def stop(self):
        '''
        Stop the feed source by raising StopIteration in the queue.
        This will signal the end of the feed source.
        '''
        self.stopped = True
        self.queue.put(StreamEnd)

    def process(self, data):
        d = self.queue.get()
        if d is StreamEnd:
            raise StreamEnd('End of feed source')
        data._data.update(d)  # update the FrameData with the new data


class IterSource(Component):
    '''
    Component that iterates over a list or generator of data.
    It will yield each item in the iterable as a FrameData object.
    '''
    def __init__(self, iterable):
        super().__init__()
        if not isinstance(iterable, (Iterable, types.GeneratorType)):
            raise TypeError(f'Expected iterable or generator, got {type(iterable)}')
        self.iterable = iter(iterable)

    def process(self, data):
        try:
            item = next(self.iterable)
            if isinstance(item, dict):
                data._data.update(item)
            elif isinstance(item, FrameData):
                data._data.update(item._data)
            else:
                raise TypeError(
                    f'Unsupported item type: {type(item)}, expected dict or FrameData.'
                )
        except StopIteration:
            raise StreamEnd


class IterSink(Component):
    '''
    Component that acts as a sink for the pipeline, collecting data from the pipeline
    and buffering it into a generator.
    '''
    def __init__(self, buffer_size=100):
        super().__init__()
        self.queue = queue.Queue(maxsize=buffer_size)

    def process(self, data):
        self.queue.put(data)

    def __iter__(self):
        while True:
            data = self.queue.get()
            if data is StopIteration:
                break
            yield data

    def end(self):
        self.queue.put(StopIteration)  # signal the end of the sink


class AsReady(Component):
    '''
    Runs the underlying component if it is not already executing.
    Otherwise, skips processing the data and marks it as not processed using data.processed = False.
    Note: use .fields(processed='newname') to rename the processed field if needed.
    '''
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
