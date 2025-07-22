import threading
import time
import types

import pipeline as pl
#from clarifai.utils import video_utils

START_TIME = time.time()


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

  def __init__(self, message, interval=None):
    super().__init__()
    self.message = message
    self.interval = interval
    self.last_print = 0

  def process(self, data):
    if self.interval is not None and pl.ts() - self.last_print < self.interval:
      return
    if isinstance(self.message, types.FunctionType):
      print(self.message(data))
    else:
      print(self.message)
    self.last_print = pl.ts()
    #print(f'{data.count}  latency: {pl.ts() - data.create_time:.3f}  throughput: {meter.get():.3f}')


class FrameReader(pl.Component):

  def __init__(self, video_frames_generator):
    super().__init__()
    self.video_frames_generator = video_frames_generator

  def process(self, data):
    try:
      data.frame = next(self.video_frames_generator)
    except StopIteration:
      raise pl.Drop


class ExecuteIfReady(pl.Component):

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
    acquired = self.semaphore.acquire(blocking=False)
    if acquired:
      try:
        self.component.process(data)
        data.processed_frame = True
      finally:
        self.semaphore.release()
    else:
      data.processed_frame = False


#video = video_utils.stream_frames_from_file('demo.mp4')
#def _teststream():
#  while True:
#    yield from video_utils.stream_frames_from_file('demo.mp4')


#def testgen():
#    for i in range(100000):
#        frame = types.SimpleNamespace()
#        frame.pts = i
#        frame.time = pl.ts()
#        yield frame
#video = testgen()


def test():
  #video = _teststream()

  meter = pl.ThroughputMeter()
  meter2 = pl.ThroughputMeter()
  print(meter.id, meter2.id)

  def _print_msg(data):
    #return f'SINK  t:{data.frame.time}   processed:{data.processed_frame}   latency: {pl.ts() - data.create_time:.3f}  throughput: {meter.get():.3f}'
    return f'SINK  c:{data.count}  latency: {pl.ts() - data.create_time:.3f}  throughput: {meter.get():.3f}  meter2: {meter2.get():.3f}  sleeps: {data.sleeps}'

  #source = (FrameReader(video) | pl.FixedRateLimiter(30))
  #source = FrameReader(video)

  pipeline = (
      #source | Counter()  #| pl.AdaptiveRateLimiter(meter, initial_rate=30, drop=False)
      Sleep(1/100.)
      | Counter()
      | pl.AdaptiveRateLimiter(meter, initial_rate=100, print_stats_interval=1.0)
      | meter2
      | Sleep(0.03)
      | meter #| Print(_print_msg)
  )

  engine = pl.PipelineEngine(pipeline)
  engine.run()
