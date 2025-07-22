import enum
import itertools
import logging
import os
import queue
import random
import threading
import time
import traceback
import types
import weakref

try:
    import torch
    have_torch = True
except ImportError:
    have_torch = False

logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO').upper())


class Drop(Exception):
  '''
  Exception to signal that the current item was dropped/skipped by the component.
  '''


class State(enum.Enum):
  NONE = 0
  PENDING = 1
  QUEUED = 2
  STARTED = 3
  DROPPED = 4
  COMPLETED = 5
  FAILED = 6

  def __gt__(self, other):
    return self.value > other.value

  def __ge__(self, other):
    return self.value >= other.value


_START_TIME = time.monotonic()


def ts():
  return time.monotonic() - _START_TIME


class FrameData:
  _initialized = False

  def __init__(self):
    self._data = {}
    self._initialized = True

  def set(self, key, value):
    assert self._initialized
    self._data[key] = value

  def get(self, key):
    assert self._initialized
    #print('count:', self._data.get('count'), 'get field:', key, 'data:', self._data.keys())
    if key not in self._data:
      raise KeyError(f"Key '{key}' not in FrameData.")
    return self._data[key]

  def pop(self, key):
    assert self._initialized
    if key not in self._data:
      raise KeyError(f"Key '{key}' not in FrameData.")
    return self._data.pop(key)

  def has(self, key):
    assert self._initialized
    return key in self._data

  def __contains__(self, item):
    return self.has(item)

  def __getitem__(self, item):
    return self.get(item)

  def __setitem__(self, key, value):
    self.set(key, value)

  def __getattr__(self, item):
    #print('getattr', item, 'initialized:', self._initialized)
    if self._initialized:
      return self.get(item)
    return super().__getattribute__(item)

  def __setattr__(self, key, value):
    if self._initialized:
      self.set(key, value)
    else:
      super().__setattr__(key, value)

  def __getstate__(self):
    state = self.__dict__.copy()
    if have_torch:
        # Ensure that tensors are cloned to avoid cross-process issues
        # Right now cuda tensors can be sent and received between 2 procs, but not passed to a 3rd proc
        state = find_tensors(state, lambda t: t.clone() if t.is_cuda else t)
    return state

  def __repr__(self):
    return f"FrameData({', '.join(f'{k}={v}' for k, v in self._data.items())})"


def find_tensors(obj, callback, parent=None, index=None):
    if isinstance(obj, torch.Tensor):
        obj = callback(obj)
        if obj is not None:
            parent[index] = obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            find_tensors(v, callback, obj, k)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            find_tensors(v, callback, obj, i)
    elif hasattr(obj, '__dict__'):
        find_tensors(obj.__dict__, callback)
    return obj


class Item:

  _ID_COUNTER = itertools.count()

  def __init__(self):
    self.id = next(Item._ID_COUNTER)
    self.states = {}
    self.error = None
    self.data = FrameData()

  def __repr__(self):
    return f'Item({self.id})'

  def set_state(self, component_id, state):
    now = ts()
    logging.debug("%f %s %s, -> %s", now, component_id, self, state)
    self.states[component_id] = (state, now)

  def get_state(self, component_id):
    if component_id not in self.states:
      return State.NONE
    return self.states[component_id][0]


class PipelineEngine:

  def __init__(self, components=None, max_buffer_size=None):
    self.work_buffer = []  # buffer for work items
    self.components = []  # list of all components
    self.changed_event = threading.Event()
    self.max_buffer_size = max_buffer_size
    self.running = True
    self.global_meter = ThroughputMeter()  # overall throughput meter for the engine
    if components is not None:
      self.add(components)

  def add(self, component_or_range):
    '''
    Add a component or a ComponentRange to the pipeline engine.
    If a ComponentRange is provided, all components in the range will be added.
    '''
    if isinstance(component_or_range, ComponentRange):
      self.add_component(component_or_range.end, recursive=True)
      assert component_or_range.start in self.components, "ComponentRange start should have been added by recursive add."
    elif isinstance(component_or_range, Component):
      self.add_component(component_or_range, recursive=True)
    else:
      raise TypeError(f"Expected Component or ComponentRange, got {type(component_or_range)}")

  def add_component(self, component, recursive=True):
    if component in self.components:
      assert component.engine is self
      return
    assert component.engine is None, "Component is already part of another pipeline engine."
    component.engine = self
    self.components.append(component)
    if recursive:
      for c in component.dependencies:
        self.add_component(c)

  def run(self):
    self._current_loop_start = 0
    self._verify_components()
    if self.max_buffer_size is None:
      self.max_buffer_size = sum(c.queue.maxsize for c in self.components)
    try:
      for component in self.components:
        logging.debug("Starting component %s", component.id)
        component.start()
      self.changed_event.set()
      while True:
        self.changed_event.wait()
        if self._current_loop_start == ts():
          # loop rounds faster than the time resolution, wait so time is always monotonic
          time.sleep(1e-6)
          continue
        self.changed_event.clear()
        self._current_loop_start = ts()
        self._start_items()
        self._schedule_components()
        self._cleanup()
        self._start_items()
    finally:
      self.running = False

  def callback(self, item, component_id):
    # called upon item completion, failure, or drop by a component to signal a state change
    self.changed_event.set()

  def _start_items(self, new=1):
    while (len(self.work_buffer) < self.max_buffer_size and
           (len(self.work_buffer) < new or
            any(s[0].value for s in self.work_buffer[-new].states.values()))):
      self.work_buffer.append(Item())
      self.changed_event.set()

  def _schedule_components(self):
    for component in self.components:
      component_id = component.id
      # go through items oldest to most recent to check if the component can be scheduled
      for item in self.work_buffer:
        if item.get_state(component_id) >= State.QUEUED:
          # already scheduled this item, check next in buffer list
          continue
        if all(item.get_state(dep.id) == State.COMPLETED for dep in component.dependencies):
          # all dependencies are completed, we can schedule this item
          item.set_state(component_id, State.PENDING)  # may or may not be queued here now, always mark as pending first
          component.enqueue(item)
        elif any(item.get_state(dep.id) == State.DROPPED for dep in component.dependencies):
          # propagate dropped in dep chain, and continue checking next item
          item.set_state(component_id, State.DROPPED)
        else:
          # go to next component, this one is blocked on dependencies
          break

  def _cleanup(self):
    # cleanup all items that have no work left
    num_remove = 0
    for item in self.work_buffer:
      if any(s[0] in (State.PENDING, State.QUEUED, State.STARTED) for s in item.states.values()):
        # this item, or others after it that are blocked by ordering, can still be processed
        break
      if any(s[1] >= self._current_loop_start for s in item.states.values()):
        # this item's last state change was after we last checked for scheduling, so it might still be in progress
        break
      if item.error:
        logging.error(
            f"Error in component:\n{''.join(traceback.format_exception(None, item.error, item.error.__traceback__))}"
        )
      num_remove += 1
    if num_remove:
      logging.debug("cleaning %s", self.work_buffer[:num_remove])
      del self.work_buffer[:num_remove]

  def _verify_components(self):
    # add global meter behind every component
    for component in self.components:
      self.global_meter.depends_on(component)
    self.add_component(self.global_meter, recursive=False)
    # AdaptiveRateLimiter requires a downstream meter, set to global meter by default
    for component in self.components:
      if isinstance(component, AdaptiveRateLimiter) and component.downstream_meter is None:
        component.downstream_meter = self.global_meter
    # check dependency ids for unknown components outside of the engine context
    all_ids = {c.id for c in self.components}
    for component in self.components:
      for dep in component.dependencies:
        if dep.id not in all_ids:
          raise Exception(
              f"Component {component.id} depends on unknown component {dep_id} not part of the engine."
          )
    # check for cycles
    self._check_cycles()

  def _check_cycles(self):
    # https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm
    in_degree = {c: 0 for c in self.components}
    for c in self.components:
      for dep in c.dependencies:
        in_degree[dep] += 1
    queue = [c for c, d in in_degree.items() if d == 0]
    while queue:
      c = queue.pop()
      for dep in c.dependencies:
        in_degree[dep] -= 1
        if in_degree[dep] == 0:
          queue.append(dep)
    if any(d != 0 for d in in_degree.values()):
      raise Exception("Dependency cycle detected.")

  def components_between(self, a, b=None):
    '''
    Return list of components between a and b (inclusive) according to the dependency graph.
    If b is None, return all components downstream of a.
    '''
    nodes = {a}
    visited = set()

    def dfs(node, path):
      if node in visited:
        return
      visited.add(node)
      if node in nodes:  # found a path that leads to a node that leads to a
        nodes.update(path)
        return
      path.append(node)
      for c in node.dependencies:
        dfs(c, path)
      path.pop()

    if b is not None:
      dfs(b, [])
    else:
      for c in self.components:
        dfs(c, [])

    assert a in nodes
    if b is not None:
      assert b in nodes
    return [c for c in self.components if c in nodes]


class Component:

  _ID_COUNTER = map(str, itertools.count())
  _ALL_COMPONENTS = {}

  def __init__(self, num_threads=1, queue_size=None):
    self.id = self.__class__.__name__ + '-' + next(Component._ID_COUNTER)
    Component._ALL_COMPONENTS[self.id] = weakref.ref(self)
    self.engine = None
    self.queue_size = queue_size
    self.num_threads(num_threads)  # sets num_threads and self.queue
    self._fields_map = None
    self.dependencies = set()
    self.average_qsize = 0
    self.threads = []

  def num_threads(self, num_threads):
    assert num_threads > 0
    assert not self.engine, "Cannot change number of threads after component has been added to an engine."
    self._num_threads = num_threads
    queue_size = self.queue_size or 2 * num_threads
    self.queue = queue.Queue(maxsize=queue_size)
    return self

  def fields(self, **fields_map):
    if self._fields_map is None:
      self._fields_map = {}
    self._fields_map.update(fields_map)
    return self

  def depends_on(self, other):
    self.dependencies.add(other)
    return self

  def __or__(self, other):
    other.depends_on(self)
    return ComponentRange(self, other)

  def enqueue(self, item):
    try:
      self.average_qsize = self.average_qsize * 0.9 + self.queue.qsize() * 0.1
      self.queue.put(item, block=False)
    except queue.Full:
      pass
    else:
      item.set_state(self.id, State.QUEUED)
      #self.engine.callback(item, self.id)  # should need cb only for done transitions

  def start(self):
    for i in range(self._num_threads):
      thread = threading.Thread(target=self.run_loop, daemon=True, name=self.id + '-' + str(i))
      self.threads.append(thread)
      thread.start()

  def pipeline_thread_init(self):
    '''
    This method is called once per thread before the run loop starts.
    Subclasses can override this to perform thread-specific initialization.
    '''
    pass

  def run_loop(self):
    self.pipeline_thread_init()
    while True:
      try:
        item = self.queue.get()  # blocking get for the next item
        item.set_state(self.id, State.STARTED)
        try:
          self._process(item.data)
        except Drop:
          item.set_state(self.id, State.DROPPED)
        except Exception as e:
          item.error = e
          item.set_state(self.id, State.FAILED)
        else:
          item.set_state(self.id, State.COMPLETED)
        finally:
          self.queue.task_done()
          self.engine.callback(item, self.id)
      except Exception:
        logging.exception('Internal error in component %s', self.id)
        time.sleep(1)

  def _process(self, item_data):
    if not self._fields_map:
      self.process(item_data)
      return

    # map between data and component fields names
    data = FrameData()
    data._data.update(item_data._data)  # copy existing data fields
    for comp_key, data_key in self._fields_map.items():
      if data_key in item_data:
        data.set(comp_key, data.pop(data_key))
    self.process(data)
    for comp_key, data_key in self._fields_map.items():
      if comp_key in data:
        data.set(data_key, data.pop(comp_key))
    item_data._data.update(data._data)  # update original item data with processed fields

  def process(self, item_data):
    raise NotImplementedError("Subclasses must implement this method.")


class ComponentRange:
  def __init__(self, start, end):
    self.start = start
    self.end = end

  def depends_on(self, other):
    if isinstance(other, ComponentRange):
      self.start.depends_on(other.end)
    elif isinstance(other, Component):
      self.start.depends_on(other)
    else:
      raise TypeError(f"depends_on() expects a Component or ComponentRange, got {type(other)}")
    return self

  def __or__(self, other):
    if isinstance(other, ComponentRange):
      other.start.depends_on(self.end)
      return ComponentRange(self.start, other.end)
    elif isinstance(other, Component):
      other.depends_on(self.end)
      return ComponentRange(self.start, other)
    else:
      raise TypeError(f"a | b expected a Component or ComponentRange, got {type(other)}")

  def __repr__(self):
    return f"ComponentRange({self.start.id}, {self.end.id})"


class ThroughputMeter(Component):

  def __init__(self, alpha=0.02, update_interval=0.1, print=False):
    super().__init__()
    self.lock = threading.Lock()
    self.alpha = alpha
    self.alpha_inv = 1 / alpha
    self.update_interval = update_interval
    self.print = print
    self.reset()

  def process(self, data):
    self.update()
    if self.print:
      logging.info("%s throughput: %s", str(self.print), self.get())

  def reset(self):
    self.start_time = ts()
    self.count = 0
    self.throughput = 0
    self.num_updates = 0

  def get(self):
    return self.throughput

  def update(self, count=1):
    with self.lock:
      now = ts()
      self.count += count
      T = self.update_interval
      if now - self.start_time > T:
        dt = now - self.start_time
        current = self.count / dt
        # a increases to alpha according to schedule for unbiased sample (i.e. not influenced by 0 init throughput value)
        if self.num_updates > self.alpha_inv:
          a = self.alpha
          #a = 1 - (1-a)**(dt/T)  # continuous time adjustment (probably not needed)
        else:
          a = 1 / min(self.alpha_inv, self.num_updates + 1)  # 1 -> alpha
        self.throughput = self.throughput * (1 - a) + current * a
        self.num_updates += 1
        self.start_time = now
        self.count = 0


class FixedRateLimiter(Component):

  def __init__(self, rate=30, drop=False):
    super().__init__()
    self.rate = rate
    self.drop = drop
    self.last_time = 0

  def process(self, data):
    elapsed = ts() - self.last_time
    interval = 1.0 / self.rate
    remaining = interval - elapsed
    if remaining > 0:
      if self.drop:
        if random.random() < remaining / interval:
          raise Drop
      else:
        time.sleep(interval - elapsed)
    self.last_time = ts()


class AdaptiveRateLimiter(Component):

  def __init__(self, downstream_meter=None, initial_rate=30, delta=0.1, target_qsize=0.1, drop=True, print_stats_interval=0):
    super().__init__()
    self.downstream_meter = downstream_meter
    self.initial_rate = initial_rate
    self.delta = delta
    self.target_qsize = target_qsize
    self.drop = drop
    self.last_time = 0
    self._downstream_components = None
    self._incoming_meter = ThroughputMeter()
    self._print_stats_time = 0
    self._print_stats_interval = print_stats_interval
    self._print_stats_outgoing_meter = ThroughputMeter()

  def process(self, data):
    if self._downstream_components is None:
      self._downstream_components = self.engine.components_between(self, self.downstream_meter)
      self._downstream_components.remove(self)

    self._incoming_meter.update()

    now = ts()
    elapsed = now - self.last_time

    achieved_throughput = self.downstream_meter.get()
    qsize = max(c.average_qsize for c in self._downstream_components)

    # adjust rate based on throughput and queue size
    if achieved_throughput == 0:
      rate = self.initial_rate
    else:
      if qsize < self.target_qsize:
        rate = achieved_throughput * (1 + self.delta)
      else:
        rate = achieved_throughput * (1 - self.delta)

    # sleep for the remaining time according to the rate
    interval = 1.0 / rate
    remaining = interval - elapsed
    incoming_rate = self._incoming_meter.get()
    if remaining > 0:
      if self.drop:
        if random.random() < (remaining * incoming_rate if incoming_rate else 1):
          raise Drop
      else:
        time.sleep(remaining)

    if self._print_stats_interval:
      self._print_stats_outgoing_meter.update()

    if self._print_stats_interval and now - self._print_stats_time > self._print_stats_interval:
      self._print_stats_time = now
      logging.info("THROUGHPUT TARGET: %s ACHIEVED: %s  QSIZE: %s, IN: %s, OUT: %s", rate,
                   achieved_throughput, qsize,
                   self._incoming_meter.get(), self._print_stats_outgoing_meter.get())

    self.last_time = ts()
