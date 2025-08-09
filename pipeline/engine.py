import enum
import itertools
import logging
import queue
import threading
import time
import traceback
import weakref

try:
    import torch
    have_torch = True
except ImportError:
    have_torch = False

__all__ = [
    'PipelineEngine', 'Component', 'ComponentChain', 'FrameData', 'State', 'Drop',
    'StreamEnd', 'ts'
]


class Drop(Exception):
    '''
    Exception to signal that the current item was dropped/skipped by the component.
    '''


class StreamEnd(Exception):
    '''
    Exception to signal that the end of the stream was reached.
    This can be used by a component to indicate that no more items will be processed,
    for example a reader source that reaches the end of a video file.
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


NO_DEFAULT = object()  # sentinel for no default value in get


class FrameData:
    '''
    FrameData is a container for data associated with a single item (frame) in the pipeline.
    All fields are stored as key-value pairs, and can be accessed like a dictionary or an object.
    '''

    _initialized = False

    def __init__(self):
        self._data = {}
        self._initialized = True

    def set(self, key, value):
        assert self._initialized
        self._data[key] = value

    def get(self, key, default=NO_DEFAULT):
        assert self._initialized
        #print('count:', self._data.get('count'), 'get field:', key, 'data:', self._data.keys())
        if key not in self._data:
            if default is NO_DEFAULT:
                raise KeyError(f"Key '{key}' not in FrameData.")
            return default
        return self._data[key]

    def pop(self, key, default=NO_DEFAULT):
        assert self._initialized
        if key not in self._data:
            if default is NO_DEFAULT:
                raise KeyError(f"Key '{key}' not in FrameData.")
            return default
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
            try:
                return self.get(item)
            except KeyError:
                raise AttributeError(item)
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


class Item:
    '''
    Item represents a single unit of work in the pipeline, usually a frame.

    It contains its processing states for each component, used by the engine to track
    progress and determine which components can be scheduled for processing the item.
    '''

    _ID_COUNTER = itertools.count()

    def __init__(self, changed_event):
        self.id = next(Item._ID_COUNTER)
        self.states = {}
        self.error = None
        self.data = FrameData()
        self.changed_event = changed_event

    def __repr__(self):
        return f'Item({self.id} : {", ".join(f"{k}={v[0].name}" for k, v in self.states.items())})'

    def set_state(self, component_id, state):
        now = ts()
        logging.debug("%f %s %s, -> %s", now, component_id, self, state)
        self.states[component_id] = (state, now)
        self.changed_event.set()  # signal that a state has changed

    def get_state(self, component_id):
        if component_id not in self.states:
            return State.NONE
        return self.states[component_id][0]


class PipelineEngine:
    '''
    Main Engine class that runs the pipeline and schedules components to process items.
    '''
    def __init__(self, components=None, max_buffer_size=None):
        self.work_buffer = []  # buffer for work items
        self.components = []  # list of all components
        self.changed_event = threading.Event()
        self.max_buffer_size = max_buffer_size
        self.running = None
        self.done_callback = None  # callback to check if the pipeline is done
        self.is_done = False  # set to True once the callback returns True
        self._end_event = threading.Event()
        if components is not None:
            self.add(components)

    def add(self, components):
        '''
        Add a Component or a ComponentChain to the pipeline engine.
        If a ComponentChain is provided, all components in the dependency chain will be added.
        '''
        if isinstance(components, ComponentChain):
            components.link_dependencies()
            for component in components.components:
                self._add_component(component)
        elif isinstance(components, Component):
            self._add_component(components)
        else:
            raise TypeError(
                f"Expected Component or ComponentRange, got {type(components)}"
            )

    def _add_component(self, component):
        if component in self.components:
            assert component.engine is self
            return
        assert component.engine is None, "Component is already part of another pipeline engine."
        component.engine = self
        self.components.append(component)

    def run_until(self, done_callback, block=True):
        '''
        Run the pipeline until the done_callback returns True.

        After the callback returns True, the pipeline will stop processing new items,
        but will finish processing all items that are already in the pipeline buffer.
        '''
        self.done_callback = done_callback
        self.run(block=block)

    def run_forever(self, block=True):
        '''
        Run the pipeline indefinitely until stopped manually, even if a component reaches StreamEnd.
        '''
        self.done_callback = lambda: False
        self.run(block=block)

    def run(self, block=True):
        '''
        Main loop of the pipeline engine.

        By defualt, runs until any component signals StreamEnd and all
        remaining items in the work buffer are processed.  Use run_until() to
        change the stopping condition.

        Starts all components, and enters the main loop that processes items.
        The loop runs in the calling thread, and will not return until the pipeline finishes.

        If block is False, the loop will run in a separate thread and this
        method will return immediately.
        '''
        if not block:
            # run in a separate thread
            self._engine_thread = threading.Thread(target=self.run, daemon=True)
            self._engine_thread.start()
            return

        if self.done_callback is None:
            self.done_callback = lambda: any(c.is_done for c in self.components)

        self._engine_thread = threading.current_thread()

        self._current_loop_start = 0
        self._add_global_meter()
        self._verify_components()

        if self.max_buffer_size is None:
            self.max_buffer_size = 2 * sum(c._num_threads for c in self.components)

        self.running = True
        self.is_done = False
        try:
            for component in self.components:
                logging.debug("Starting component %s", component.id)
                component.start()
            self.changed_event.set()
            # main loop runs until is_done() is True and there are no items left to process
            while not self.is_done or self.work_buffer:
                self.changed_event.wait(timeout=1.0)
                self.changed_event.clear()
                if self._current_loop_start == ts():
                    # loop rounds faster than the time resolution, wait so time is always monotonic
                    time.sleep(1e-6)
                    continue
                self._current_loop_start = ts()
                self._check_done()
                self._start_items()
                self._schedule_components()
                self._cleanup()
                self._start_items()

            # after done, call all end() methods of components to clean up resources
            for component in self.components:
                if hasattr(component, 'end'):
                    try:
                        component.end()
                    except Exception as e:
                        logging.error(
                            f"Error in component {component.id} end method: {e}"
                        )
        finally:
            self.running = False
            self._end_event.set()

    def stop(self):
        '''
        Stop the pipeline engine, which will stop processing new items and finish processing
        all items that are already in progress.
        '''
        self.is_done = True

    def wait(self):
        '''
        Wait for the engine thread to finish processing all items and exit.
        '''
        self._end_event.wait()  # wait until the engine is done

    def get_fps(self):
        '''
        Get the current throughput in frames per second.
        '''
        if not hasattr(self, 'global_meter'):
            return None
        return self.global_meter.get()

    def _check_done(self):
        if self.is_done:
            return True
        if self.done_callback is not None:
            self.is_done = self.done_callback()
        return self.is_done

    def _start_items(self, new=1):
        if self.is_done:
            # if we reach the done condition, stop adding new items to the work buffer
            # and let the existing items finish processing
            return
        while (
            len(self.work_buffer) < self.max_buffer_size and (
                len(self.work_buffer) < new
                or any(s[0].value for s in self.work_buffer[-new].states.values())
            )
        ):
            self.work_buffer.append(Item(self.changed_event))

    def _schedule_components(self):
        for component in self.components:
            component_id = component.id
            # go through items oldest to most recent to check if the component can be scheduled
            for item in self.work_buffer:
                if item.get_state(component_id) >= State.QUEUED:
                    # already scheduled this item, check next in buffer list
                    continue
                if all(
                    item.get_state(dep.id) == State.COMPLETED
                    for dep in component.dependencies
                ):
                    # all dependencies are completed, we can schedule this item
                    item.set_state(
                        component_id, State.PENDING
                    )  # may or may not be queued here now, always mark as pending first
                    queued = component._enqueue_item(item)
                    if not queued:
                        # if the queue was full, do not try queuing subsequent items;
                        # instead, go back to top of item loop after there is space
                        # to maintain order
                        break
                elif any(
                    item.get_state(dep.id) == State.DROPPED
                    for dep in component.dependencies
                ):
                    # propagate dropped in dep chain, and continue checking next item
                    item.set_state(component_id, State.DROPPED)
                else:
                    # go to next component, this one is blocked on dependencies
                    break

    def _cleanup(self):
        # cleanup all items that have no work left
        num_remove = 0
        for item in self.work_buffer:
            if any(
                s[0] in (State.PENDING, State.QUEUED, State.STARTED)
                for s in item.states.values()
            ):
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

    def _add_global_meter(self):
        from .meters import AdaptiveRateLimiter, ThroughputMeter  # avoid circular import

        # add global meter behind every component
        self.global_meter = ThroughputMeter()  # overall throughput meter for the engine
        for component in self.components:
            self.global_meter.depends_on(component)
        self._add_component(self.global_meter)
        # AdaptiveRateLimiter requires a downstream meter, set to global meter by default
        for component in self.components:
            if (
                isinstance(component, AdaptiveRateLimiter)
                and component.downstream_meter is None
            ):
                component.downstream_meter = self.global_meter

    def _verify_components(self):
        # check dependency ids for unknown components outside of the engine context
        all_ids = {c.id for c in self.components}
        for component in self.components:
            for dep in component.dependencies:
                if dep.id not in all_ids:
                    raise Exception(
                        f"Component {component.id} depends on unknown component {dep.id} not part of the engine."
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

    def __init__(self):
        self.id = self.__class__.__name__ + '-' + next(Component._ID_COUNTER)
        Component._ALL_COMPONENTS[self.id] = weakref.ref(self)
        self.engine = None
        self.num_threads(1)  # sets up num_threads and self.queue to initial values
        self._fields_map = None
        self.dependencies = set()
        self.average_qsize = 0
        self.threads = []
        self.is_done = False
        self._item_queue = queue.Queue()

    def num_threads(self, num_threads):
        assert num_threads > 0
        assert not self.engine, "Cannot change number of threads after component has been added to an engine."
        self._num_threads = num_threads
        return self

    def fields(self, **fields_map):
        if self._fields_map is None:
            self._fields_map = {}
        self._fields_map.update(fields_map)
        return self

    def depends_on(self, other):
        self.dependencies.add(other)

    def __or__(self, other):
        if isinstance(other, ComponentChain):
            return ComponentChain([self] + other.components)
        elif isinstance(other, Component):
            return ComponentChain([self, other])
        else:
            raise TypeError(
                f"a | b expected a Component or ComponentChain, got {type(other)}"
            )

    def _enqueue_item(self, item):
        try:
            self.average_qsize = self.average_qsize * 0.9 + self._item_queue.qsize() * 0.1
            self._item_queue.put(item, block=False)
        except queue.Full:
            return False
        else:
            item.set_state(self.id, State.QUEUED)
        return True

    def start(self):
        for i in range(self._num_threads):
            thread = threading.Thread(
                target=self.run_loop, daemon=True, name=self.id + '-' + str(i)
            )
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
                item = self._item_queue.get()  # blocking get for the next item
                item.set_state(self.id, State.STARTED)
                try:
                    if self.is_done:
                        raise StreamEnd
                    self._process(item.data)
                except Drop:
                    item.set_state(self.id, State.DROPPED)
                except StreamEnd:
                    item.set_state(self.id, State.DROPPED)
                    self.is_done = True
                except Exception as e:
                    item.error = e
                    item.set_state(self.id, State.FAILED)
                else:
                    item.set_state(self.id, State.COMPLETED)
                finally:
                    self._item_queue.task_done()
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
        item_data._data.update(
            data._data
        )  # update original item data with processed fields

    def process(self, item_data):
        raise NotImplementedError("Subclasses must implement this method.")


class ComponentChain:
    def __init__(self, components):
        self.components = components

    def __or__(self, other):
        if isinstance(other, ComponentChain):
            return ComponentChain(self.components + other.components)
        elif isinstance(other, Component):
            return ComponentChain(self.components + [other])
        else:
            raise TypeError(
                f"a | b expected a Component or ComponentChain, got {type(other)}"
            )

    def __repr__(self):
        return f"ComponentChain({' | '.join(c.id for c in self.components)})"

    def link_dependencies(self):
        '''
        Link dependencies between components in the chain.
        Each component depends on the previous one in the chain.
        '''
        for i in range(1, len(self.components)):
            self.components[i].depends_on(self.components[i - 1])


##################### Utility Functions #####################


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
