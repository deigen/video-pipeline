'''
Multiprocess component for the pipelines.

This component allows you to run a class in a separate process, enabling
parallel processing of data.  As for all components, the class must
implement a `process` method that takes a `FrameData` object as input.

The `Multiprocess` component can create multiple instances of the class, each running
in their own process.  Only data fields that are used by the child process
are sent to the child process, and only fields that are set in the child are sent back.

Usage:

from multiprocess import Multiprocess

class MyClass(pl.Component):
    def __init__(self, value):
        self.value = value

    def process(self, data):
        # Simulate some processing
        data.y = self.value + data.x

# Create a Multiprocess component
mp = Multiprocess(MyClass, value=10)
mp.num_instances(4)  # Set the number of instances (processes)

# "mp" is now a component that can be used in a pipeline.
# Each instance of MyClass runs in its own process; only data fields that
# are accessed or set will be sent to (or back from) the child process.
'''

import multiprocessing
import threading
import traceback

import pipeline as pl


class Multiprocess(pl.Component):
    def __init__(self, cls, **init_kwargs):
        super().__init__()
        self.cls = cls
        self.init_kwargs = init_kwargs
        self.instances = {}  # maps thread id to process client
        self._data_fields = set()

    def pipeline_thread_init(self):
        client = start_process(self.cls, **self.init_kwargs)
        self.instances[id(threading.current_thread())] = client

    def process(self, data):
        instance = self.instances[id(threading.current_thread())]
        instance.process(data)

    def end(self):
        # Close all child processes
        for instance in self.instances.values():
            instance.exit()

    def num_instances(self, n):
        # processes will be created in pipeline_thread_init, one for each component thread
        self.num_threads(n)  # child processes are 1-1 with component threads
        return self


def start_process(cls, *args, **kwargs):
    ctx = multiprocessing.get_context('spawn')

    parent_conn, child_conn = ctx.Pipe()

    proc = ctx.Process(
        target=_child_process_server_loop, args=(cls, args, kwargs, child_conn)
    )
    proc.start()

    class _MPComponentClient(MPComponentClient):
        __name__ = f'MPComponentClient<{cls.__name__}>'

    return _MPComponentClient(proc, parent_conn, child_conn)


class MPComponentClient:
    def __init__(self, server_proc, parent_conn, child_conn):
        self._server_proc = server_proc
        self._parent_conn = parent_conn
        self._child_conn = child_conn
        self._data_fields = set()
        self._my_thread = threading.current_thread()
        self.wait_for_ready()

    def process(self, data, **kwargs):
        send_data = _LazyFrameData(self._child_conn)
        for field in self._data_fields:
            if hasattr(data, field):
                setattr(send_data, field, getattr(data, field))
        for code, result in self._call('process', (send_data,), kwargs):
            if code == 'return':
                return result
            elif code == 'fetch_data':
                assert isinstance(
                    result, list
                ), f"Expected list of fields, got {type(result)}"
                send_data = pl.FrameData()
                self._data_fields.update(result)  # save fields for sending
                for field in result:
                    try:
                        send_data[field] = data[field]
                    except KeyError as e:
                        send_data[field] = e
                self._parent_conn.send(('fill_data', send_data))
            elif code == 'fill_data':
                assert isinstance(
                    result, pl.FrameData
                ), f"Expected FrameData, got {type(result)}"
                for field in result._data.keys():
                    data[field] = result[field]
            else:
                raise ValueError(f"Unexpected code: {code}, result: {result}")

    def _call(self, name, args, kwargs):
        assert threading.current_thread() == self._my_thread, \
            'Each component engine thread is 1-1 with its child process for MP'
        self._parent_conn.send((name, args, kwargs))
        code = None
        while code != 'return':
            code, result = self._parent_conn.recv()
            if code == 'error':
                assert isinstance(
                    result, Exception
                ), f"Expected an exception, got {result}"
                raise result
            yield code, result

    def wait_for_ready(self):
        # Wait for the child process to be ready
        for code, result in self._call('ready', None, None):
            if (code, result) != ('return', 'ready'):
                raise RuntimeError(f"Expected 'ready', got {code}, {result}")

    def exit(self):
        self._parent_conn.send(('exit', None, None))
        self._server_proc.join()


def _child_process_server_loop(cls, init_args, init_kwargs, child_conn):
    instance = cls(*init_args, **init_kwargs)
    instance.pipeline_thread_init()  # in-thread component init callback
    while True:
        try:
            method_name, method_args, method_kwargs = child_conn.recv()
            #print('Child process received:', method_name, method_args, method_kwargs)
            if method_name == 'exit':
                if hasattr(instance, 'end'):
                    instance.end()
                break
            if method_name == 'ready':
                child_conn.send(('return', 'ready'))
                continue
            method = getattr(instance, method_name)
            result = method(*method_args, **method_kwargs)
            data = [arg for arg in method_args if isinstance(arg, _LazyFrameData)]
            assert len(data) <= 1, "Only one _LazyFrameData argument is allowed"
            changed = data[0]._get_changed_data() if data else None
            # for now use two sends, maybe can fold into one if needed
            child_conn.send(('fill_data', changed))
            child_conn.send(('return', result))
        except KeyboardInterrupt:
            print("Child process received KeyboardInterrupt, exiting.")
            break
        except EOFError:
            print("Child process received EOF, exiting.")
            break
        except Exception as e:
            print(e, traceback.format_exc())
            child_conn.send(('error', e))


class _LazyFrameData(pl.FrameData):
    """
    A lazy FrameData that fetches fields from the parent process when needed.
    This is used to avoid sending all data fields immediately, allowing for
    more efficient communication.
    """
    def __init__(self, conn, *args, **kwargs):
        self._changed_fields = set()  # fields that have been changed in this process
        self._conn = conn
        super().__init__(*args, **kwargs)

    def set(self, key, value):
        self._changed_fields.add(key)
        super().set(key, value)

    def get(self, item):
        #print('getattr field:', item)
        try:
            return super().get(item)
        except KeyError:
            pass
        #print('   fetching field:', item)
        # Fetch the field from the parent process
        self._conn.send(('fetch_data', [item]))
        code, result = self._conn.recv()
        if code != 'fill_data':
            raise RuntimeError(f"Expected 'fill_data', got {code}")
        if isinstance(result, KeyError):
            raise result
        assert isinstance(result, pl.FrameData), f"Expected FrameData, got {type(result)}"
        #print('   setting field:', item, 'to', result[item])
        super().set(item, result[item])  # set without adding to _changed_fields
        return getattr(self, item)

    def has(self, key):
        if key in self._data:
            return True
        try:
            self.get(key)  # This will fetch the item if not present
        except KeyError:
            return False
        return key in self._data

    def _get_changed_data(self):
        data = pl.FrameData()
        for field in self._changed_fields:
            data[field] = self[field]
        return data


class _TestClass:
    def __init__(self, value):
        self.value = value

    def process(self, data):
        # Simulate some processing
        data.y = self.value + data.x
        return data


def test():

    client = start_process(
        _TestClass, 10
    )  # client is a proxy to MyClass in the child process
    data = pl.FrameData()
    data.x = 5  # Set some initial data
    client.process(data)  # Call the process method in the child process
    print(f"Processed data: y: {data.y}, x: {data.x}")
    assert data.y == 15, f"Expected 15, got {data.y}"  # Check the result
    assert data.x == 5, f"Expected 5, got {data.x}"  # Check that x is unchanged


if __name__ == "__main__":
    test()
