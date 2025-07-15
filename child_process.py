'''
Server that proxies a class in a child process.

Example usage:

class MyClass:
    def __init__(self, value):
        # init runs in the child process
        self.value = value
        
    def f(self, x):
        # method runs in the child process
        return self.value + x

client = child_process.Process(MyClass, 10)  # client is a proxy to MyClass in the child process
result = client.f(5)  # result is 15, computed in the child process
'''

import os
import sys
import multiprocessing

def Process(cls, *args, **kwargs):
    parent_conn, child_conn = multiprocessing.Pipe()

    def child_process():
        instance = cls(*args, **kwargs)
        while True:
            try:
                method_name, method_args, method_kwargs = child_conn.recv()
                if method_name == 'exit':
                    break
                if method_name == 'ready':
                    child_conn.send('ready')
                    continue
                method = getattr(instance, method_name)
                result = method(*method_args, **method_kwargs)
                child_conn.send(result)
            except Exception as e:
                child_conn.send(e)

    process = multiprocessing.Process(target=child_process)
    process.start()

    class ClientProxy:
        __name__ = f'ClientProxy<{cls.__name__}>'

        def __init__(self):
            self._process = process
            self._parent_conn = parent_conn
            self._child_conn = child_conn
            self.wait_for_ready()

        def __getattr__(self, name):
            return lambda *args, **kwargs: self._call(name, args, kwargs)

        def _call(self, name, args, kwargs):
            parent_conn.send((name, args, kwargs))
            result = parent_conn.recv()
            if isinstance(result, Exception):
                raise result
            return result

        def wait_for_ready(self):
            # Wait for the child process to be ready
            if not parent_conn.poll(5):
                raise RuntimeError("Child process did not start in time")
            if self._call('ready', None, None) != 'ready':
                raise RuntimeError("Child process did not signal readiness")

        def exit(self):
            parent_conn.send(('exit', None, None))
            process.join()

    client = ClientProxy()

    return client
