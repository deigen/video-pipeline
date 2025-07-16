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
    ctx = multiprocessing.get_context('spawn')

    parent_conn, child_conn = ctx.Pipe()

    process = ctx.Process(target=_child_process_server_loop, args=(cls, args, kwargs, child_conn))
    process.start()

    class _ClientProxy(ClientProxy):
        __name__ = f'ClientProxy<{cls.__name__}>'

    client = _ClientProxy(process, parent_conn, child_conn)

    return client


class ClientProxy:
    def __init__(self, process, parent_conn, child_conn):
        self._process = process
        self._parent_conn = parent_conn
        self._child_conn = child_conn
        self.wait_for_ready()

    def __getattr__(self, name):
        return lambda *args, **kwargs: self._call(name, args, kwargs)

    def _call(self, name, args, kwargs):
        self._parent_conn.send((name, args, kwargs))
        result = self._parent_conn.recv()
        if isinstance(result, Exception):
            raise result
        return result

    def wait_for_ready(self):
        # Wait for the child process to be ready
        if self._call('ready', None, None) != 'ready':
            raise RuntimeError("Child process did not signal readiness")

    def exit(self):
        self._parent_conn.send(('exit', None, None))
        self._process.join()


def _child_process_server_loop(cls, args, kwargs, child_conn):
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


class _TestClass:
    def __init__(self, value):
        self.value = value

    def f(self, x):
        return self.value + x


def test():

    client = Process(_TestClass, 10)  # client is a proxy to MyClass in the child process
    result = client.f(5)  # result is 15, computed in the child process
    assert result == 15, f"Expected 15, got {result}"
    client.exit()
    print("Done")

if __name__ == "__main__":
    test()
