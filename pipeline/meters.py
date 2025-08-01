import logging
import random
import threading
import time

from pipeline.engine import Component, Drop, ts

__all__ = ["ThroughputMeter", "FixedRateLimiter", "AdaptiveRateLimiter"]


class ThroughputMeter(Component):
    '''
    Measures the throughput of the data stream in items per second.
    '''
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
    '''
    Limits the rate of processing to a fixed number of frames per second.

    If drop is True, drops frames that come in before the next release time.
    If drop is False, sleeps until the next release time so all frames are processed.
    '''
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
    '''
    Adaptively limits the rate of the stream to match the achievable throughput.
    Frames are dropped when downstream processing is slower than the incoming rate.

    The target release rate is adjusted based on the max queue size of downstream components,
    using a target queue size default of 0.1 to avoid pileup.

    Throughput is measured by the downstream_meter, which by defualt is the global meter
    of the pipeline engine that includes all components, but can be set to any other meter
    to include only specific components in rate limit.
    '''
    def __init__(
        self, downstream_meter=None, initial_rate=30, delta=0.1, target_qsize=0.1,
        drop=True, print_stats_interval=0
    ):
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
            self._downstream_components = self.engine.components_between(
                self, self.downstream_meter
            )
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
            logging.info(
                "THROUGHPUT TARGET: %s ACHIEVED: %s  QSIZE: %s, IN: %s, OUT: %s", rate,
                achieved_throughput, qsize, self._incoming_meter.get(),
                self._print_stats_outgoing_meter.get()
            )

        self.last_time = ts()
