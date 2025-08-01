import unittest

import pipeline as pl


class TestComponents(unittest.TestCase):
    def test_0010_function_component(self):
        def sample_function(data):
            data.result = "Processed"

        function_component = pl.Function(sample_function)

        # Test processing
        data = pl.FrameData()
        function_component.process(data)
        self.assertEqual(data.result, "Processed")

    def test_0020_counter_component(self):
        counter = pl.Counter()

        # Test processing
        data = pl.FrameData()
        for i in range(5):
            counter.process(data)
            assert data.count == i

    def test_0030_sleep_component(self):

        sleep_component = pl.Sleep(0.1)

        # Test processing
        data = pl.FrameData()
        sleep_component.process(data)

    def test_0040_print_component(self):
        print_component = pl.Print(message="Data: {data}", interval=0.1)

        # Test processing
        data = pl.FrameData()
        print_component.process(data)

    def test_0050_limit_num_frames_component(self):
        limit_component = pl.LimitNumFrames(num_frames=3)

        # Test processing
        data = pl.FrameData()
        for i in range(5):
            if i < 3:
                limit_component.process(data)
            else:
                with self.assertRaises(pl.StreamEnd):
                    limit_component.process(data)

    @unittest.skip("Breakpoint pdb functionality is not testable in this environment")
    def test_0060_breakpoint_component(self):
        breakpoint_component = pl.Breakpoint()

        # Test processing
        data = pl.FrameData()
        # This will trigger a breakpoint in the code, which cannot be tested here
        breakpoint_component.process(data)

    def test_0070_fixed_rate_limiter_component(self):
        limiter = pl.FixedRateLimiter(rate=2, drop=True)

        # TODO: check the rate matches the expected rate by e.g. patching ts()
        # Test processing
        data = pl.FrameData()
        for i in range(5):
            try:
                limiter.process(data)
            except pl.Drop:
                continue

    def test_0080_throughput_meter_component(self):
        meter = pl.ThroughputMeter(
            update_interval=0.0
        )  # 0 means update every process call
        # Test processing
        data = pl.FrameData()
        for i in range(5):
            meter.process(data)
            self.assertGreater(meter.get(), 0)

    @unittest.skip("TODO AdaptiveRateLimiter needs engine to be tested")
    def test_0090_adaptive_rate_limiter_component(self):
        meter = pl.ThroughputMeter(
            update_interval=0.0
        )  # 0 means update every process call
        limiter = pl.AdaptiveRateLimiter(meter)

        for i in range(5):
            data = pl.FrameData()
            meter.process(data)
            try:
                limiter.process(data)
            except pl.Drop:
                continue

    def test_0100_multiprocess_component(self):
        multiprocess_component = pl.Multiprocess(pl.Counter)

        # TODO test function or mock pipeline engine that calls this
        multiprocess_component.pipeline_thread_init()

        # Test processing
        data = pl.FrameData()
        for i in range(5):
            multiprocess_component.process(data)
            self.assertEqual(data.count, i)

    def test_1000_pipeline(self):
        # Create a minimal pipeline with one component
        counter = pl.Counter()
        engine = pl.PipelineEngine()
        engine.add(counter)
        engine.run_until(lambda: counter.count >= 10)


if __name__ == "__main__":
    unittest.main()
