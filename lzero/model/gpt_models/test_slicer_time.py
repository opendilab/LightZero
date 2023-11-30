import torch
import timeit
import unittest
import numpy as np
from random import randint
from slicer import Slicer, Head, Embedder

class TestSlicerTime(unittest.TestCase):
    def setUp(self):
        self.num_tests = 100000  # Number of times to run the test
        self.max_blocks = 20


    def test_slicer_time_act_tokens_pattern(self):
        act_tokens_pattern = torch.zeros(17)  # 17
        act_tokens_pattern[-1] = 1   # [0,...,0,1]
        self.block_mask = act_tokens_pattern
        self.slicer = Slicer(self.max_blocks, self.block_mask)
        self.test_times_act_tokens_pattern = []
        for _ in range(self.num_tests):
            num_steps = randint(0, 17)  # Random test values
            prev_steps = randint(0, 20*17) 

            start_time = timeit.default_timer()
            # Test slice computation
            slice_ = self.slicer.compute_slice(num_steps=num_steps,  prev_steps=prev_steps)
            end_time = timeit.default_timer()

            self.test_times_act_tokens_pattern.append(end_time - start_time)

            # Add your assertions here
            # assert torch.equal(slice_, torch.tensor([]))

            # Test caching
            # cache_key = (num_steps, prev_steps)
            # assert cache_key in self.slicer.cache
            # assert torch.equal(slice_, self.slicer.cache[cache_key])

        avg_time = np.mean(self.test_times_act_tokens_pattern)
        std_dev = np.std(self.test_times_act_tokens_pattern)
        print(f"For act_tokens_pattern: Average execution time: {avg_time} seconds, standard deviation: {std_dev} seconds")

    def test_slicer_time_all_but_last_obs_tokens_pattern(self):
        all_but_last_obs_tokens_pattern = torch.ones(17)
        all_but_last_obs_tokens_pattern[-2] = 0  # [1,...,0,1]
        self.block_mask = all_but_last_obs_tokens_pattern
        self.slicer = Slicer(self.max_blocks, self.block_mask)
        self.test_times_all_but_last_obs_tokens_pattern = []
        for _ in range(self.num_tests):
            num_steps = randint(0, 17)  # Random test values
            prev_steps = randint(0, 20*17) 

            start_time = timeit.default_timer()
            # Test slice computation
            slice_ = self.slicer.compute_slice(num_steps=num_steps,  prev_steps=prev_steps)
            end_time = timeit.default_timer()

            self.test_times_all_but_last_obs_tokens_pattern.append(end_time - start_time)

            # Add your assertions here
            # assert torch.equal(slice_, torch.tensor([]))

            # Test caching
            # cache_key = (num_steps, prev_steps)
            # assert cache_key in self.slicer.cache
            # assert torch.equal(slice_, self.slicer.cache[cache_key])

        avg_time = np.mean(self.test_times_all_but_last_obs_tokens_pattern)
        std_dev = np.std(self.test_times_all_but_last_obs_tokens_pattern)
        print(f"For all_but_last_obs_tokens_pattern: Average execution time: {avg_time} seconds, standard deviation: {std_dev} seconds")

if __name__ == "__main__":
    unittest.main()