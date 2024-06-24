import unittest
import numpy as np
from zoo.pooltool.image_representation import array_to_grayscale

class TestArrayToGrayscale(unittest.TestCase):

    def test_correct_input_conversion(self):
        test_data = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [123, 234, 56]]], dtype=np.uint8)
        expected_output = np.array([[76, 149, 29, 180]], dtype=np.uint8)
        grayscale_output = array_to_grayscale(test_data)
        np.testing.assert_array_equal(grayscale_output, expected_output)


if __name__ == '__main__':
    unittest.main()
