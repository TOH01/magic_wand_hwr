import unittest
from training import load_motion_data
from training import create_windows
from typing import List

class TestLoadMotionData(unittest.TestCase):
    def test_load_motion_data(self):
        test_file = 'training_test_data.txt'

        expected_data: List[List[float]] = [
            [0.569, -0.370, 2.741, -126.099, 49.622, 49.927],
            [0.443, -0.165, 2.833, -127.197, 82.458, 41.931],
            [0.186, -0.086, 2.527, 3.845, 169.373, 15.442],
            [-0.218, 0.051, 1.379, 54.199, 189.087, 30.334],
            [-0.195, 0.241, 0.980, 48.767, 177.734, 54.443],
            [-0.147, 0.577, 0.812, -149.231, 114.197, 50.110],
            [-0.148, 1.092, 0.834, -227.051, 106.812, 4.517],
            [-0.637, 1.811, -0.156, -208.740, 135.620, -60.669],
        ]

        data = load_motion_data(test_file)
        self.assertEqual(len(data), len(expected_data))
        for row_loaded, row_expected in zip(data, expected_data):
            for val_loaded, val_expected in zip(row_loaded, row_expected):
                self.assertAlmostEqual(val_loaded, val_expected, places=6)


class TestCreateWindows(unittest.TestCase):
    def test_create_windows_basic(self):
        # Use loaded data from file as base
        data = load_motion_data('training_test_data.txt')

        data_long = data * 10
        
        self.assertEqual(len(data_long), 80)

        sample_rate = 6
        motion_length = 2.0

        windows = create_windows(data_long, sample_rate, motion_length)
        
        # Expect 6 window because 2*6=12 samples per window, we have 80 samples, so 6 full window + leftover (ignored)
        self.assertEqual(len(windows), 6)
        # Expect first window to have 12 samples
        self.assertEqual(len(windows[0]), 12)


if __name__ == '__main__':
    unittest.main()