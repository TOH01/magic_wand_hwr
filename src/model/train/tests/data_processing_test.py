import unittest
from model.train.data_processing import load_motion_data
from model.train.data_processing import create_windows
from model.train.data_processing import create_label_windows
from model.train.data_processing import load_dataset_from_config
from typing import List
import os
import json


class TestLoadMotionData(unittest.TestCase):
    def test_load_motion_data(self):
        test_file = os.path.join(os.path.dirname(__file__), 'training_data/data_for_tests.txt')

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
        
        test_file = os.path.join(os.path.dirname(__file__), 'training_data/data_for_tests.txt')

        data = load_motion_data(test_file)

        data_long = data * 10
        
        self.assertEqual(len(data_long), 80)

        sample_rate = 6
        motion_length = 2.0

        windows = create_windows(data_long, sample_rate, motion_length)
        
        # Expect 6 window because 2*6=12 samples per window, we have 80 samples, so 6 full window + leftover (ignored)
        self.assertEqual(len(windows), 6)
        # Expect first window to have 12 samples
        self.assertEqual(len(windows[0]), 12)


class TestCreateLabelWindows(unittest.TestCase):
    def test_label_windows(self):
        num_windows = 4
        label = 7
        labels = create_label_windows(num_windows, label)

        # Check the length matches number of windows
        self.assertEqual(len(labels), num_windows)

        # Check all labels are the expected integer
        for l in labels:
            self.assertEqual(l, label)

class TestLoadDatasetFromConfigDict(unittest.TestCase):
    def test_load_dataset_from_real_config(self):
        # Path to test config
        config_path = os.path.join(os.path.dirname(__file__), 'config_for_tests.json')

        # Load config dictionary
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Call function
        X, y = load_dataset_from_config(config)
        
        # Basic shape checks
        self.assertIsInstance(X, list)
        self.assertIsInstance(y, list)
        self.assertGreater(len(X), 0)
        self.assertEqual(len(X), len(y))

        # Check that each sample has correct window shape
        expected_window_size = int(config["sample_rate"] * config["motion_duration"])
        for window in X:
            self.assertEqual(len(window), expected_window_size)
            for sample in window:
                self.assertEqual(len(sample), 6)  # [aX, aY, aZ, gX, gY, gZ]

        # Check labels match allowed values
        valid_labels = {motion["label"] for motion in config["motions"]}
        for label in y:
            self.assertIn(label, valid_labels)
            
        # Check random sample
        test_motion = config["motions"][0]
        expected_label = test_motion["label"]
        
        # Load file directly, to avoid path issues
        test_file = os.path.join(os.path.dirname(__file__), "training_data/circle_test_data.txt")
        raw_data = load_motion_data(test_file)

        expected_windows = create_windows(raw_data, config["sample_rate"], config["motion_duration"])

        # Find first set of windows with matching label
        matched = [(i, x) for i, (x, lbl) in enumerate(zip(X, y)) if lbl == expected_label]

        # Only compare as many windows as exist in the manual data
        compare_count = min(len(expected_windows), len(matched))
        for i in range(compare_count):
            self.assertEqual(matched[i][1], expected_windows[i])


if __name__ == '__main__':
    unittest.main()