import os
import json
import unittest
from model.train.main import create_full_model_from_config

class TestModelTraining(unittest.TestCase):
    def test_create_full_model_from_config(self):
        # Path to test config
        config_path = os.path.join(os.path.dirname(__file__), 'config_for_tests.json')

        # Ensure config exists
        self.assertTrue(os.path.exists(config_path), "Config file does not exist")

        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Call the training pipeline
        create_full_model_from_config(config)

        # Check that model file was created
        expected_output = os.path.join(config["output_dir"], f"{config['name']}.tflite")
        self.assertTrue(os.path.isfile(expected_output), f"Expected model file not found: {expected_output}")

if __name__ == '__main__':
    unittest.main()
