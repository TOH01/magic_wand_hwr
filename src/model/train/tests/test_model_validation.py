import unittest
import os
import json
import numpy as np
import tensorflow as tf

from model.train.data_processing import load_motion_data, create_windows


class TestModelValidation(unittest.TestCase):
    def test_model_validation_on_motions(self):
        # Load validation config
        config_path = os.path.join(os.path.dirname(__file__), 'validation_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)

        model_name = config.get("model_name", "model")
        model_path = os.path.join(config["output_dir"], f"{model_name}.tflite")

        self.assertTrue(os.path.exists(model_path), f"Model file not found at: {model_path}")

        # Load and prepare the TFLite model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        sample_rate = config["sample_rate"]
        motion_duration = config["motion_duration"]

        # Loop through each test motion
        for motion in config["validation_motions"]:
            expected_label = motion["label"]
            data_path = os.path.join(os.path.dirname(config_path), motion["data_path"])
            confidence_threshold = motion.get("confidence", 0.8)

            self.assertTrue(os.path.exists(data_path), f"Test data file missing: {data_path}")

            data = load_motion_data(data_path)
            windows = create_windows(data, sample_rate, motion_duration)
            
            if not windows:
                self.fail(f"No windows created from motion file: {data_path}")

            for window in windows:
                input_data = np.array([window], dtype=np.float32)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])
                predicted_label = int(np.argmax(output[0]))
                
                print(f"Expected : {expected_label}, Got : {predicted_label}, Confidence Expected Label : {output[0][expected_label]}, Confidence Predicetd Label : {output[0][predicted_label]}")
                
                self.assertEqual(expected_label, predicted_label)
                self.assertGreaterEqual(output[0][expected_label], confidence_threshold)


if __name__ == '__main__':
    unittest.main()