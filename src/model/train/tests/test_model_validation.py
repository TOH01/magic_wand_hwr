import unittest
import os
import json
import numpy as np
import tensorflow as tf

from model.train.data_processing import load_motion_data, create_windows
from model.train.main import create_full_model_from_config, load_normalization_params


class TestModelValidation(unittest.TestCase):
    def test_model_validation_on_motions(self):

        config_training_path = os.path.join(os.path.dirname(__file__), 'config_for_tests.json')
        print(f"Loading training config from: {config_training_path}")
        # Load training config
        with open(config_training_path, 'r') as f:
            config_training = json.load(f)

        print("Starting training pipeline with training config...")
        create_full_model_from_config(config_training)
        print("Training pipeline finished.")

        model_name = config_training.get("name", "model")
        model_path = os.path.join(config_training["output_dir"], f"{model_name}.tflite")
        print(f"Looking for TFLite model at: {model_path}")
        self.assertTrue(os.path.exists(model_path), f"Model file not found at: {model_path}")

        header_path = os.path.join(config_training["output_dir"], "normalization.h")
        print(f"Loading normalization parameters from: {header_path}")
        norm_params = load_normalization_params(header_path)

        acc_mean = norm_params["ACC_MEAN"]
        acc_std = norm_params["ACC_STD"]
        gyro_mean = norm_params["GYRO_MEAN"]
        gyro_std = norm_params["GYRO_STD"]

        print(f"Normalization params:\n ACC mean: {acc_mean}, std: {acc_std}\n Gyro mean: {gyro_mean}, std: {gyro_std}")

        # Load and prepare the TFLite model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        sample_rate = config_training["sample_rate"]
        motion_duration = config_training["motion_duration"]

        print(f"Sample rate: {sample_rate}, Motion duration: {motion_duration}")

        # Loop through each test motion
        for motion in config_training["motions"]:
            expected_label = motion["label"]
            
            if not motion.get("validation_data_path"):
                print(f"Skipping motion '{motion.get('name', motion.get('name'))}' — no validation_data_path provided.")
                continue
            
            data_path = os.path.join(os.path.dirname(config_training_path), motion["validation_data_path"])
            confidence_threshold = motion.get("confidence", 0.8)

            print(f"\nTesting motion with label={expected_label} from file: {data_path}")
            print(f"Confidence threshold set to: {confidence_threshold}")
            self.assertTrue(os.path.exists(data_path), f"Test data file missing: {data_path}")

            data = load_motion_data(data_path)
            print(f"Loaded raw data shape: {len(data)} samples")

            windows = create_windows(data, sample_rate, motion_duration)
            print(f"Created {len(windows)} windows from test data")

            if not windows:
                self.fail(f"No windows created from motion file: {data_path}")

            for i, window in enumerate(windows):
                window = np.array(window, dtype=np.float32)
                # Normalize window data
                window[:, :3] = (window[:, :3] - acc_mean) / acc_std
                window[:, 3:6] = (window[:, 3:6] - gyro_mean) / gyro_std

                input_data = np.array([window], dtype=np.float32)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])

                predicted_label = int(np.argmax(output[0]))
                expected_confidence = output[0][expected_label]
                predicted_confidence = output[0][predicted_label]

                print(f"Window {i+1}/{len(windows)}:")
                print(f"  Expected label: {expected_label}")
                print(f"  Predicted label: {predicted_label}")
                print(f"  Confidence for expected label: {expected_confidence:.4f}")
                print(f"  Confidence for predicted label: {predicted_confidence:.4f}")
                print(f"  Raw output vector: {output[0]}")

                self.assertEqual(expected_label, predicted_label,
                                 f"Prediction mismatch on window {i+1}: expected {expected_label}, got {predicted_label}")
                self.assertGreaterEqual(expected_confidence, confidence_threshold,
                                        f"Confidence below threshold on window {i+1}: {expected_confidence} < {confidence_threshold}")

if __name__ == '__main__':
    unittest.main()