from typing import Dict, List
import tensorflow as tf
import numpy as np
import os
import json

from model.train.data_processing import load_motion_data, create_windows, create_label_windows
from model.train.model import train_model

def create_full_model_from_config(config: Dict):
    """
    Load data and train model from a config dict.

    Args:
        config: Config dictionary with keys:
            - sample_rate: int
            - motion_duration: float
            - motions: list of dicts with 'label' and 'data_file'
              (data_file paths should be absolute or relative to current working directory)
    """
    sample_rate = config["sample_rate"]
    motion_duration = config["motion_duration"]
    epochs = config.get("epochs", 10)            # default to 10 if missing
    batch_size = config.get("batch_size", 32)    # default to 32 if missing
    out_dir = config.get("output_dir", "./output")
    model_name = config.get("name", "model") 

    all_windows: List[List[List[float]]] = []
    all_labels: List[int] = []

    for motion in config["motions"]:
        label = motion["label"]
        data_file = motion["data_path"]

        data_path = os.path.join(os.path.dirname(__file__), data_file)

        # Load raw motion data from file
        data = load_motion_data(data_path)

        # Create windows of the motion data
        windows = create_windows(data, sample_rate, motion_duration)

        # Create labels for these windows
        labels = create_label_windows(len(windows), label)

        all_windows.extend(windows)
        all_labels.extend(labels)

    # Convert data and labels to NumPy arrays
    X = np.array(all_windows, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
                 
    # Train model on all data and labels
    model = train_model(X, y, epochs=epochs, batch_size=batch_size)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    os.makedirs(out_dir, exist_ok=True)

    model_path = os.path.join(out_dir, f"{model_name}.tflite")
    with open(model_path, 'wb') as f:
        f.write(tflite_model)


if __name__ == '__main__':
    # Path to your config JSON file
    config_path = os.path.join(os.path.dirname(__file__), 'active_config.json')
    
    # Load config dictionary from JSON file
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Call your full model training and saving function with the loaded config
    create_full_model_from_config(config)