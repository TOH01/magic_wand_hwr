from typing import Dict, List
import tensorflow as tf
import numpy as np
import os
import json
import re

from model.train.data_processing import load_motion_data, create_windows, create_label_windows
from model.train.model import train_model

import re

def load_normalization_params(header_path):
    params = {}
    with open(header_path, 'r') as f:
        content = f.read()
        params["ACC_MEAN"] = float(re.search(r"#define ACC_MEAN ([\d\.\-eE]+)", content).group(1))
        params["ACC_STD"] = float(re.search(r"#define ACC_STD ([\d\.\-eE]+)", content).group(1))
        params["GYRO_MEAN"] = float(re.search(r"#define GYRO_MEAN ([\d\.\-eE]+)", content).group(1))
        params["GYRO_STD"] = float(re.search(r"#define GYRO_STD ([\d\.\-eE]+)", content).group(1))
    return params

def generate_normalization_header(acc_mean, acc_std, gyro_mean, gyro_std, file_name='../model/normalization_params.h'):
    with open(file_name, 'w') as f:
        f.write('#ifndef NORMALIZATION_PARAMS_H\n')
        f.write('#define NORMALIZATION_PARAMS_H\n\n')
        
        # Write normalization parameters as defines
        f.write('// Normalization parameters for accelerometer and gyroscope data, automatically generated when training model\n')
        
        # Accelerometer mean and std
        f.write(f'#define ACC_MEAN {acc_mean:.6f}\n')
        f.write(f'#define ACC_STD {acc_std:.6f}\n')
        
        # Gyroscope mean and std
        f.write(f'#define GYRO_MEAN {gyro_mean:.6f}\n')
        f.write(f'#define GYRO_STD {gyro_std:.6f}\n')
        
        f.write('\n#endif // NORMALIZATION_PARAMS_H\n')
        

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
    epochs = config.get("epochs", 10)
    batch_size = config.get("batch_size", 32)
    out_dir = config.get("output_dir", "./output")
    model_name = config.get("name", "model")

    print(f"Config summary:")
    print(f" Sample rate: {sample_rate}")
    print(f" Motion duration: {motion_duration}")
    print(f" Epochs: {epochs}, Batch size: {batch_size}")
    print(f" Output dir: {out_dir}")
    print(f" Model name: {model_name}")

    all_windows: List[List[List[float]]] = []
    all_labels: List[int] = []

    for motion in config["motions"]:
        label = motion["label"]
        data_file = motion["data_path"]

        data_path = os.path.join(os.path.dirname(__file__), data_file)
        print(f"\nLoading motion data for label={label} from: {data_path}")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Motion data file not found: {data_path}")

        # Load raw motion data from file
        data = load_motion_data(data_path)
        print(f"  Raw data shape: {len(data)} samples")
        if len(data) == 0:
            raise ValueError(f"No data loaded from {data_path}")

        # Create windows of the motion data
        windows = create_windows(data, sample_rate, motion_duration)
        print(f"  Created {len(windows)} windows from data")

        if len(windows) == 0:
            raise ValueError(f"No windows created for motion label {label} from file {data_path}")

        # Create labels for these windows
        labels = create_label_windows(len(windows), label)

        all_windows.extend(windows)
        all_labels.extend(labels)

    print(f"\nTotal windows collected: {len(all_windows)}")
    print(f"Total labels collected: {len(all_labels)}")
    
    # Convert data and labels to NumPy arrays
    X = np.array(all_windows, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    
    print(f"\nShape of input data array X: {X.shape}")
    print(f"Shape of labels array y: {y.shape}")

    # Check label distribution
    unique, counts = np.unique(y, return_counts=True)
    label_distribution = dict(zip(unique, counts))
    print(f"Label distribution in dataset: {label_distribution}")

    # Shuffle dataset with fixed seed for reproducibility
    indices = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(indices)

    X = X[indices]
    y = y[indices]
    
    split = int(0.8 * len(X))
    
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # Calculate normalization parameters from training data
    acc_mean, acc_std = np.mean(X_train[:, :, :3]), np.std(X_train[:, :, :3])
    gyro_mean, gyro_std = np.mean(X_train[:, :, 3:6]), np.std(X_train[:, :, 3:6])
    
    print(f"\nNormalization parameters:")
    print(f" ACC mean: {acc_mean:.5f}, ACC std: {acc_std:.5f}")
    print(f" Gyro mean: {gyro_mean:.5f}, Gyro std: {gyro_std:.5f}")

    # Normalize training and test data
    X_train[:, :, :3] = (X_train[:, :, :3] - acc_mean) / acc_std
    X_train[:, :, 3:6] = (X_train[:, :, 3:6] - gyro_mean) / gyro_std
    X_test[:, :, :3] = (X_test[:, :, :3] - acc_mean) / acc_std
    X_test[:, :, 3:6] = (X_test[:, :, 3:6] - gyro_mean) / gyro_std

    # Check sample data after normalization
    print("\nSample normalized training data (first window, first 3 time steps):")
    print(X_train[0, :3, :])

    # Print label distribution again in train and test
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    print(f"\nTraining label distribution: {dict(zip(unique_train, counts_train))}")
    print(f"Test label distribution: {dict(zip(unique_test, counts_test))}")

    os.makedirs(out_dir, exist_ok=True)     

    generate_normalization_header(acc_mean, acc_std, gyro_mean, gyro_std, os.path.join(out_dir, "normalization.h"))      

    # Train model on all data and labels
    print("\nStarting model training...")
    model = train_model(X_train, y_train, X_test, y_test, epochs=epochs, batch_size=batch_size)
    print("Training complete.")

    # Convert model to TFLite
    print("Converting model to TensorFlow Lite format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    print("Conversion complete.")

    model_path = os.path.join(out_dir, f"{model_name}.tflite")
    with open(model_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Model saved to: {model_path}")


if __name__ == '__main__':
    # Path to your config JSON file
    config_path = os.path.join(os.path.dirname(__file__), 'active_config.json')
    
    # Load config dictionary from JSON file
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Call your full model training and saving function with the loaded config
    create_full_model_from_config(config)