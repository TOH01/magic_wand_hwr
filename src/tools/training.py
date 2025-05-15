import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import json


def load_single_motion_data(data_path : str, label : str, sample_rate : int, duration : float):
    data = np.loadtxt(data_path, delimiter=",", skiprows=1)  # (N, 6)

    window_size = int(sample_rate * duration)

    windows = []
    
    for i in range(0, len(data) - window_size + 1, window_size):
            window = data[i:i+window_size]
            # only append full windows
            if len(window) == window_size:
                windows.append(window)

    X = np.array(windows)  # shape: (num_windows, window_size, 6)
    y = np.full((len(X),), label)

    return


def load_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        headers = f.readline().strip().split(',')
        for line in f:
            values = line.strip().split(',')
            row = [float(v) for v in values]
            data.append(row)
    print(f"Loaded {len(data)} samples from {file_path}")
    print(f"Columns: {headers}")
    return np.array(data)

def create_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv1D(filters=8, kernel_size=5, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

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

def main(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    window_size = config["window_size"]
    output_dir = config["output_dir"]
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_segments = []
    all_labels = []

    for motion in config["motions"]:
        motion_data = load_data(motion["data_path"])
        motion_segments = [motion_data[i:i + window_size] for i in range(0, len(motion_data) - window_size, window_size // 2)]
        motion_labels = np.full(len(motion_segments), motion["label"])

        # Normalize segments based on duration and sample rate
        acc_mean, acc_std = np.mean(motion_segments[:, :, :3]), np.std(motion_segments[:, :, :3])
        gyro_mean, gyro_std = np.mean(motion_segments[:, :, 3:6]), np.std(motion_segments[:, :, 3:6])

        generate_normalization_header(acc_mean, acc_std, gyro_mean, gyro_std)
        
        motion_segments[:, :, :3] = (motion_segments[:, :, :3] - acc_mean) / acc_std
        motion_segments[:, :, 3:6] = (motion_segments[:, :, 3:6] - gyro_mean) / gyro_std
        
        all_segments.append(motion_segments)
        all_labels.append(motion_labels)
    
    X = np.concatenate(all_segments)
    y = np.concatenate(all_labels)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]
    
    # Split data
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print("Creating and training model...")
    model = create_model((window_size, 6))
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    print("Converting to TFLite model...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    model_path = os.path.join(output_dir, 'motion_detector.tflite')
    with open(model_path, 'wb') as f:
        f.write(tflite_model)

    header_path = os.path.join(output_dir, 'motion_detector_model.h')
    with open(header_path, 'w') as f:
        f.write("// Auto-generated model header\n\n")
        f.write("const unsigned char motion_detector_model[] = {\n  ")
        f.write(', '.join(f"0x{byte:02x}" for byte in tflite_model))
        f.write("\n};\n\n")
        f.write(f"const unsigned int motion_detector_model_len = {len(tflite_model)};\n")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])

    history_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(history_path)
    print(f"Training history saved to {history_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to the JSON configuration file")
    args = parser.parse_args()
    main(args.config_file)
