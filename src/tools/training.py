import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse

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


def main(circle_path, non_circle_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    window_size = 75

    print("Loading circle data...")
    circle_data = load_data(circle_path)
    segments = [circle_data[i:i + window_size] for i in range(0, len(circle_data) - window_size, window_size // 2)]
    circle_labels = np.ones(len(segments))

    print("Loading non-circle data...")
    non_circle_data = load_data(non_circle_path)
    non_circle_segments = [non_circle_data[i:i + window_size] for i in range(0, len(non_circle_data) - window_size, window_size // 2)]
    non_circle_labels = np.zeros(len(non_circle_segments))

    X = np.concatenate([segments, non_circle_segments])
    y = np.concatenate([circle_labels, non_circle_labels])
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    acc_mean, acc_std = np.mean(X_train[:, :, :3]), np.std(X_train[:, :, :3])
    gyro_mean, gyro_std = np.mean(X_train[:, :, 3:6]), np.std(X_train[:, :, 3:6])

    generate_normalization_header(acc_mean, acc_std, gyro_mean, gyro_std)

    X_train[:, :, :3] = (X_train[:, :, :3] - acc_mean) / acc_std
    X_train[:, :, 3:6] = (X_train[:, :, 3:6] - gyro_mean) / gyro_std
    X_test[:, :, :3] = (X_test[:, :, :3] - acc_mean) / acc_std
    X_test[:, :, 3:6] = (X_test[:, :, 3:6] - gyro_mean) / gyro_std

    print("Creating and training model...")
    model = create_model((window_size, 6))
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    print("Converting to TFLite model...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    model_path = os.path.join(out_dir, 'circle_detector.tflite')
    with open(model_path, 'wb') as f:
        f.write(tflite_model)

    header_path = os.path.join(main_dir, 'src/model/circle_detector_model.h')
    with open(header_path, 'w') as f:
        f.write("// Auto-generated model header\n\n")
        f.write("const unsigned char circle_detector_model[] = {\n  ")
        f.write(', '.join(f"0x{byte:02x}" for byte in tflite_model))
        f.write("\n};\n\n")
        f.write(f"const unsigned int circle_detector_model_len = {len(tflite_model)};\n")

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

    history_path = os.path.join(out_dir, 'training_history.png')
    plt.savefig(history_path)
    print(f"Training history saved to {history_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("circle_path", type=str, help="Path to the significant gesture data CSV")
    parser.add_argument("non_circle_path", type=str, help="Path to the random data CSV")
    script_dir = os.path.dirname(os.path.abspath(__file__))

    main_dir = os.path.dirname(os.path.dirname(script_dir))  # Go up to main directory
    default_out_dir = os.path.join(main_dir, 'out')  # Output in main/out
    parser.add_argument("--out_dir", type=str, default=default_out_dir, help="Output directory")
    args = parser.parse_args()
    main(args.circle_path, args.non_circle_path, args.out_dir)