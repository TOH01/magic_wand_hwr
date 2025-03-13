import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# Simple function to load data from CSV
def load_data(file_path):
    # Load the data manually
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


# Create a simple model
def create_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv1D(filters=8, kernel_size=5, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


# Main function
def main():
    # Parameters
    window_size = 75  # 1 second of data at 75Hz

    # Load circle data
    print("Loading circle data...")
    circle_data = load_data("circle.csv")

    # Basic preprocessing
    # Assume columns are: aX, aY, aZ, gX, gY, gZ

    # Create windows for circle data
    segments = []
    for i in range(0, len(circle_data) - window_size, window_size // 2):
        segment = circle_data[i:i + window_size]
        segments.append(segment)

    segments = np.array(segments)
    print(f"Created {len(segments)} windows of size {window_size}")

    # Create labels (all 1 for circle data)
    circle_labels = np.ones(len(segments))

    # Create some random non-circle data
    print("Creating synthetic non-circle data...")
    non_circle_segments = np.random.randn(*segments.shape) * np.std(segments) + np.mean(segments)
    non_circle_labels = np.zeros(len(non_circle_segments))

    # Combine data
    X = np.concatenate([segments, non_circle_segments])
    y = np.concatenate([circle_labels, non_circle_labels])

    # Shuffle the data
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # Split into train and test
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Calculate normalization parameters (for accelerometer and gyroscope)
    acc_data = X_train[:, :, :3]
    gyro_data = X_train[:, :, 3:6]

    acc_mean = np.mean(acc_data)
    acc_std = np.std(acc_data)
    gyro_mean = np.mean(gyro_data)
    gyro_std = np.std(gyro_data)

    print("Normalization parameters:")
    print(f"acc_mean = {acc_mean}f;")
    print(f"acc_std = {acc_std}f;")
    print(f"gyro_mean = {gyro_mean}f;")
    print(f"gyro_std = {gyro_std}f;")

    # Normalize the data
    X_train[:, :, :3] = (X_train[:, :, :3] - acc_mean) / acc_std
    X_train[:, :, 3:6] = (X_train[:, :, 3:6] - gyro_mean) / gyro_std
    X_test[:, :, :3] = (X_test[:, :, :3] - acc_mean) / acc_std
    X_test[:, :, 3:6] = (X_test[:, :, 3:6] - gyro_mean) / gyro_std

    # Create and train model
    print("Creating and training model...")
    model = create_model((window_size, 6))
    print(model.summary())

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_test, y_test)
    )

    # Convert to TensorFlow Lite model
    print("Converting to TFLite model...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    #converter.optimizations = [tf.lite.Optimize.DEFAULT] this optimazion will break modul for arduino
    tflite_model = converter.convert()

    # Save the TF Lite model
    with open('circle_detector.tflite', 'wb') as f:
        f.write(tflite_model)

    # Generate C header
    print("Generating C header file...")
    with open('circle_detector_model.h', 'w') as f:
        f.write("// Auto-generated model header\n\n")
        f.write("const unsigned char circle_detector_model[] = {\n  ")

        # Write bytes in hex format
        byte_count = 0
        for byte in tflite_model:
            f.write(f"0x{byte:02x}, ")
            byte_count += 1
            if byte_count % 12 == 0:
                f.write("\n  ")

        f.write("\n};\n\n")
        f.write(f"const unsigned int circle_detector_model_len = {len(tflite_model)};\n")

    print(f"TensorFlow Lite model size: {len(tflite_model) / 1024:.2f} KB")

    # Plot training history
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

    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history saved to 'training_history.png'")

    print("Done! You can now upload the model to your Arduino.")


if __name__ == "__main__":
    main()