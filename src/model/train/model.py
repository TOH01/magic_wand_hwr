import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(window_size: int, num_features: int, num_classes: int) -> tf.keras.Model:
    """
    Build a simple model for motion data classification.

    Args:
        window_size: Number of time steps in each window.
        num_features: Number of features per time step (e.g., 6).
        num_classes: Number of motion classes.

    Returns:
        Compiled Keras model.
    """
    model = models.Sequential([
        layers.Input(shape=(window_size, num_features)),
        layers.Conv1D(32, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(64, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_model(X: np.ndarray, y: np.ndarray, epochs=10, batch_size=32):
    """
    Train the model on the windowed motion data.

    Args:
        X: Numpy array of shape (num_samples, window_size, num_features).
        y: Numpy array of integer labels, shape (num_samples,).
        epochs: Number of training epochs.
        batch_size: Batch size.

    Returns:
        Trained Keras model.
    """
    window_size = X.shape[1]
    num_features = X.shape[2]
    num_classes = len(set(y))

    model = build_model(window_size, num_features, num_classes)
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    print(history.history)         

    return model

