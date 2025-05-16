import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

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
        tf.keras.layers.Input(shape=(window_size, num_features)),
        tf.keras.layers.Conv1D(16, 3, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model(X_train: np.ndarray, y_train: np.ndarray,X_test: np.ndarray, y_test: np.ndarray, epochs=10, batch_size=32):
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
    window_size = X_train.shape[1]
    num_features = X_train.shape[2]
    num_classes = len(set(y_train))
    
    model = build_model(window_size, num_features, num_classes)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    print(history.history)         

    return model

