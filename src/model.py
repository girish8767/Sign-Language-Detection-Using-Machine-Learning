import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import os

def create_model(actions, input_shape):
    """
    Create LSTM model for sign language detection
    """
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=input_shape))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
    return model

def load_data(actions):
    """
    Load data from the data directory
    """
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    
    sequences, labels = [], []
    for action_idx, action in enumerate(actions):
        for sequence in range(15):  # Changed from 30 to 15 to match our reduced data collection
            window = []
            for frame_num in range(20):  # Changed from 30 to 20 frames
                try:
                    res = np.load(os.path.join(data_path, action, str(sequence), f"{frame_num}.npy"))
                    window.append(res)
                except FileNotFoundError:
                    print(f"Warning: File not found: {os.path.join(data_path, action, str(sequence), f'{frame_num}.npy')}")
                    # If file not found, use zeros with the same shape as expected keypoints
                    # Typical keypoints shape is 1662 (33*4 + 468*3 + 21*3 + 21*3)
                    res = np.zeros(1662)
                    window.append(res)
            sequences.append(window)
            labels.append(action_idx)  # Use enumerate index instead of .index() method
    
    X = np.array(sequences)
    y = tf.keras.utils.to_categorical(labels).astype(int)
    
    return X, y