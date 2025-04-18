from sklearn.model_selection import train_test_split
import numpy as np
import os
from src.model import create_model, load_data
import tensorflow as tf

def train_model(actions, epochs=2000):
    """
    Train the sign language detection model
    """
    # Load data
    X, y = load_data(actions)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
    
    # Create model
    input_shape = (X.shape[1], X.shape[2])
    model = create_model(np.array(actions), input_shape)
    
    # Create logs directory
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Train model
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    model.fit(X_train, y_train, epochs=epochs, callbacks=[tb_callback])
    
    # Evaluate model
    model.evaluate(X_test, y_test)
    
    # Save model
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)
    model.save(os.path.join(models_dir, 'action.h5'))
    
    return model