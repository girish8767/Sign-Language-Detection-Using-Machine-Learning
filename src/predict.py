import cv2
import numpy as np
import tensorflow as tf
from src.utils import mediapipe_detection, draw_landmarks, extract_keypoints
import mediapipe as mp
import os

mp_holistic = mp.solutions.holistic

def load_model():
    """
    Load the trained model
    """
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    model = tf.keras.models.load_model(os.path.join(models_dir, 'action.h5'))
    return model

def real_time_prediction(actions, model=None, threshold=0.5):
    """
    Make real-time predictions using the webcam
    """
    if model is None:
        model = load_model()
    
    sequence = []
    sentence = []
    predictions = []
    
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            
            # Draw landmarks
            draw_landmarks(image, results)
            
            # Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            # Changed from 30 to 20 to match the new sequence_length
            sequence = sequence[-10:]  
            
            if len(sequence) == 10:  # Changed from 30 to 20
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))
                
                # Visualization logic
                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])
                
                if len(sentence) > 5:
                    sentence = sentence[-5:]
                
                # Visualization
                cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, ' '.join(sentence), (3,30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Show to screen
            cv2.imshow('OpenCV Feed', image)
            
            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()