import cv2
import numpy as np
import os
from src.utils import mediapipe_detection, draw_landmarks, extract_keypoints
import mediapipe as mp
import time

mp_holistic = mp.solutions.holistic

def create_folders(actions):
    """
    Create folders for data collection
    """
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    for action in actions:
        # Reduced from 30 to 15 sequences
        for sequence in range(10):
            try:
                os.makedirs(os.path.join(data_path, action, str(sequence)))
            except:
                pass

def collect_data(actions, no_sequences=10, sequence_length=10):
    """
    Collect data for sign language detection
    """
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    
    # Try to open the camera
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        print("Please check your camera permissions in System Preferences > Security & Privacy > Camera")
        print("After enabling permissions, restart the application.")
        return
    
    # Wait a moment for camera to initialize
    time.sleep(1)
    
    # Read a test frame
    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        print("Error: Could not read from camera.")
        print("Please check your camera connections and permissions.")
        cap.release()
        return
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in actions:
            for sequence in range(no_sequences):
                for frame_num in range(sequence_length):
                    ret, frame = cap.read()
                    
                    # Check if frame was successfully captured
                    if not ret or frame is None:
                        print("Error: Failed to capture frame.")
                        continue
                    
                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)
                    
                    # Draw landmarks
                    draw_landmarks(image, results)
                    
                    # Apply wait logic
                    if frame_num == 0:
                        cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15, 12),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(2000)
                    else:
                        cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15, 12),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)
                    
                    # Export keypoints
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(data_path, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)
                    
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
        
        cap.release()
        cv2.destroyAllWindows()