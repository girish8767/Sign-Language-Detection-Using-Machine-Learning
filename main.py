import os
import numpy as np
from src.data_collection import create_folders, collect_data
from src.train import train_model
from src.predict import real_time_prediction

# In the main.py file, we need to make sure the model training is adjusted for the reduced data

def main():
    # Define actions
    actions = np.array(['hello', 'thanks', 'iloveyou'])
    
    # Create folders for data collection
    create_folders(actions)
    
    # Ask user what they want to do
    print("Sign Language Detection System")
    print("1. Collect Data")
    print("2. Train Model")
    print("3. Real-time Prediction")
    choice = input("Enter your choice (1-3): ")
    
    if choice == '1':
        # Pass the reduced parameters
        collect_data(actions, no_sequences=10, sequence_length=10)
    elif choice == '2':
        # Reduce epochs since we have less data
        model = train_model(actions, epochs=1000)
        print("Model trained and saved successfully!")
    elif choice == '3':
        real_time_prediction(actions)
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()