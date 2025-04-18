# Sign Language Detection Using Machine Learning

This project uses computer vision and machine learning to detect and interpret sign language gestures in real-time using a webcam.

## Project Overview

The system uses MediaPipe for hand landmark detection and a deep learning model (LSTM) to classify different sign language gestures. The application can:

1. Collect training data for different sign language gestures
2. Train a machine learning model on the collected data
3. Perform real-time sign language detection using a webcam

## Project Structure

Sign-Language-Detection-Using-Machine-Learning/
├── data/                  # Directory for storing collected gesture data
├── models/                # Directory for storing trained models
├── logs/                  # TensorBoard logs for model training
├── src/                   # Source code
│   ├── __init__.py       # Python package initialization
│   ├── utils.py          # Utility functions for MediaPipe and landmark detection
│   ├── data_collection.py # Functions for collecting training data
│   ├── model.py          # Model definition and data loading
│   ├── train.py          # Model training functionality
│   └── predict.py        # Real-time prediction functionality
├── main.py               # Main application entry point
├── requirements.txt      # Project dependencies
├── LICENSE              # MIT License file
└── README.md            # Project documentation

## Setup Instructions

### Prerequisites

- Python 3.7 or higher
- Webcam
- Required Python packages (listed in requirements.txt)
- CUDA-compatible GPU (recommended for faster training)

### Installation

1. Clone the repository:
git clone https:/githubcom/girish8767/Sign-Language-Detection-Using-Machine-Learning.git 
cd Sign-Language-Detection-Using-Machine-Learning

2. Install the required dependencies:
pip install -r requirements.txt

3. Ensure your webcam is properly connected and has necessary permissions.

## Usage

Run the main script to start the application: python main.py


You will be presented with three options:

1. **Collect Data**: Record sign language gestures for training
   - The system will guide you through recording multiple sequences for each gesture
   - Follow the on-screen instructions to perform each gesture

2. **Train Model**: Train the model on your collected data
   - This will process the collected data and train an LSTM model
   - Training progress will be displayed in the console

3. **Real-time Prediction**: Test the trained model with your webcam
   - The system will display the detected gestures in real-time
   - Press 'q' to exit the prediction mode

## Customizing Gestures

By default, the system is set up to recognize three gestures: 'hello', 'thanks', and 'iloveyou'. To add or modify gestures:

1. Edit the `actions` array in `main.py` to include your desired gestures
2. Run the data collection process for the new gestures
3. Retrain the model with the updated dataset

## Model Architecture

The project uses a sequential LSTM (Long Short-Term Memory) model with the following architecture:

- Input layer: LSTM with 64 units
- Hidden layer 1: LSTM with 128 units
- Hidden layer 2: LSTM with 64 units
- Dense layer 1: 64 units with ReLU activation
- Dense layer 2: 32 units with ReLU activation
- Output layer: Dense layer with softmax activation (number of units equals number of gestures)

## Troubleshooting

### Camera Access Issues

If you encounter camera access issues:
- On macOS: Go to System Preferences > Security & Privacy > Privacy > Camera and ensure your terminal or Python has permission to access the camera
- Restart your terminal or application after granting permissions

### Model Performance

If the model is not performing well:
- Collect more training data for each gesture
- Ensure good lighting conditions during both training and prediction
- Try to perform gestures consistently during training

## Technical Implementation Details

### Data Collection
The system captures video frames from your webcam and uses MediaPipe to extract hand landmarks. These landmarks are saved as NumPy arrays for each frame of each sequence.

### Model Training
The collected landmark data is processed and fed into an LSTM model, which learns to recognize patterns in the sequence of hand positions that correspond to different gestures.

### Real-time Prediction
During prediction, the system continuously captures frames, extracts landmarks, and feeds sequences of these landmarks to the trained model to predict the gesture being performed.

## Future Improvements

Potential enhancements for this project:
- Support for more complex sign language gestures
- Integration with text-to-speech for audible output
- Mobile application deployment
- Transfer learning with pre-trained models for improved accuracy
- Real-time translation of continuous sign language sentences

## Contributors

- Your Name

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MediaPipe for the hand landmark detection framework
- TensorFlow for the machine learning framework
- The sign language community for inspiration

