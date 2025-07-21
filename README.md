🎵 Emotion-Based Music Recommender
This Streamlit application uses real-time webcam feed to detect a user's emotion and then generates a dynamic YouTube search query based on the detected emotion, a user-selected language, and an optional preferred artist. The goal is to provide a quick way for users to find music tailored to their mood.

✨ Features
Real-time Emotion Detection: Utilizes MediaPipe Face Mesh and a pre-trained Keras model to detect emotions (e.g., Happy, Sad, Neutral).

Dynamic YouTube Search: Generates a YouTube search URL based on:

Detected Emotion (e.g., "happy songs")

Selected Language (e.g., "English", "Hindi", or "Any")

Optional Preferred Artist (e.g., "Arijit Singh")

Interactive UI: Built with Streamlit for an intuitive user experience.

Customizable Predictions: Set a maximum number of searches to perform.

Live Probability Display: (For debugging/development) Shows the model's confidence for each emotion in real-time.

Clean & Responsive Design: Custom CSS for improved readability and aesthetics.

🚀 Getting Started
Follow these steps to set up and run the project on your local machine.

Prerequisites
1. Before you begin, ensure you have the following installed:
Python 3.8+ (Recommended)
pip (Python package installer, usually comes with Python)

2. Webcam

3. Installation
Clone the Repository (or download files):
If you have a Git repository, clone it:
git clone <your-repository-url>
cd <your-repository-directory>

Otherwise, ensure all project files (.py scripts, .h5 model, .pkl encoder) are in the same directory.

4. Create requirements.txt:
Create a file named requirements.txt in your project's root directory and add the following lines to it:

    streamlit==1.28.1

    streamlit-webrtc==0.47.1

    opencv-python==4.8.1.78

    mediapipe==0.10.7

    tensorflow==2.13.0

    numpy==1.24.3

    Pillow==10.0.1

    av==10.0.0

    scikit-learn==1.3.0

5. Create run_project.bat (Windows Only):
Create a file named run_project.bat in your project's root directory and add the following content:

@echo off

echo 🎵 Emotion-Based Music Recommender 🎵

echo =====================================

echo.

:: Check if virtual environment exists

if not exist "venv" (

    echo Creating virtual environment...
    
    python -m venv venv
    
)

:: Activate virtual environment

echo Activating virtual environment...

call venv\Scripts\activate.bat

:: Install dependencies

echo Installing dependencies...

pip install -r requirements.txt

echo.

echo ✅ Setup complete!

echo.

echo Choose an option:

echo 1. Collect emotion data

echo 2. Train model

echo 3. Run main application

echo.

set /p choice=Enter your choice (1-3): 

if "%choice%"=="1" (

    echo Starting data collection...
    
    python data_collection.py
    
) else if "%choice%"=="2" (

    echo Starting model training...
    
    python train_model.py
    
) else if "%choice%"=="3" (

    echo Starting main application...
    
    streamlit run main_app.py
    
) else (

    echo Invalid choice. Please run the script again.
    
)

pause

🧠 Train the Emotion Detection Model
Before running the main application, you need to train the emotion detection model. This script will load your collected data (e.g., happy_data.npy, sad_data.npy), train a neural network, and save the model files (emotion_model.h5 and label_encoder.pkl).

Ensure Data Collection: Make sure you have collected emotion data using your data_collection.py script (which is not provided here but is implied by the presence of .npy files). You should have files like happy_data.npy, neutral_data.npy, sad_data.npy, etc., in the same directory.

Important Note on Bias: If your model consistently predicts one emotion (e.g., "Neutral" or "Sad"), it's highly likely due to an imbalance in your collected data. Ensure you have a roughly equal number of samples for each emotion, and that the expressions are clear and distinct during data collection.

Run the Training Script:

If using run_project.bat, select option 2.

Otherwise, manually run:

python train_model.py

This script will output information about the data loaded, model architecture, and training progress. Upon successful completion, it will save emotion_model.h5 and label_encoder.pkl in your project directory.

🚀 Running the Application
Once the model is trained and its files are present, you can run the Streamlit application.

Option 1: Using the Batch File (Windows Recommended)
Simply double-click run_project.bat.

A command prompt will open, guide you through virtual environment setup and dependency installation (if not already done), and then present a menu.

Enter 3 to "Run main application".

Option 2: Using the Command Line (Manual)
Create and activate your virtual environment (if not already done by the batch file):

python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

Install dependencies (if not already done):

pip install -r requirements.txt

Run the Streamlit app:

streamlit run main_app.py

This will open the application in your default web browser.

💡 Usage
Start Emotion Detection: Click the "🎥 Start Emotion Detection" button to activate your webcam and begin emotion analysis.

Controls: Use the sidebar to:

Select your Language Preference for the YouTube search.

Enter a Preferred Artist (optional) to include in the search query.

Adjust Max Predictions to control how many searches are generated before the webcam stops.

Observe Results:

The "Detected Emotion" box will show your current mood.

"Confidence" will display the model's certainty.

"Searches" will track the number of queries generated.

A "Search for [Emotion]" box will appear with the generated YouTube search query and a clickable link to "Go to YouTube Search". Clicking this link will open a new browser tab with YouTube search results based on your inputs.

Stop Detection: Click the "⏹️ Stop Detection" button to turn off your webcam and halt the process.

⚠️ Troubleshooting
"Error loading model...": Ensure emotion_model.h5 and label_encoder.pkl are in the same directory as your Streamlit script and that they were successfully generated by train_model.py.

"Could not open webcam.": Check if your webcam is connected, not in use by another application, and that your browser/OS has granted permission for Streamlit to access it.

Model always predicts one emotion (e.g., "Neutral" or "Sad"): This is typically due to data imbalance in your training dataset. Review the "Samples per emotion" output from train_model.py. You will need to collect more diverse and balanced data for all emotion classes and retrain the model.
