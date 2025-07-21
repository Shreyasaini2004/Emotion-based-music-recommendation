import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pickle
import time
# Removed webbrowser import as we will use Streamlit's markdown for links
# import webbrowser
# Removed dependency on music_database.py as per user request
# from music_database import get_music_recommendation, get_youtube_search_url 
import threading # Not directly used in the Streamlit UI for this logic, but kept if other parts rely on it

# Page config
st.set_page_config(page_title="🎵 Emotion Music Recommender", page_icon="🎵", layout="wide")

# Custom CSS for better visibility and styling
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #ff6b6b;
        font-size: 3rem;
        margin-bottom: 1rem;
        font-family: 'Inter', sans-serif;
    }
    .emotion-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff6b6b;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        font-family: 'Inter', sans-serif;
        color: #333; /* Explicitly set text color to dark grey for readability */
    }
    .emotion-box h3 { /* Target h3 inside emotion-box */
        color: #333; /* Ensure heading is dark */
    }
    .song-recommendation {
        background-color: #e8f4fd;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4ecdc4;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        font-family: 'Inter', sans-serif;
        color: #333; /* Explicitly set text color to dark grey for readability */
    }
    .song-recommendation h4 { /* Target h4 inside song-recommendation */
        color: #333; /* Ensure heading is dark */
    }
    .prediction-count {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        font-family: 'Inter', sans-serif;
        color: #333; /* Explicitly set text color to dark grey for readability */
    }
    /* New CSS for the Confidence Box */
    .confidence-box {
        background-color: #e6f7ff; /* A light blue to distinguish, or match others */
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #007bff; /* A blue border */
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        font-family: 'Inter', sans-serif;
        color: #333; /* Ensure text is dark */
        text-align: center; /* Center the confidence value */
    }
    .confidence-box h4 { /* Style heading inside confidence box */
        color: #333;
        margin-bottom: 5px; /* Small margin below heading */
    }
    .confidence-value { /* Style the actual percentage value */
        font-size: 2.5rem; /* Make it larger */
        font-weight: bold;
        color: #007bff; /* Blue color for the value */
    }

    .stButton>button {
        background-color: #4CAF50; /* Green */
        color: white;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
        border: none;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_emotion_model():
    """
    Loads the pre-trained emotion detection model and label encoder.
    Uses st.cache_resource to avoid reloading on every rerun.
    """
    try:
        # Ensure 'emotion_model.h5' and 'label_encoder.pkl' are in the same directory
        # as your Streamlit app or provide full paths.
        model = load_model('emotion_model.h5')
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        return model, label_encoder
    except Exception as e:
        st.error(f"❌ Error loading model or label encoder. Please ensure 'emotion_model.h5' and 'label_encoder.pkl' are in the correct directory. Error: {str(e)}")
        return None, None

def initialize_mediapipe():
    """
    Initializes MediaPipe Face Mesh for landmark detection.
    """
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False, # Process video streams
        max_num_faces=1,         # Detects only one face
        refine_landmarks=True,   # Provides more detailed landmarks
        min_detection_confidence=0.5 # Minimum confidence for detection
    )
    return face_mesh, mp_face_mesh

def extract_face_landmarks(image, face_mesh):
    """
    Extracts 3D face landmarks from an image using MediaPipe Face Mesh.
    Args:
        image (np.array): BGR image frame from webcam.
        face_mesh (mediapipe.solutions.face_mesh.FaceMesh): Initialized FaceMesh object.
    Returns:
        np.array: Flattened array of landmark coordinates (x, y, z) or None if no face detected.
    """
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        face_data = []
        for lm in landmarks.landmark:
            # Extract x, y, z coordinates. z represents depth.
            face_data.extend([lm.x, lm.y, lm.z])
        return np.array(face_data).reshape(1, -1) # Reshape for model input
    return None

def predict_emotion(model, label_encoder, face_data):
    if face_data is not None:
        prediction = model.predict(face_data, verbose=0)[0]
        emotion_idx = np.argmax(prediction)
        emotion = label_encoder.inverse_transform([emotion_idx])[0]
        confidence = prediction[emotion_idx]

        # --- Add this for debugging (prints to terminal where Streamlit is run) ---
        all_emotions = label_encoder.inverse_transform(np.arange(len(prediction)))
        
        print("\n--- All Emotion Probabilities ---")
        for i, prob in enumerate(prediction):
            print(f"{all_emotions[i].title()}: {prob:.4f}")
        print("-------------------------------\n")
        # -----------------------------

        return emotion, confidence, prediction # Also return raw prediction for Streamlit display
    return None, 0, None

def get_youtube_search_url_dynamic(emotion, language_preference, preferred_artist):
    """
    Generate YouTube search URL based on emotion, language, and preferred artist.
    """
    search_query_parts = []
    if emotion:
        search_query_parts.append(f"{emotion} songs")
    if language_preference and language_preference != "Any":
        search_query_parts.append(language_preference)
    if preferred_artist:
        search_query_parts.append(preferred_artist)
    
    # Combine parts into a single query string
    search_query = " ".join(search_query_parts)
    
    # Replace spaces with + for URL encoding
    search_query = search_query.replace(" ", "+")
    
    youtube_url = f"https://www.youtube.com/results?search_query={search_query}"
    return youtube_url, search_query # Return both URL and the constructed query

def main():
    st.markdown('<h1 class="main-title">🎵 Emotion-Based Music Recommender</h1>', unsafe_allow_html=True)
    st.markdown("### Detect your emotion via webcam and get music tailored to your mood 🎧")

    st.write("✅ App Loaded")

    # Load the emotion model and label encoder
    model, label_encoder = load_emotion_model()
    if model is None or label_encoder is None:
        # If model loading failed, stop execution
        return

    # Initialize MediaPipe Face Mesh
    face_mesh, mp_face_mesh = initialize_mediapipe()

    st.sidebar.title("🎛️ Controls")
    # Language preference for music recommendations
    language_preference = st.sidebar.selectbox("Language Preference:", ["Any", "English", "Hindi"])
    # New: Input for preferred artist
    preferred_artist = st.sidebar.text_input("Preferred Artist (Optional):", help="Enter an artist name to include in the YouTube search.")

    # Slider to set the maximum number of predictions
    max_predictions = st.sidebar.slider("Max Predictions:", min_value=1, max_value=20, value=10, help="The maximum number of times an emotion will be detected and a search query generated before stopping the webcam.")

    # Initialize session state variables if they don't exist
    if 'detection_active' not in st.session_state:
        st.session_state.detection_active = False
    if 'prediction_count' not in st.session_state:
        st.session_state.prediction_count = 0
    # 'recommendations' list will now store search queries and URLs
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = []
    # 'last_emotion' is now primarily for internal logging/debug, not gating new recommendations
    if 'last_emotion' not in st.session_state: 
        st.session_state.last_emotion = None

    col1, col2 = st.columns([2, 1]) # Create two columns for layout

    with col1:
        camera_placeholder = st.empty() # Placeholder for the webcam feed
        start_button = st.button("🎥 Start Emotion Detection")
        stop_button = st.button("⏹️ Stop Detection")

        if start_button:
            st.session_state.detection_active = True
            st.session_state.prediction_count = 0
            st.session_state.recommendations = []
            st.session_state.last_emotion = None # Reset last emotion on start
            st.write("🔍 Starting detection...")
            # Streamlit reruns the script from top when state changes,
            # so the webcam loop will start in the next rerun.

        if stop_button:
            st.session_state.detection_active = False
            st.write("⛔ Detection stopped.")

    with col2:
        # Placeholders for dynamic content in the right column
        emotion_placeholder = st.empty()
        confidence_placeholder = st.empty()
        count_placeholder = st.empty()
        recommendation_display_placeholder = st.empty() # For displaying current recommendation

        # Add a placeholder for live probabilities in the sidebar for debugging
        live_probabilities_placeholder = st.sidebar.empty()


    # This block runs only if detection is active
    if st.session_state.detection_active:
        cap = cv2.VideoCapture(0) # Open the default webcam
        if not cap.isOpened():
            st.error("❌ Could not open webcam. Please ensure it's not in use by another application and grant permissions.")
            st.session_state.detection_active = False # Stop detection if webcam fails
            return

        # Loop for continuous emotion detection
        while st.session_state.detection_active and st.session_state.prediction_count < max_predictions:
            ret, frame = cap.read() # Read a frame from the webcam
            if not ret:
                st.error("❌ Failed to read from webcam. Exiting detection.")
                st.session_state.detection_active = False
                break

            frame = cv2.flip(frame, 1) # Flip frame horizontally for mirror effect
            camera_placeholder.image(frame, channels="BGR", width=400, caption="Live Webcam Feed")

            face_data = extract_face_landmarks(frame, face_mesh)

            if face_data is not None:
                emotion, confidence, raw_prediction = predict_emotion(model, label_encoder, face_data) # Get raw prediction

                # --- Display Live Probabilities in Sidebar ---
                if raw_prediction is not None:
                    all_emotions = label_encoder.inverse_transform(np.arange(len(raw_prediction)))
                    prob_df = {
                        "Emotion": [e.title() for e in all_emotions],
                        "Probability": raw_prediction.round(4)
                    }
                    live_probabilities_placeholder.subheader("Live Emotion Probabilities")
                    live_probabilities_placeholder.dataframe(prob_df, use_container_width=True)
                # --------------------------------------------

                # Only process if a confident emotion is detected
                if emotion and confidence > 0.5:
                    # Update the displayed emotion and confidence
                    emotion_placeholder.markdown(
                        f'<div class="emotion-box"><h3>Detected Emotion: <span style="color:#20B2AA;">{emotion.title()}</span></h3></div>',
                        unsafe_allow_html=True
                    )
                    
                    # Confidence box now uses custom CSS class
                    confidence_placeholder.markdown(
                        f"""
                        <div class="confidence-box">
                            <h4>Confidence:</h4>
                            <div class="confidence-value">{confidence:.2%}</div>
                        </div>
                        """, unsafe_allow_html=True
                    )

                    # Increment prediction count for ANY valid detection, up to max_predictions
                    st.session_state.prediction_count += 1
                    count_placeholder.markdown(
                        f'<div class="prediction-count">Searches: <strong>{st.session_state.prediction_count}</strong> / {max_predictions}</div>',
                        unsafe_allow_html=True
                    )

                    # Generate YouTube search URL dynamically
                    youtube_url, search_query_text = get_youtube_search_url_dynamic(
                        emotion, language_preference, preferred_artist
                    )

                    search_data = {
                        'emotion': emotion,
                        'language': language_preference,
                        'artist': preferred_artist,
                        'search_query': search_query_text,
                        'url': youtube_url,
                        'count': st.session_state.prediction_count
                    }
                    st.session_state.recommendations.append(search_data) 

                    # Update last_emotion for internal tracking
                    st.session_state.last_emotion = emotion 

                    # Display the current search query and clickable link
                    with recommendation_display_placeholder.container():
                        st.markdown(f"""
                        <div class="song-recommendation">
                        <h4>🔍 Search for {search_data['emotion'].title()}</h4>
                        <p><strong>Query:</strong> {search_data['search_query']}</p>
                        <p><a href="{search_data['url']}" target="_blank" style="color:#007bff; text-decoration:none;">▶️ Go to YouTube Search</a></p>
                        </div>
                        """, unsafe_allow_html=True)

            # Slight delay to control prediction rate and allow UI to update
            time.sleep(0.5)

        # Release the webcam when detection stops or max predictions reached
        cap.release()
        st.session_state.detection_active = False # Ensure detection is marked as stopped
        st.success("✅ Detection complete! Max searches reached or stopped manually.")


if __name__ == '__main__':
    main()