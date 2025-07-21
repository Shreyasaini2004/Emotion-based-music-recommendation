import cv2
import mediapipe as mp
import numpy as np
import os
import time

def collect_emotion_data():
    # Get emotion name from user
    emotion_name = input("Enter emotion name (happy, sad, angry, neutral, surprised, etc.): ").lower()
    samples_to_collect = int(input("Enter number of samples to collect (recommended: 50-100): "))
    
    # Initialize MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print(f"\nCollecting data for emotion: {emotion_name}")
    print("=" * 50)
    print("CONTROLS:")
    print("  Press 'c' to CAPTURE a sample")
    print("  Press 'q' or 'ESC' to QUIT")
    print("  Make sure your face is clearly visible")
    print("=" * 50)
    print(f"Target: {samples_to_collect} samples\n")
    
    collected_data = []
    sample_count = 0
    
    while sample_count < samples_to_collect:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
            
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = face_mesh.process(rgb_frame)
        
        # Draw landmarks if face detected
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS
                )
        
        # Display info on frame
        cv2.putText(frame, f"Emotion: {emotion_name.upper()}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Samples: {sample_count}/{samples_to_collect}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'C' to capture, 'Q' or ESC to quit", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add a colored border to indicate status
        if results.multi_face_landmarks:
            cv2.rectangle(frame, (5, 5), (frame.shape[1]-5, frame.shape[0]-5), (0, 255, 0), 3)  # Green border
        else:
            cv2.rectangle(frame, (5, 5), (frame.shape[1]-5, frame.shape[0]-5), (0, 0, 255), 3)  # Red border
        
        cv2.imshow('Emotion Data Collection', frame)
        
        # Wait for key press with longer delay
        key = cv2.waitKey(30) & 0xFF
        
        if key == ord('c'):  # Capture sample
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                face_data = []
                
                # Extract normalized landmarks (x, y, z coordinates)
                for landmark in landmarks.landmark:
                    face_data.extend([landmark.x, landmark.y, landmark.z])
                
                collected_data.append(face_data)
                sample_count += 1
                print(f"Sample {sample_count} captured!")
                
                # Brief pause after capture
                time.sleep(0.5)
            else:
                print("No face detected! Please make sure your face is visible.")
                
        elif key == ord('q') or key == 27:  # Quit (q key or ESC key)
            print("Quitting data collection...")
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    
    # Save collected data
    if collected_data:
        data_array = np.array(collected_data)
        filename = f"{emotion_name}_data.npy"
        np.save(filename, data_array)
        print(f"\nData saved to {filename}")
        print(f"Shape: {data_array.shape}")
        print(f"Total samples collected: {len(collected_data)}")
    else:
        print("No data collected!")

if __name__ == "__main__":
    collect_emotion_data()