import cv2
import joblib
import numpy as np
from river import tree

# Load the trained model
model = joblib.load('hoeffding_model.pkl')

# Define the label encoder (use the same one you used during training)
label_encoder = joblib.load('label_encoder.pkl')

# Simulate feature extraction from a frame (you can adjust this)
def extract_features_from_frame(frame):
    # For simplicity, let's convert the frame to grayscale and flatten it as a feature vector.
    # In a real scenario, this would be replaced with actual feature extraction logic.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))  # Resize to a fixed size
    features = resized.flatten()  # Flatten the 2D array to 1D
    return features

# Set up the video capture
cap = cv2.VideoCapture(0)  # 0 is the default camera

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Extract features from the frame
    features = extract_features_from_frame(frame)

    # Format the features for prediction (as a dictionary for the river model)
    x = {f'feat_{i}': features[i] for i in range(len(features))}

    # Make prediction
    y_pred = model.predict_one(x)

    # Decode the label
    if y_pred is not None:
        predicted_label = label_encoder.inverse_transform([y_pred])[0]
    else:
        predicted_label = "Unknown"

    # Display the resulting frame with the predicted activity
    cv2.putText(frame, f"Predicted: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Real-Time Human Activity Recognition', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture object and close all windows
cap.release()
cv2.destroyAllWindows()





