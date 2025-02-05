import cv2
import numpy as np
from deepface import DeepFace

# Load OpenCV's pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect multiple faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        # Extract the face ROI
        face = frame[y:y + h, x:x + w]

        try:
            # Analyze the detected face for emotion
            result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)

            # Extract the dominant emotion
            dominant_emotion = result[0]['dominant_emotion']

            # Set bounding box color
            if dominant_emotion in ["sad", "angry"]:
                box_color = (0, 0, 255)  # Red for sad or angry
            else:
                box_color = (0, 255, 0)  # Green for other emotions

            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

            # Display detected emotion above the face
            cv2.putText(frame, f"{dominant_emotion}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

        except Exception as e:
            print("Error:", e)

    # Show the output frame
    cv2.imshow('Multi-Face Emotion Recognition', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()