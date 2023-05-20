from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("Model\\Number.h5", compile=False)

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Initialize the video capture
cap = cv2.VideoCapture(0)

@app.route('/')
def index():
    return render_template('index.html')

def gen():
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        if frame is not None:
            # Preprocess the frame and extract hand landmarks
            landmarks = preprocess_image(frame)

            if landmarks is not None:
                # Make a prediction using the trained model
                prediction = model.predict(landmarks)
                class_index = np.argmax(prediction)
                class_label = ['Class 0', 'Class 1', 'Class 2',
                               'Class 3', 'Class 4', 'Class 5',
                               'Class 6', 'Class 7', 'Class 8',
                               'Class 9'][class_index]

                # Draw the predicted class label on the frame
                cv2.putText(frame, class_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Convert the frame to JPEG format
            ret, jpeg = cv2.imencode('.jpg', frame)

            # Yield the frame as a byte string
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

def preprocess_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    hand_landmarks = results.multi_hand_landmarks

    if hand_landmarks:
        landmark_list = []
        for landmark in hand_landmarks[0].landmark:
            landmark_list.extend([landmark.x, landmark.y, landmark.z])
        return np.array([landmark_list])
    else:
        return None

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    
    app.run(host="0.0.0.0",port=80,debug=True)


# Release the video capture and close all windows when the app is stopped
