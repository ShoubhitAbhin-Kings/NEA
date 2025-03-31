import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
from PIL import Image

# Load the trained model
model = load_model('sign_language_model.keras')

# Define class labels (ensure these match the training class order)
class_labels = ['A', 'B', 'C', 'D', 'E']

# Initialize webcam
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

while True:
    success, img = cap.read()
    if not success:
        break

    hands, img = detector.findHands(img, draw=True)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Extract and preprocess the hand region
        hand_img = img[y:y+h, x:x+w]
        hand_img = cv2.resize(hand_img, (300, 300))
        hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        hand_img = hand_img / 255.0  # Normalize
        hand_img = np.expand_dims(hand_img, axis=0)

        # Predict the sign language letter
        prediction = model.predict(hand_img)
        predicted_class = np.argmax(prediction)
        predicted_label = class_labels[predicted_class]

        # Display the result
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, predicted_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Sign Language Recognition", img)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()