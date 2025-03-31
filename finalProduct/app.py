import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
from PIL import Image
from collections import deque  # For the queue functionality
import datetime  # For timestamping the file name
import time  # For managing the delay
from extraFiles.basicQueue import gestureQueue

modelToBeLoaded = 'CNNModels/sign_language_model.keras'

# Try loading the model and handle any exceptions
try:
    model = load_model(modelToBeLoaded)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)  # Exit if the model cannot be loaded

# Define class labels (ensure these match the training class order, so when more letters are added make sure they match)
classLabels = ['A', 'B', 'C', 'D', 'E']

# Define SignLanguageTranslatorUI class
class signLanguageTranslatorApp:
    def __init__(self, window_title: str, quit_message: str):
        self.window_title = window_title
        self.quit_message = quit_message
        self.classLabels = classLabels
        self.predicted_label = ""
        self.predictions = []
        self.gesture_queue = gestureQueue(max_size=10)  # Initialize gesture queue with a max size of 10
        self.last_predicted_label = None  # Track the last predicted letter added to the queue
        self.stable_frame_count = 0  # Counter to track how many frames the prediction has been stable
        self.stable_frame_threshold = 10  # Number of frames the prediction must remain stable before being enqueued
        self.status_message = ""  # For displaying status messages on the screen
        self.status_timestamp = None  # Timestamp when the status message is shown

        # Initialize the OpenCV window
        cv2.namedWindow(self.window_title)

    def showTitle(self, img):
        """Show the title text at the top of the window"""
        cv2.putText(img, self.window_title, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    def showQuitMessage(self, img):
        """Show the instructions to quit the program"""
        cv2.putText(img, self.quit_message, (50, img.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    def showPredictions(self, img):
        """Show the predicted label and probabilities on the image"""
        y_offset = 100  # Starting point for displaying predictions
        for i, (label, prob) in enumerate(zip(self.classLabels, self.predictions)):
            cv2.putText(img, f"{label}: {prob*100:.2f}%", (50, y_offset + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    def showGestureQueue(self, img):
        """Display the gesture queue on the screen"""
        y_offset = img.shape[0] - 100  # Starting point for displaying the queue
        queue_str = "Queue: " + "".join(self.gesture_queue.getQueue())  # Get all letters in the queue
        cv2.putText(img, queue_str, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    def showStatusMessage(self, img):
        """Display a status message on the screen for 2 seconds"""
        yOffset = img.shape[0] - 150  # Starting point for displaying the message
        if self.status_message and self.status_timestamp:
            elapsed_time = time.time() - self.status_timestamp  # Calculate elapsed time
            if elapsed_time < 2:
                cv2.putText(img, self.status_message, (50, yOffset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            else:
                self.status_message = ""  # Clear message after 2 seconds
                self.status_timestamp = None  # Reset timestamp

    def updateDisplay(self, predicted_label, predictions, img):
        """Update the displayed information"""
        self.predicted_label = predicted_label
        self.predictions = predictions
        self.showTitle(img)
        self.showPredictions(img)
        self.showQuitMessage(img)
        self.showGestureQueue(img)  # Display the gesture queue
        self.showStatusMessage(img)  # Display status message

    def processPredictedLetter(self, predicted_label):
        """Add predicted letter to the queue if stable for the defined number of frames"""
        if predicted_label != self.last_predicted_label:
            self.stable_frame_count = 0  # Reset counter if the predicted label changes
            self.last_predicted_label = predicted_label
        else:
            self.stable_frame_count += 1

        if self.stable_frame_count >= self.stable_frame_threshold:
            self.gesture_queue.enqueue(predicted_label)
            self.stable_frame_count = 0  # Reset after enqueuing

    def saveQueueToFile(self):
        """Save the current gesture queue to a text file with a timestamp"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sequenceOfLetters{timestamp}.txt"
        with open(filename, 'w') as f:
            # Join the queue elements into a single string and write it to the file
            f.write("".join(self.gesture_queue.getQueue()))
        print(f"Queue saved to {filename}")
        self.status_message = f"Queue saved as {filename}"  # Update the status message
        self.status_timestamp = time.time()  # Set the timestamp for the message

# Initialize webcam and hand detector
try:
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)
    if not cap.isOpened():
        raise Exception("Webcam not accessible. Please check your connection.")
except Exception as e:
    print(f"Error initializing webcam: {e}")
    exit(1)  # Exit if the webcam cannot be accessed

# Initialize the UI class
ui = signLanguageTranslatorApp("Sign Language Translator", "Press Q to Quit | Press C to Clear Queue | Press S to Save Queue")

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
        predicted_label = classLabels[predicted_class]
        probabilities = prediction[0]  # Get probabilities for all classes

        # Update the UI with predictions and process the predicted letter
        ui.processPredictedLetter(predicted_label)  # Check if letter should be enqueued
        ui.updateDisplay(predicted_label, probabilities, img)

    # Show the image with updated UI
    cv2.imshow("Sign Language Recognition", img)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Clear the queue on 'c' key press
    if cv2.waitKey(1) & 0xFF == ord('c'):
        ui.gesture_queue.clear()
        ui.status_message = "Queue cleared"  # Display message on screen
        ui.status_timestamp = time.time()  # Set timestamp for the message
    
    # Save the queue to a text file on 's' key press
    if cv2.waitKey(1) & 0xFF == ord('s'):
        ui.saveQueueToFile()
        ui.status_message = "File saved"  # Display message on screen
        ui.status_timestamp = time.time()

cap.release()
cv2.destroyAllWindows()