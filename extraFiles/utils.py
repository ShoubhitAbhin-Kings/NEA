import cv2
import numpy as np

def preprocess_frame(frame):
    # Resize the frame to 300x300
    frame = cv2.resize(frame, (300, 300))
    # Normalize the frame (scale pixel values between 0 and 1)
    frame = frame.astype('float32') / 255.0
    return frame