import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# Initialise webcam and hand detector
def initialiseCamera():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)
    return cap, detector

# Capture and process hand image
def captureHandImage(cap, detector, offset, imgSize):
    success, img = cap.read()
    if not success:
        return None, None
    
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]
        imgCrop = img[max(0, y - offset): min(img.shape[0], y + h + offset + 5),
                      max(0, x - offset): min(img.shape[1], x + w + offset + 5)]
        return imgCrop, img
    return None, img

# Resize the image to fit the required size
def resizeImage(imgCrop, imgSize, h, w):
    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
    aspectRatio = h / w
    if aspectRatio > 1:
        k = imgSize / h
        wCal = math.ceil(k * w)
        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
        wGap = math.ceil((imgSize - wCal) / 2)
        imgWhite[:, wGap:wGap + wCal] = imgResize
    else:
        k = imgSize / w
        hCal = math.ceil(k * h)
        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
        hGap = math.ceil((imgSize - hCal) / 2)
        imgWhite[hGap:hGap + hCal, :] = imgResize
    return imgWhite

# Save the image
def saveImage(imgWhite, folder, counter):
    try:
        cv2.imwrite(f"{folder}/Image_{time.time()}.jpg", imgWhite)
        counter += 1
        print(f"Image number {counter} saved")
    except Exception as e:
        print(f"Error saving image: {e}")

# Main function
def main():
    cap, detector = initialiseCamera()
    offset = 30
    imgSize = 300
    counter = 0
    folder = 'savedData/notAugmented/E'
    
    while True:
        imgCrop, img = captureHandImage(cap, detector, offset, imgSize)
        if imgCrop is not None:
            imgWhite = resizeImage(imgCrop, imgSize, *imgCrop.shape[:2])
            cv2.imshow("Image White", imgWhite)
        
        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == ord("g"):
            saveImage(imgWhite, folder, counter)
            counter += 1
        if key == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()