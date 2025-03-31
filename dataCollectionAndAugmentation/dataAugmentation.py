# dataAugmentation.py is a Python script that reads images from a folder, applies data augmentation techniques, and saves the augmented images to a new folder.

import os
import cv2
import numpy as np
import time
import math

# Define the path to your data
data_dir = '/Users/shoubhitabhin/Documents/VSCode Projects/JanMMLV3/savedData/notAugmented'

# Ensure augmented data folder exists
augmented_data_dir = os.path.join(data_dir, 'augmented')
if not os.path.exists(augmented_data_dir):
    os.makedirs(augmented_data_dir)

# Loop through each letter folder in the saved data
for letter_folder in os.listdir(data_dir):
    letter_path = os.path.join(data_dir, letter_folder)

    # Skip files like .DS_Store (ERROR ENCOUNTERED WHEN TESTING)
    if not os.path.isdir(letter_path):
        continue

    print(f"Processing images for letter: {letter_folder}")
    
    for img_name in os.listdir(letter_path):
        img_path = os.path.join(letter_path, img_name)

        # Skip non-image files
        if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        # Read the image
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        augmented_images = []

        # Example of flipping the image horizontally (only flip once)
        img_flip = cv2.flip(img, 1)
        augmented_images.append(img_flip)

        # Example of rotating the image (rotate between -15° and 15°)
        rows, cols = img.shape[:2]
        angle = 15  # You can make this a variable if you want more randomness
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        img_rotate = cv2.warpAffine(img, M, (cols, rows))
        augmented_images.append(img_rotate)

        """
        Possible reason for failure of B and D since they are rectangualr in shape, solution from ChatGPT below
        # Resize to a new size (e.g., 64x64)
        img_resize = cv2.resize(img, (64, 64))
        augmented_images.append(img_resize) 
        """

        #  --- The code below is from ChatGPT --- 

        # Resize while maintaining aspect ratio and padding to 300x300
        h, w = img.shape[:2]
        scale = 300 / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized_img = cv2.resize(img, (new_w, new_h))

        # Create a blank black image (300x300) and center the resized image
        padded_img = np.zeros((300, 300, 3), dtype=np.uint8)
        x_offset = (300 - new_w) // 2
        y_offset = (300 - new_h) // 2
        padded_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_img

        augmented_images.append(padded_img)

        # --- End of ChatGPT code ---

        # Save augmented images
        for idx, augmented_img in enumerate(augmented_images):
            augmented_img_name = f"{letter_folder}_{img_name.split('.')[0]}_aug_{idx+1}.jpg"
            augmented_img_path = os.path.join(augmented_data_dir, letter_folder, augmented_img_name)
            if not os.path.exists(os.path.join(augmented_data_dir, letter_folder)):
                os.makedirs(os.path.join(augmented_data_dir, letter_folder))
            
            cv2.imwrite(augmented_img_path, augmented_img)

        print(f"Augmented {img_name} and saved {len(augmented_images)} new images.")

print("Data augmentation completed!")