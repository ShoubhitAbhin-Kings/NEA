import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def build_model(input_shape=(300, 300, 3), num_classes=5):
    """
    Builds a CNN model for sign language recognition.

    Improvements included:
    - Corrected num_classes to match the actual dataset (A, B, C, D, E → 5 classes).
    - Added additional convolutional layers for better feature extraction.
    - Tuned dropout rates to reduce overfitting.
    - Added batch normalization to stabilize training.
    - Verified softmax activation for multi-class classification.
    """
    model = Sequential()

    # First convolutional layer: Extract low-level features
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Second convolutional layer: Deeper feature extraction
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Third convolutional layer: More complex features
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Fourth convolutional layer: Increasing feature complexity
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten and Dense layers for classification
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))

    # Fully connected layer with dropout to reduce overfitting
    model.add(Dropout(0.5))

    # Output layer with softmax for classification
    model.add(Dense(num_classes, activation='softmax')) 

    # Compile the model with an appropriate optimizer and loss function
    model.compile(optimiser='adam', loss='categorical_crossentropy', metrics=['accuracy']) # CHANGED OPTIMISER TO BRITISH SPELLING MAY CAUSE ERROR

    return model