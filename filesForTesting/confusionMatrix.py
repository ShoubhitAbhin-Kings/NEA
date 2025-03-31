import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

modelToBeLoaded = '/Users/shoubhitabhin/Documents/VSCode Projects/JanMMLV3/CNNModels/theModel.h5'
data_dir = '/Users/shoubhitabhin/Documents/VSCode Projects/JanMMLV3/savedData/trainOnThese'

def generate_confusion_matrix(model, validation_datagen, data_dir, batch_size):
    """
    Generate and plot a confusion matrix for the model's predictions on the validation set.
    """
    # Prepare validation data generator
    val_generator = validation_datagen.flow_from_directory(
        os.path.join(data_dir, 'evaluation'),
        target_size=(300, 300),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False  # Ensure that the order is kept to match true labels and predictions
    )

    # Get true labels (y_true) and predictions (y_pred)
    y_true = val_generator.classes
    y_pred = model.predict(val_generator, steps=len(val_generator), verbose=1)
    y_pred = np.argmax(y_pred, axis=1)  # Convert predicted probabilities to class indices

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot the confusion matrix using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=val_generator.class_indices.keys(), yticklabels=val_generator.class_indices.keys())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def load_model_and_generate_confusion_matrix(model_path, data_dir, batch_size=32):
    """
    Load a trained model and generate a confusion matrix for it.
    """
    # Load the trained model in H5 format
    model = tf.keras.models.load_model(modelToBeLoaded)

    # Initialize the validation data generator
    validation_datagen = ImageDataGenerator(rescale=1./255)

    # Generate confusion matrix for the model
    generate_confusion_matrix(model, validation_datagen, data_dir, batch_size)

if __name__ == "__main__":
    # Specify the path to the trained model and data directory used
    model_path = modelToBeLoaded
    data_dir = data_dir
    
    load_model_and_generate_confusion_matrix(model_path, data_dir)