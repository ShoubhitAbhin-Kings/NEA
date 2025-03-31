# SAVE KERAS MODEL AS A H5 MODEL FOR THE CONFUSION MATRIX

import tensorflow as tf

# Load your existing model (assuming it's in the .keras format)
model = tf.keras.models.load_model('/Users/shoubhitabhin/Documents/VSCode Projects/JanMMLV3/CNNModels/theModel.keras')

# Save it in .h5 format
model.save('theModel.h5')

print("Model has been successfully saved in H5 format.")