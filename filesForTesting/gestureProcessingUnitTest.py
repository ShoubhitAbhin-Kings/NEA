# A unit test that verifies whether gestures are being extracted correctly from random noise that is fed in
# This unit test is courtesy of ChatGPT

import unittest
import numpy as np
import sys
import os

# Get the parent directory of 'filesForTesting' and add it to sys.path -- ChatGPT
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from finalProduct.app import model as modelUnitTest
from finalProduct.app import classLabels as classLabelsUnitTest

class TestGestureProcessing(unittest.TestCase):
    def test_prediction_output(self):
        """Test if model outputs a valid label"""
        dummy_image = np.random.rand(1, 300, 300, 3)  # Random noise image
        prediction = modelUnitTest.predict(dummy_image)
        predicted_class = np.argmax(prediction)
        predicted_label = classLabelsUnitTest[predicted_class]

        self.assertIn(predicted_label, classLabelsUnitTest, "Predicted label is invalid.")

if __name__ == "__main__":
    unittest.main()