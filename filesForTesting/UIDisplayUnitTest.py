# A unit test for the UI display

import unittest
import time
import sys
import os

# Get the parent directory of 'filesForTesting' and add it to sys.path -- ChatGPT
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from finalProduct.app import SignLanguageTranslatorApp as SignLanguageTranslatorUIUnitTest

class TestUIDisplay(unittest.TestCase):
    def setUp(self):
        """Initialize UI instance"""
        self.ui = SignLanguageTranslatorUIUnitTest("Test UI", "Press Q to quit")

    def test_prediction_display(self):
        """Test if the predicted label updates correctly"""
        self.ui.updateDisplay("B", [0.1, 0.9, 0.0, 0.0, 0.0], None)  # Dummy image
        self.assertEqual(self.ui.predicted_label, "B")

    def test_status_message_timeout(self):
        """Test if the status message disappears after 2 seconds"""
        self.ui.status_message = "Test Message"
        self.ui.status_timestamp = time.time()
        time.sleep(2.1)  # Wait for message timeout
        self.assertEqual(self.ui.status_message, "")

if __name__ == "__main__":
    unittest.main()