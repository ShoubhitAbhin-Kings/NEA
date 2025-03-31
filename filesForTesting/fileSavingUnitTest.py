# A unit test for the file saving mechanism

import unittest
import os
import datetime
import sys
import os

# Get the parent directory of 'filesForTesting' and add it to sys.path -- ChatGPT
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from finalProduct.app import SignLanguageTranslatorApp as SignLanguageTranslatorUIUnitTest  

class TestFileSaving(unittest.TestCase):
    def setUp(self):
        """Set up a UI instance with a queue"""
        self.ui = SignLanguageTranslatorUIUnitTest("Test UI", "Press Q to quit")
        self.ui.gesture_queue = ["A", "B", "C"]  # Simulating a queue

    def test_save_queue_to_file(self):
        """Check if the queue is saved properly to a text file"""
        self.ui.saveQueueToFile()
        
        # Find the most recently saved file
        timestamp = datetime.datetime.now().strftime("%Y%m%d")
        files = [f for f in os.listdir() if f.startswith(f"gesture_queue_{timestamp}")]
        self.assertGreater(len(files), 0, "No file was created.")

        # Check file contents
        latest_file = max(files, key=os.path.getctime)
        with open(latest_file, 'r') as f:
            content = f.read()
        self.assertEqual(content, "ABC")

        # Clean up
        os.remove(latest_file)

if __name__ == "__main__":
    unittest.main()