# A unit test for the queue data structure

import unittest
from queue import Queue

class TestGestureQueue(unittest.TestCase):
    def setUp(self):
        """Set up a fresh queue for each test"""
        self.gesture_queue = Queue()

    def test_enqueue_dequeue(self):
        """Test if elements are enqueued and dequeued in FIFO order"""
        self.gesture_queue.put("A")
        self.gesture_queue.put("B")
        self.gesture_queue.put("C")

        self.assertEqual(self.gesture_queue.get(), "A")
        self.assertEqual(self.gesture_queue.get(), "B")
        self.assertEqual(self.gesture_queue.get(), "C")

    def test_empty_queue(self):
        """Test behavior when dequeuing from an empty queue"""
        with self.assertRaises(Exception):  # Default Queue raises Exception when empty
            self.gesture_queue.get_nowait()

    def test_queue_clearing(self):
        """Test if the queue can be cleared correctly"""
        self.gesture_queue.put("A")
        self.gesture_queue.put("B")
        self.gesture_queue.queue.clear()
        self.assertEqual(len(self.gesture_queue.queue), 0)

if __name__ == "__main__":
    unittest.main()