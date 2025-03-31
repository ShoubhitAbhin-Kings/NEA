# New GestureQueue class
class gestureQueue:
    def __init__(self, max_size: int):
        self.queue = deque(maxlen=max_size)  # Use deque for efficient FIFO operations
        self.max_size = max_size

    def enqueue(self, letter: str):
        """Add a letter to the queue"""
        self.queue.append(letter)

    def dequeue(self):
        """Remove a letter from the queue"""
        if self.queue:
            return self.queue.popleft()
        return None

    def clear(self):
        """Clear the entire queue"""
        self.queue.clear()

    def getQueue(self):
        """Return the list of letters in the queue"""
        return list(self.queue)