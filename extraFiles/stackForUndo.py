# used for undoing in the gesture queue

class undoStack:
    def __init__(self, max_size=10):
        self.stack = []
        self.max_size = max_size

    def push(self, prediction):
        if len(self.stack) >= self.max_size:
            self.stack.pop(0)  # Remove the oldest item
        self.stack.append(prediction)

    def pop(self):
        if self.stack:
            return self.stack.pop()
        return None

    def get_stack(self):
        return self.stack