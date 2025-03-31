# While I implemented a linked list structure to handle the gesture queue for educational purposes and to demonstrate my understanding of data structures, I ultimately chose to use Python’s inbuilt deque for performance reasons. The deque provides O(1) time complexity for both enqueue and dequeue operations, and it is optimized for use cases like mine. This choice avoids unnecessary overhead, ensuring that my program runs efficiently without introducing complexity for maintaining linked list pointers 


class Node:
    def __init__(self, gesture):
        self.gesture = gesture
        self.next = None

class GestureQueue:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0

    def enqueue(self, gesture):
        new_node = Node(gesture)
        if not self.tail:
            self.head = self.tail = new_node
        else:
            self.tail.next = new_node
            self.tail = new_node
        self.size += 1

    def dequeue(self):
        if self.head is None:
            return None
        gesture = self.head.gesture
        self.head = self.head.next
        if not self.head:
            self.tail = None
        self.size -= 1
        return gesture

    def clear(self):
        self.head = self.tail = None
        self.size = 0

    def getQueue(self):
        current = self.head
        gestures = []
        while current:
            gestures.append(current.gesture)
            current = current.next
        return gestures

    def isEmpty(self):
        return self.size == 0

# Usage Example
queue = GestureQueue()
queue.enqueue('A')
queue.enqueue('B')
queue.enqueue('C')
print(queue.getQueue())  # ['A', 'B', 'C']
queue.dequeue()
print(queue.getQueue())  # ['B', 'C']