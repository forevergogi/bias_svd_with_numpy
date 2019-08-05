
import Queue
import threading
import time

exitFlag = 0

class ThreadQueue(threading.Thread):
    def __init__(self,threadID,name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name

    def run(self):
        print("Starting" + self.name)
        print("Exiting" + self.name)

