from timeit import default_timer as timer

class Timer(object):
    def __init__(self, label="default"):
        self.label = label

    def __enter__(self):
        self.start = timer()

    def __exit__(self, type, value, tb):
        end = timer()
        print(self.label + ": " + str(end - self.start))
