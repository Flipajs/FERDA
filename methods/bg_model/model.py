__author__ = 'fnaiser'

import threading


class Model(threading.Thread):
    def __init__(self):
        super(Model, self).__init__()
        self.model_ready = False

    def compute_model(self):
        raise NotImplementedError("Should have implemented this!")

    def run(self):
        """
        this method is called only when you want to run it in parallel.
        :return:
        """
        self.compute_model()

    def get_model(self):
        raise NotImplementedError("Should have implemented this!")