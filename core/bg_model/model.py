__author__ = 'fnaiser'

from PyQt4 import QtCore

class Model(QtCore.QThread):
    def __init__(self):
        super(Model, self).__init__()
        self.model_ready = False
        self.bg_model = None

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