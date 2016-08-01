from PyQt4 import QtGui, QtCore
import sys
from utils.paired_dict import PairedDict

class IdsNamesWidget(QtGui.QWidget):
    def __init__(self, callback):
        super(IdsNamesWidget, self).__init__()

        self.callback_ = callback

        self.vbox = QtGui.QVBoxLayout()
        self.setLayout(self.vbox)

        self.ids_vbox = QtGui.QVBoxLayout()
        self.vbox.addLayout(self.ids_vbox)
        self.query_hbox = QtGui.QHBoxLayout()
        self.vbox.addLayout(self.query_hbox)
        self.finish_button = QtGui.QPushButton('finish')
        self.finish_button.clicked.connect(self.finish)
        self.vbox.addWidget(self.finish_button)

        self.ids_labels = []
        self.id_names = []

        self.id_name = QtGui.QLineEdit()
        self.id_name.returnPressed.connect(self.add_id)
        self.add_id_button = QtGui.QPushButton('add id')
        self.add_id_button.clicked.connect(self.add_id)
        self.query_hbox.addWidget(self.id_name)
        self.query_hbox.addWidget(self.add_id_button)

    def add_id(self):
        name = str(self.id_name.text())
        self.id_names.append(name)
        self.id_name.setText('')
        lab_ = QtGui.QLabel(name)
        self.ids_labels.append(lab_)
        self.ids_vbox.addWidget(lab_)

    def finish(self):
        id_names_pd = PairedDict()
        for i, name in zip(range(len(self.id_names)), self.id_names):
            id_names_pd[i] = name

        self.callback_(id_names_pd)


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    ex = IdsNamesWidget()
    ex.show()

    app.exec_()
    app.deleteLater()
    sys.exit()