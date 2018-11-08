from __future__ import print_function
from __future__ import unicode_literals
from builtins import str
from builtins import range
from builtins import object
from PyQt4 import QtGui, QtCore
import sys
from functools import partial


class EditTrackletAdvanced(QtGui.QWidget):
    def __init__(self, tracklet, num_animals, callback):
        super(EditTrackletAdvanced, self).__init__()

        self.callback = callback
        self.num_animals = num_animals
        self.tracklet = tracklet

        self.vbox = QtGui.QVBoxLayout()

        self.setLayout(self.vbox)

        self.vbox.addWidget(QtGui.QLabel('tracklet id: '+str(tracklet.id())))
        self.hbox = QtGui.QHBoxLayout()
        self.vbox.addLayout(self.hbox)
        self.ps_layout = QtGui.QVBoxLayout()
        self.ns_layout = QtGui.QVBoxLayout()

        self.hbox.addLayout(self.ps_layout)
        self.hbox.addLayout(self.ns_layout)

        self.fix_tracklet_only_b = QtGui.QPushButton('fix_tracklet_only')
        self.fix_tracklet_only_b.clicked.connect(partial(self.confirm, 'fix_tracklet_only'))
        self.vbox.addWidget(self.fix_tracklet_only_b)

        self.fix_affected_b = QtGui.QPushButton('fix_affected')
        self.fix_affected_b.clicked.connect(partial(self.confirm, 'fix_affected'))
        self.vbox.addWidget(self.fix_affected_b)

        self.ns = []
        self.ps = []

        for i in range(self.num_animals):
            ch = QtGui.QCheckBox(str(i))
            ch.setChecked(False)

            if i in tracklet.P:
                ch.setChecked(True)
            ch.stateChanged.connect(partial(self.p_changed, i))

            self.ps_layout.addWidget(ch)
            self.ps.append(ch)

            ch = QtGui.QCheckBox(str(i))
            ch.setChecked(False)
            if i in tracklet.N:
                ch.setChecked(True)
            ch.stateChanged.connect(partial(self.n_changed, i))
            self.ns_layout.addWidget(ch)
            self.ns.append(ch)

        self.setFixedSize(self.minimumSizeHint())
        # TODO: print conflicting chunks

    def p_changed(self, i):
        if self.ps[i].isChecked():
            self.ns[i].setChecked(False)

    def n_changed(self, i):
        if self.ns[i].isChecked():
            self.ps[i].setChecked(False)

    def confirm(self, method):
        P = set()
        N = set()
        for i in range(self.num_animals):
            if self.ps[i].isChecked():
                P.add(i)

            if self.ns[i].isChecked():
                N.add(i)

        self.callback(self.tracklet, P, N, method=method)
        self.close()


class FakeTracklet(object):
    def __init__(self):
        self.N = set()
        self.P = set()
        self.id_ = 1

    def id(self):
        return self.id_


def confirmed(tracklet, P, N, method='fix_tracklet_only'):
    print(P, N, method)


if __name__ == '__main__':
    import sys
    app = QtGui.QApplication(sys.argv)

    tracklet = FakeTracklet()

    tracklet.N = set([1, 2])
    tracklet.P = set([3])

    ex = EditTrackletAdvanced(tracklet, 6, confirmed)
    ex.showMaximized()

    app.exec_()
    app.deleteLater()
    sys.exit()
