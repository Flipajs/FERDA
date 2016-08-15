from PyQt4 import QtGui, QtCore
from gui.gui_utils import cvimg2qtpixmap
import transformation_trainer


class GroundTruthWidget(QtGui.QWidget):
    def __init__(self, project, classifier):
        super(GroundTruthWidget, self).__init__()
        self.project = project
        self.classifier = classifier

        self.setLayout(QtGui.QVBoxLayout())
        self.buttons_l = QtGui.QHBoxLayout()
        self.imgs = QtGui.QHBoxLayout()
        self.left = QtGui.QVBoxLayout()
        self.left_label = QtGui.QLabel()
        self.left_img = QtGui.QLabel()
        self.right = QtGui.QVBoxLayout()
        self.right_label = QtGui.QLabel()
        self.right_img = QtGui.QLabel()

        self.no = QtGui.QPushButton('no (N)', self)
        self.yes = QtGui.QPushButton('yes (M)', self)
        self.quit = QtGui.QPushButton('save and quit', self)

        self._prepare_layouts()
        self._prepare_buttons()

        self.setWindowTitle('Ground Truth Widget')

        self.results = {}
        self.regions = None

        self.last_left = - 1
        self.last_right = -1

    def set_data(self, regions):
        self.regions = regions
        self._next()

    def _next(self):
        if self.regions and len(self.regions) > 0:
            self.current = self.regions.pop()
            self._resolve(self.current[0], self.current[1])
        else:
            self.buttons_l.addWidget(QtGui.QLabel("Every region from input already marked"))
            self.no.setDisabled(True)
            self.yes.setDisabled(True)
            self.left_img.hide()
            self.right_img.hide()
            # self.close()

    def get_results(self):
        return self.results

    def _resolve(self, r1, r2):
        self._add_region_left(r1)
        self._add_region_right(r2)

    def _add_region_left(self, r):
        img = self.project.img_manager.get_crop(r.frame(), r, width=700, height=700, margin=300)
        self.left_label.setText('Last id: '  + str(self.last_left))
        self.last_left = r.id()
        self.left_img.setPixmap(cvimg2qtpixmap(img))

    def _add_region_right(self, r):
        img = self.project.img_manager.get_crop(r.frame(), r, width=700, height=700, margin=300)
        self.right_label.setText('Last id: ' + str(self.last_right))
        self.last_right = r.id()
        self.right_img.setPixmap(cvimg2qtpixmap(img))

    def _prepare_layouts(self):
        self.layout().addLayout(self.imgs)
        self.imgs.addLayout(self.left)
        self.imgs.addLayout(self.right)
        self.layout().addLayout(self.buttons_l)

        self.left.addWidget(self.left_label)
        self.left.addWidget(self.left_img)
        self.right.addWidget(self.right_label)
        self.right.addWidget(self.right_img)
        self.left.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignBottom)
        self.right.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom)
        self.buttons_l.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignTop)

    def _prepare_buttons(self):
        self.buttons_l.addWidget(self.no)
        self.buttons_l.addWidget(self.yes)
        self.buttons_l.addWidget(self.quit)
        self.connect(self.no, QtCore.SIGNAL('clicked()'), self.no_function)
        self.connect(self.yes, QtCore.SIGNAL('clicked()'), self.yes_function)
        self.connect(self.quit, QtCore.SIGNAL('clicked()'), self.close)
        self.connect(QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_N), self), QtCore.SIGNAL('activated()'),
                     self.no_function)
        self.connect(QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_M), self), QtCore.SIGNAL('activated()'),
                     self.yes_function)
        self.yes.setFixedWidth(100)
        self.no.setFixedWidth(100)

    def no_function(self):
        self.results[transformation_trainer.hash_region_tuple(self.current)] = False
        self._next()

    def yes_function(self):
        self.results[transformation_trainer.hash_region_tuple(self.current)] = True
        self._next()

    def closeEvent(self, QCloseEvent):
        super(GroundTruthWidget, self).closeEvent(QCloseEvent)
        self.classifier.accept_results(self.results)
