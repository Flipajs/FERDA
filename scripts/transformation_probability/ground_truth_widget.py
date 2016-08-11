from PyQt4 import QtGui, QtCore
from gui.gui_utils import cvimg2qtpixmap
import transformation_classifier

class GroundTruthWidget(QtGui.QWidget):
    def __init__(self, project):
        super(GroundTruthWidget, self).__init__()
        self.project = project

        self.setLayout(QtGui.QVBoxLayout())
        self.buttons_l = QtGui.QHBoxLayout()
        self.imgs = QtGui.QHBoxLayout()
        self.left = QtGui.QVBoxLayout()
        self.left_img = QtGui.QLabel()
        self.right = QtGui.QVBoxLayout()
        self.right_img = QtGui.QLabel()

        self.no = QtGui.QPushButton('no (N)', self)
        self.yes = QtGui.QPushButton('yes (M)', self)
        self.no_action = QtGui.QAction('no', self)
        self.yes_action = QtGui.QAction('yes', self)

        self._prepare_layouts()
        self._prepare_buttons()

        self.setWindowTitle('Ground Truth Widget')

        self.results = {}
        self.regions = None

    def set_data(self, regions):
        self.regions = regions
        self._next()

    def _next(self):
        print len(self.regions)
        if self.regions and len(self.regions) > 0:
            self.current = self.regions.pop()
            self._resolve(self.current[0], self.current[1])
        else:
            self.buttons_l.addWidget(QtGui.QLabel("Every region from input already marked"))

    def get_results(self):
        return self.results

    def _resolve(self, r1, r2):
        self._add_region_left(r1)
        self._add_region_right(r2)

    def _add_region_left(self, r):
        img = self.project.img_manager.get_crop(r.frame(), r, width=300, height=300)
        self.left_img.setPixmap(cvimg2qtpixmap(img))

    def _add_region_right(self, r):
        img = self.project.img_manager.get_crop(r.frame(), r, width=300, height=300)
        self.right_img.setPixmap(cvimg2qtpixmap(img))

    def _prepare_layouts(self):
        self.layout().addLayout(self.imgs)
        self.imgs.addLayout(self.left)
        self.imgs.addLayout(self.right)
        self.layout().addLayout(self.buttons_l)

        self.left.addWidget(self.left_img)
        self.right.addWidget(self.right_img)
        self.left.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignBottom)
        self.right.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom)
        self.buttons_l.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignTop)

    def _prepare_buttons(self):
        self.buttons_l.addWidget(self.no)
        self.buttons_l.addWidget(self.yes)
        self.no_action.triggered.connect(self._no_action)
        self.yes_action.triggered.connect(self._yes_action)
        self.no_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_N))
        self.yes_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_M))
        self.no.addAction(self.no_action)
        self.yes.addAction(self.yes_action)
        self.yes.setFixedWidth(100)
        self.no.setFixedWidth(100)

    def _no_action(self):
        self.results[transformation_classifier.hash_region_tuple(self.current)] = False
        self._next()

    def _yes_action(self):
        self.results[transformation_classifier.hash_region_tuple(self.current)] = True
        self._next()