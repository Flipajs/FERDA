from PyQt4 import QtCore
from PyQt4 import QtGui

from gui.gui_utils import cvimg2qtpixmap
import transformation_trainer


class ViewWidget(QtGui.QWidget):
    def __init__(self, project, data):
        super(ViewWidget, self).__init__()
        self.project = project
        self.data = data
        self.regions = data.keys()
        print type(self.regions)
        for r in self.regions:
            print r[0].id(), r[1].id()

        self.setLayout(QtGui.QVBoxLayout())
        self.buttons = QtGui.QHBoxLayout()
        self.imgs = QtGui.QHBoxLayout()
        self.left = QtGui.QVBoxLayout()
        self.left_label = QtGui.QLabel()
        self.left_img = QtGui.QLabel()
        self.right = QtGui.QVBoxLayout()
        self.right_label = QtGui.QLabel()
        self.right_img = QtGui.QLabel()
        self.info = QtGui.QLabel()

        self.next_b = QtGui.QPushButton('next', self)
        self.prev_b = QtGui.QPushButton('previous', self)

        self.current_index = -1

        self._prepare_layouts()
        self._prepare_buttons()

        self.setWindowTitle('View Widget')

        if len(self.regions) > 0:
            self._next()
        self.prev_b.setDisabled(True)

    def _view(self, r1, r2):
        self._add_region_left(r1)
        self._add_region_right(r2)
        res = self.data[(r1, r2)]
        self.info.setText("Tagged {0}, should be {1}".format(res, not res))

    def _add_region_left(self, r):
        img = self.project.img_manager.get_crop(r.frame(), r, width=700, height=700, margin=300)
        self.left_label.setText('id: ' + str(r.id()))
        self.left_img.setPixmap(cvimg2qtpixmap(img))

    def _add_region_right(self, r):
        img = self.project.img_manager.get_crop(r.frame(), r, width=700, height=700, margin=300)
        self.right_label.setText('id: ' + str(r.id()))
        self.right_img.setPixmap(cvimg2qtpixmap(img))

    def _prepare_layouts(self):
        self.layout().addLayout(self.imgs)
        self.imgs.addLayout(self.left)
        self.imgs.addLayout(self.right)
        self.layout().addLayout(self.buttons)
        self.left.addWidget(self.left_label)
        self.left.addWidget(self.left_img)
        self.right.addWidget(self.right_label)
        self.right.addWidget(self.right_img)
        self.buttons.addWidget(self.info)
        self.left.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignBottom)
        self.right.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom)
        self.buttons.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignTop)

    def _prepare_buttons(self):
        self.buttons.addWidget(self.prev_b)
        self.buttons.addWidget(self.next_b)
        self.connect(self.prev_b, QtCore.SIGNAL('clicked()'), self._prev)
        self.connect(self.next_b, QtCore.SIGNAL('clicked()'), self._next)
        self.next_b.setFixedWidth(100)
        self.prev_b.setFixedWidth(100)

    def _next(self):
        if self.current_index < len(self.regions):
            self.current_index += 1
            self.current = self.regions[self.current_index]
            self._view(self.current[0], self.current[1])
            self.prev_b.setDisabled(False)
            if self.current_index == len(self.data) - 1:
                self.next_b.setDisabled(True)

    def _prev(self):
        if self.current_index >= 0:
            self.current_index -= 1
            self.current = self.regions[self.current_index]
            self._view(self.current[0], self.current[1])
            self.next_b.setDisabled(False)
            if self.current_index == 0:
                self.prev_b.setDisabled(True)
