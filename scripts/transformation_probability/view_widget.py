from PyQt4 import QtCore
from PyQt4 import QtGui

from PyQt4.QtGui import QTableWidgetItem

from gui.gui_utils import cvimg2qtpixmap

MARGIN = 300
HEIGHT = 500
WIDTH = 700


class ViewWidget(QtGui.QWidget):
    def __init__(self, project, regions, classification, probability, classifier, avg_feat_v):
        super(ViewWidget, self).__init__()
        self.project = project
        self.regions = regions
        self.classification = classification
        self.classifier = classifier
        self.probability = probability
        self.avg_feat_v = avg_feat_v
        print avg_feat_v

        self.setLayout(QtGui.QVBoxLayout())
        self.buttons = QtGui.QHBoxLayout()
        self.table_layout = QtGui.QHBoxLayout()
        self.imgs = QtGui.QHBoxLayout()
        self.left = QtGui.QVBoxLayout()
        self.right = QtGui.QVBoxLayout()

        self.left_label = QtGui.QLabel()
        self.right_label = QtGui.QLabel()
        self.left_img = QtGui.QLabel()
        self.right_img = QtGui.QLabel()
        self.info = QtGui.QLabel()
        self.desc_label = QtGui.QLabel()

        self.table = QtGui.QTableWidget()
        self._prepare_table()

        self.next_b = QtGui.QPushButton('next', self)
        self.prev_b = QtGui.QPushButton('previous', self)

        self.current_index = -1
        self._prepare_layouts()
        self._prepare_buttons()

        self.setWindowTitle('View Widget')

        if len(self.regions) > 0:
            self._next()
        self.prev_b.setDisabled(True)

    def _prepare_table(self):
        self.table.setColumnCount(2)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setVisible(False)
        self.table.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

    def _view(self, r):
        self._add_region_left(r[0])
        self._add_region_right(r[1])
        res = self.classification[r]
        p = self.probability[r]
        self.desc_label.setText("Feature Vector")
        self._set_table(self.classifier.descriptor_representation(r))
        self.info.setText(
            "Tagged <b>{0}</b>, should be <b>{1}</b> with probability: F : {2} T : {3}".format(res, not res, p[0],
                                                                                               p[1]))

    def _set_table(self, data):
        self.table.setRowCount(len(data))
        for i, (f, v) in enumerate(data):
            self.table.setItem(i, 0, QTableWidgetItem(f))
            self.table.item(i, 0).setTextAlignment(QtCore.Qt.AlignRight)
            self.table.setItem(i, 1, QTableWidgetItem("{0:.2f}".format(v)))
            self.table.item(i, 1).setTextAlignment(QtCore.Qt.AlignRight)

        self.table.resizeColumnsToContents()
        self.table.setFixedSize(self.table.horizontalHeader().length() + self.table.verticalHeader().width(),
                                self.table.verticalHeader().length())

    def _add_region_left(self, r):
        img = self.project.img_manager.get_crop(r.frame(), r, width=WIDTH, height=HEIGHT, margin=MARGIN)
        self.left_label.setText('id: ' + str(r.id()))
        self.left_img.setPixmap(cvimg2qtpixmap(img))

    def _add_region_right(self, r):
        img = self.project.img_manager.get_crop(r.frame(), r, width=WIDTH, height=HEIGHT, margin=MARGIN)
        self.right_label.setText('id: ' + str(r.id()))
        self.right_img.setPixmap(cvimg2qtpixmap(img))

    def _prepare_layouts(self):
        self.layout().addWidget(self.desc_label)
        self.layout().addLayout(self.table_layout)
        self.layout().addWidget(self.info)
        self.layout().addLayout(self.imgs)
        self.layout().addLayout(self.buttons)
        self.table_layout.addWidget(self.table)
        self.imgs.addLayout(self.left)
        self.imgs.addLayout(self.right)
        self.left.addWidget(self.left_label)
        self.left.addWidget(self.left_img)
        self.right.addWidget(self.right_label)
        self.right.addWidget(self.right_img)
        self._format_components()

    def _format_components(self):
        self.left.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignBottom)
        self.right.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignBottom)
        self.buttons.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignTop)
        self.desc_label.setTextFormat(QtCore.Qt.RichText)
        self.desc_label.setAlignment(QtCore.Qt.AlignCenter)
        self.left_label.setAlignment(QtCore.Qt.AlignCenter)
        self.right_label.setAlignment(QtCore.Qt.AlignCenter)
        self.info.setAlignment(QtCore.Qt.AlignCenter)
        self.info.setStyleSheet("font-size: 20px;")

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
            self._view(self.current)
            self.prev_b.setDisabled(False)
            if self.current_index == len(self.regions) - 1:
                self.next_b.setDisabled(True)

    def _prev(self):
        if self.current_index >= 0:
            self.current_index -= 1
            self.current = self.regions[self.current_index]
            self._view(self.current)
            self.next_b.setDisabled(False)
            if self.current_index == 0:
                self.prev_b.setDisabled(True)
