from PyQt4 import QtCore
from PyQt4 import QtGui

from PyQt4.QtGui import QTableWidgetItem
from math import ceil

from transformation_trainer import hash_region_tuple
from gui.gui_utils import cvimg2qtpixmap

MARGIN = 200
HEIGHT = 400
WIDTH = 700

class ViewWidget(QtGui.QWidget):
    def __init__(self, project, regions, classification, probability, classifier, avg_feat_v_yes, std_yes, median_yes, avg_feat_v_no, std_no, median_no, n_false):
        super(ViewWidget, self).__init__()
        self.project = project
        self.regions = regions
        self.classification = classification
        self.classifier = classifier
        self.probability = probability
        self.avg_feat_v_yes = avg_feat_v_yes
        self.std_yes = std_yes
        self.median_yes = median_yes
        self.avg_feat_v_no = avg_feat_v_no
        self.std_no = std_no
        self.median_no = median_no
        self.n_false = n_false

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
        self.table.setRowCount(len(self.avg_feat_v_yes))
        self.table.setColumnCount(8)
        self.table.verticalHeader().setVisible(False)
        self.table.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.table.setHorizontalHeaderItem(0, QTableWidgetItem(""))
        self.table.setHorizontalHeaderItem(1, QTableWidgetItem(""))
        self.table.setHorizontalHeaderItem(2, QTableWidgetItem("avg_yes"))
        self.table.setHorizontalHeaderItem(3, QTableWidgetItem("median_yes"))
        self.table.setHorizontalHeaderItem(4, QTableWidgetItem("std_yes"))
        self.table.setHorizontalHeaderItem(5, QTableWidgetItem("avg_no"))
        self.table.setHorizontalHeaderItem(6, QTableWidgetItem("median_no"))
        self.table.setHorizontalHeaderItem(7, QTableWidgetItem("std_no"))
        self.table.horizontalHeaderItem(2).setTextAlignment(QtCore.Qt.AlignRight)
        self.table.horizontalHeaderItem(3).setTextAlignment(QtCore.Qt.AlignRight)
        self.table.horizontalHeaderItem(4).setTextAlignment(QtCore.Qt.AlignRight)
        self.table.horizontalHeaderItem(5).setTextAlignment(QtCore.Qt.AlignRight)
        self.table.horizontalHeaderItem(6).setTextAlignment(QtCore.Qt.AlignRight)
        self.table.horizontalHeaderItem(7).setTextAlignment(QtCore.Qt.AlignRight)
        for i, (v, s, m, v1, s1, m1) in enumerate(zip(self.avg_feat_v_yes, self.std_yes, self.median_yes,
                                                      self.avg_feat_v_no, self.std_no, self.median_no)):
            self.table.setItem(i, 2, QTableWidgetItem(str("{0:.2f}".format(v))))
            self.table.setItem(i, 3, QTableWidgetItem(str("{0:.2f}".format(m))))
            self.table.setItem(i, 4, QTableWidgetItem(str("{0:.2f}".format(s))))
            self.table.setItem(i, 5, QTableWidgetItem(str("{0:.2f}".format(v1))))
            self.table.setItem(i, 6, QTableWidgetItem(str("{0:.2f}".format(m1))))
            self.table.setItem(i, 7, QTableWidgetItem(str("{0:.2f}".format(s1))))
            self.table.item(i, 2).setTextAlignment(QtCore.Qt.AlignRight)
            self.table.item(i, 3).setTextAlignment(QtCore.Qt.AlignRight)
            self.table.item(i, 4).setTextAlignment(QtCore.Qt.AlignRight)
            self.table.item(i, 5).setTextAlignment(QtCore.Qt.AlignRight)
            self.table.item(i, 6).setTextAlignment(QtCore.Qt.AlignRight)
            self.table.item(i, 7).setTextAlignment(QtCore.Qt.AlignRight)
        self.table.resizeRowsToContents()

    def _view(self, r):
        self._add_region_left(r[0])
        self._add_region_right(r[1])
        res = self.classification[hash_region_tuple(r)]
        p = self.probability[hash_region_tuple(r)]
        self.desc_label.setText("Feature Vector")
        self._set_table(self.classifier.descriptor_representation(r))
        self.info.setText(
            "Tagged <b>{0}</b>, should be <b>{1}</b> with probability: F : {2} T : {3}".format(res,
                                                                                               not res if self.current_index < self.n_false else res, p[0], p[1]))

    def _set_table(self, data):
        for i, (f, v) in enumerate(data):
            self.table.setItem(i, 0, QTableWidgetItem(f))
            self.table.item(i, 0).setTextAlignment(QtCore.Qt.AlignRight)
            self.table.setItem(i, 1, QTableWidgetItem("{0:.2f}".format(v)))
            self.table.item(i, 1).setTextAlignment(QtCore.Qt.AlignRight)
        self.table.resizeColumnsToContents()
        self.table.setFixedSize(self.table.horizontalHeader().length(),
                                self.table.verticalHeader().length() + self.table.horizontalHeader().height())

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
