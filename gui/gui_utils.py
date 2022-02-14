from PIL import ImageQt

from utils.drawing.points import get_contour, draw_points_crop

__author__ = 'fnaiser'

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage
from gui.settings_widgets.default import get_tooltip, get_default
import numpy as np


class IdButton(QtWidgets.QPushButton):
    def __init__(self, text, id):
        super(IdButton, self).__init__()
        self.id = id
        self.setText(text)


class SelectableQLabel(QtWidgets.QLabel):
    def __init__(self, parent=None, selected_callback=None, id=-1):
        QtWidgets.QLabel.__init__(self, parent)
        self.id_ = id
        self.selected_callback = selected_callback
        self.selected = False

        self.setMouseTracking(True)

    def mouseReleaseEvent(self, ev):
        if self.selected:
            self.set_selected(False)
        else:
            self.set_selected(True)
            if self.selected_callback:
                self.selected_callback(self, self.id_)

    def mouseMoveEvent(self, event):
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        # mbutt = QtGui.QApplication.mouseButtons()
        if modifiers == QtCore.Qt.ControlModifier:
            self.set_selected(True)
        # elif modifiers == QtCore.Qt.ControlModifier:
        #     self.set_selected(False)

    def set_selected(self, selected):
        if selected:
            self.setStyleSheet("border: 2px dashed red;")
            self.selected = True
        else:
            self.setStyleSheet("border: 0px;")
            self.selected = False


def set_input_field_bg_valid(input_widget, valid=True):
    palette = QtGui.QPalette()
    if valid:
        palette.setColor(QtGui.QPalette.Base, QtCore.Qt.white)
        # input_widget.setStyleSheet('QLineEdit { background: rgb(255, 0, 0); }')
    else:
        # input_widget.setStyleSheet('QLineEdit { background: rgb(255, 0, 0); }')
        palette.setColor(QtGui.QPalette.Base, QtCore.Qt.red)
    input_widget.setPalette(palette)


def file_name_dialog(window, text='Select files', path='', filter_=''):
    return str(QtWidgets.QFileDialog.getOpenFileName(window, text, path, filter=filter_))[0]


def file_names_dialog(window, text='Select files', path='', filter_=''):
    file_names_qstr = QtWidgets.QFileDialog.getOpenFileNames(window, text, path, filter=filter_)[0]
    return [str(s) for s in file_names_qstr]


def cvimg2qtpixmap(img):
    img_q = QImage(img.data, img.shape[1], img.shape[0], img.shape[1]*3, 13)
    pix_map = QtGui.QPixmap.fromImage(img_q.rgbSwapped())

    return pix_map


def reconnect(signal, newhandler=None, oldhandler=None):
    while True:
        try:
            if oldhandler is not None:
                signal.disconnect(oldhandler)
            else:
                signal.disconnect()
        except TypeError:
            break
    if newhandler is not None:
        signal.connect(newhandler)


def get_spin_box(from_=0, to=100, step=1, key=None):
    sb = QtWidgets.QSpinBox()
    sb.setRange(from_, to)
    sb.setSingleStep(step)
    if key:
        sb.setValue(get_settings(key, int))
        sb.setToolTip(get_tooltip(key))

    return sb


def get_double_spin_box(from_=0, to=1, step=0.01, key=None):
    sb = QtWidgets.QDoubleSpinBox()
    sb.setRange(from_, to)
    sb.setSingleStep(step)
    if key:
        sb.setValue(get_settings(key, int))
        sb.setToolTip(get_tooltip(key))

    return sb


def get_checkbox(text, key=None):
    ch = QtWidgets.QCheckBox(text)
    if key:
        ch.setToolTip(get_tooltip(key))
        ch.setChecked(get_settings(key, bool))

    return ch


def get_image_label(im):
    h = im.shape[0]
    w = im.shape[1]

    item = QtWidgets.QLabel()
    item.setScaledContents(True)
    item.setFixedWidth(w)
    item.setFixedHeight(h)
    item.setPixmap(get_pixmap_from_np_bgr(im))

    return item


def gbox_collapse_expand(gbox, height=35):
    # it is still not changed because it is called on toggle, so it is a little bit contra-intuitive
    if gbox.isChecked():
        gbox.setFixedHeight(gbox.sizeHint().height())
    else:
        gbox.setFixedHeight(height)


def get_img_qlabel(pts, img, id, height=100, width=100, filled=False):
    pts = np.asarray(pts, dtype=np.int32)
    if not filled:
        pts = get_contour(pts)
    crop = draw_points_crop(img, pts, (0, 0, 255, 0.5), square=True)

    img_q = ImageQt.QImage(crop.data, crop.shape[1], crop.shape[0], crop.shape[1] * 3, 13)
    pix_map = QtGui.QPixmap.fromImage(img_q.rgbSwapped())

    item = SelectableQLabel(id=id)

    item.setScaledContents(True)
    item.setFixedSize(height, width)
    item.setPixmap(pix_map)

    return item


class ClickableQGraphicsPixmapItem(QtWidgets.QGraphicsPixmapItem):
    def __init__(self, pixmap, id_, callback):
        super(ClickableQGraphicsPixmapItem, self).__init__(pixmap)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)
        self.id_ = id_
        self.callback = callback

    def mouseReleaseEvent(self, event):
        super(ClickableQGraphicsPixmapItem, self).mouseReleaseEvent(event)
        self.callback(self.id_)


class SelectAllLineEdit(QtWidgets.QLineEdit):
    def __init__(self, parent=None):
        super(SelectAllLineEdit, self).__init__(parent)
        self.readyToEdit = True
        self.setFixedHeight(15)

    def mousePressEvent(self, e, Parent=None):
        super(SelectAllLineEdit, self).mousePressEvent(e) #required to deselect on 2e click
        if self.readyToEdit:
            self.selectAll()
            self.readyToEdit = False

    def focusInEvent(self, e):
        super(SelectAllLineEdit, self).focusInEvent(e)
        if self.readyToEdit:
            self.selectAll()
            self.readyToEdit = False

    def focusOutEvent(self, e):
        super(SelectAllLineEdit, self).focusOutEvent(e) #required to remove cursor on focusOut
        self.deselect()
        self.readyToEdit = True


def get_pixmap_from_np_bgr(np_image):
    img_q = ImageQt.QImage(np_image.data, np_image.shape[1], np_image.shape[0], np_image.shape[1] * 3, 13)
    pix_map = QtGui.QPixmap.fromImage(img_q.rgbSwapped())
    return pix_map


def get_settings(key, type=str):
    settings = QtCore.QSettings('FERDA')
    return settings.value(key, get_default(key), type)


def set_settings(key, value):
    settings = QtCore.QSettings('FERDA')
    settings.setValue(key, value)
