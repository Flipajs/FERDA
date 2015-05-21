__author__ = 'fnaiser'

from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import QImage
from utils.misc import get_settings
import utils.img
from gui.settings.default import get_tooltip


class IdButton(QtGui.QPushButton):
    def __init__(self, text, id):
        super(IdButton, self).__init__()
        self.id = id
        self.setText(text)


class SelectableQLabel(QtGui.QLabel):
    def __init__(self, parent=None, selected_callback=None, id=-1):
        QtGui.QLabel.__init__(self, parent)
        self.id_ = id
        self.selected_callback = selected_callback
        self.selected = False

    def mouseReleaseEvent(self, ev):
        if self.selected:
            self.setStyleSheet("border: 0px;")
            self.selected = False
        else:
            self.setStyleSheet("border: 2px dashed black;")
            self.selected = True
            if self.selected_callback:
                self.selected_callback(self, self.id_)

def file_names_dialog(window, text='Select files', path='', filter_=''):
    file_names = QtGui.QFileDialog.getOpenFileNames(window, text, path, filter=filter_)

    names = []
    for f in file_names:
        names.append(str(f))

    return names

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
    sb = QtGui.QSpinBox()
    sb.setRange(from_, to)
    sb.setSingleStep(step)
    if key:
        sb.setValue(get_settings(key, int))
        sb.setToolTip(get_tooltip(key))

    return sb

def get_double_spin_box(from_=0, to=1, step=0.01, key=None):
    sb = QtGui.QDoubleSpinBox()
    sb.setRange(from_, to)
    sb.setSingleStep(step)
    if key:
        sb.setValue(get_settings(key, int))
        sb.setToolTip(get_tooltip(key))

    return sb

def get_checkbox(text, key=None):
    ch = QtGui.QCheckBox(text)
    if key:
        ch.setToolTip(get_tooltip(key))
        ch.setChecked(get_settings(key, bool))

    return ch

def get_image_label(im):
    h = im.shape[0]
    w = im.shape[1]

    item = QtGui.QLabel()
    item.setScaledContents(True)
    item.setFixedWidth(w)
    item.setFixedHeight(h)
    item.setPixmap(utils.img.get_pixmap_from_np_bgr(im))

    return item

def gbox_collapse_expand(gbox, height=35):
    # it is still not changed because it is called on toggle, so it is a little bit contra-intuitive
    if gbox.isChecked():
        gbox.setFixedHeight(gbox.sizeHint().height())
    else:
        gbox.setFixedHeight(height)
