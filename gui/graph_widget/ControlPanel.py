from PyQt4 import QtCore
from PyQt4 import QtGui


class ControlPanel(QtGui.QWidget):

    def __init__(self, graph_w_callback):
        super(ControlPanel, self).__init__()
        self.l = QtGui.QHBoxLayout()
        self.prepare_buttons(graph_w_callback)
        self.setLayout(self.l)

    def prepare_button(self, label, callback, key):
        button = QtGui.QPushButton(label)
        action = QtGui.QAction(self)
        action.triggered.connect(callback)
        button.clicked.connect(callback)
        action.setShortcut(QtGui.QKeySequence(key))
        self.addAction(action)
        self.l.addWidget(button)

    def prepare_buttons(self, graph_w_callback):
        self.prepare_button('Show info (A)', graph_w_callback.info_manager.show_all_info, QtCore.Qt.Key_A)
        self.prepare_button('Hide info (S)', graph_w_callback.info_manager.hide_all_info, QtCore.Qt.Key_S)
        self.prepare_button('Show zoomed (T)', graph_w_callback.node_zoom_manager.show_zoom_all, QtCore.Qt.Key_T)
        self.prepare_button('Hide zoomed (Y)', graph_w_callback.node_zoom_manager.hide_zoom_all, QtCore.Qt.Key_Y)
        self.prepare_button('Zoom to chunk id (L)', graph_w_callback.zoom_to_chunk_event, QtCore.Qt.Key_L)
        self.prepare_button('Change Display (F5)', graph_w_callback.toggle_show_vertically, QtCore.Qt.Key_F5)
        self.prepare_button('Decompress Display (F4)', graph_w_callback.compress_axis_toggle, QtCore.Qt.Key_F4)
        self.prepare_button('Stretch (O)', graph_w_callback.stretch, QtCore.Qt.Key_O)
        self.prepare_button('Shrink (P)', graph_w_callback.shrink, QtCore.Qt.Key_P)
