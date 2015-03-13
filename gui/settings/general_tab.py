__author__ = 'fnaiser'

from PyQt4 import QtGui, QtCore
from gui.settings.default import get_tooltip
from gui import gui_utils
from core.settings import Settings as S_

class GeneralTab(QtGui.QWidget):
    def __init__(self):
        super(GeneralTab, self).__init__()

        self.vbox = QtGui.QVBoxLayout()
        self.setLayout(self.vbox)


        self.cache_box = QtGui.QGroupBox('Cache')
        self.cache_box.setCheckable(True)
        self.cache_box.setChecked(S_.cache.use)
        self.cache_box.toggled.connect(lambda : gui_utils.gbox_collapse_expand(self.cache_box))

        if not self.cache_box.isChecked():
            gui_utils.gbox_collapse_expand(self.cache_box)

        self.cache_box.setAlignment(QtCore.Qt.AlignLeft)
        self.cache_box.setLayout(QtGui.QFormLayout())


        self.cache_mser = gui_utils.get_checkbox('', 'cache_mser')
        self.cache_box.layout().addRow('store MSERS', self.cache_mser)

        self.clear_cache = QtGui.QPushButton('Clear cache')
        self.clear_cache.clicked.connect(self.clear_cache_)

        self.vbox.addWidget(self.cache_box)
        self.populate()

    def populate(self):
        return
        # self.igbr_i_weight.setValue(get_settings('igbr_i_weight', float))

    def restore_defaults(self):
        # TODO
        return

    def clear_cache_(self):
        raise Exception("CLEAR CACHE - clear_cache_ in gui.settings.general_tab is not implemented!")