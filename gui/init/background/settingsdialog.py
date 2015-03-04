from PyQt4.QtGui import *
from PyQt4.QtCore import *

default_settings = {
    'blur_distance': 10,
    'square_line_width': 5,
    'copy_square_color': QColor("lime"),
    'position_square_color': QColor("yellow"),
    'open_image': QKeySequence(Qt.Key_O),
    'save_image': QKeySequence(Qt.Key_S),
    'fix_image': QKeySequence(Qt.Key_F),
    'settings': QKeySequence(),
    'cancel_fixing': QKeySequence(Qt.Key_Escape)
}


def get_default(key):
    return default_settings[key]


class SettingsDialog(QDialog):

    def __init__(self, parent=None, actionlist=[]):
        super(SettingsDialog, self).__init__(parent, Qt.WindowTitleHint | Qt.WindowSystemMenuHint)

        self.actionlist = actionlist

        self.setWindowTitle("Settings")

        self.component_widget = QWidget(self)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.RestoreDefaults)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.buttonBox.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self.restore_defaults)

        self.layout.addWidget(QLabel("Blur diameter:"))
        self.blur_slider = QSlider(self)
        self.blur_slider.setRange(1, 50)
        self.blur_slider.setOrientation(Qt.Horizontal)
        self.layout.addWidget(self.blur_slider)

        self.layout.addWidget(QLabel("Width of square line:"))
        self.square_slider = QSlider(self)
        self.square_slider.setRange(1, 50)
        self.square_slider.setOrientation(Qt.Horizontal)
        self.layout.addWidget(self.square_slider)

        self.layout.addWidget(QLabel("Key bindings:"))
        self.table = QTableWidget(len(self.actionlist), 2, self)
        self.layout.addWidget(self.table)
        self.table.horizontalHeader().setVisible(False)
        self.table.verticalHeader().setVisible(False)
        for i in range(len(self.actionlist)):
            self.table.setItem(i, 0, QTableWidgetItem(self.actionlist[i].text()))
            self.table.item(i, 0).setFlags(Qt.NoItemFlags)
            self.table.setItem(i, 1, QTableWidgetItem())
            self.table.item(i, 1).setFlags(Qt.NoItemFlags | Qt.ItemIsEnabled)
        self.table.itemDoubleClicked.connect(self.bind_new_key)

        self.layout.addWidget(self.buttonBox)

        self.populate()

    def restore_defaults(self):
        self.blur_slider.setValue(get_default('blur_distance'))
        self.square_slider.setValue(get_default('square_line_width'))
        for i in range(len(self.actionlist)):
            self.table.item(i, 1).setText(get_default(str(self.actionlist[i].objectName())).toString())

    def populate(self):
        settings = QSettings("Background corrector")
        self.blur_slider.setValue(settings.value('blur_distance', get_default('blur_distance'), int))
        self.square_slider.setValue(settings.value('square_line_width', get_default('square_line_width'), int))
        for i in range(len(self.actionlist)):
            self.table.item(i, 1).setText(self.actionlist[i].shortcut().toString())

    def harvest(self):
        settings = QSettings("Background corrector")
        settings.setValue('blur_distance', self.blur_slider.value())
        settings.setValue('square_line_width', self.square_slider.value())
        for i in range(len(self.actionlist)):
            settings.setValue(self.actionlist[i].objectName(), QKeySequence(self.table.item(i, 1).text()))
            self.actionlist[i].setShortcut(QKeySequence(self.table.item(i, 1).text()))

    def done(self, p_int):
        if p_int == QDialog.Accepted:
            self.harvest()
        super(SettingsDialog, self).done(p_int)

    def bind_new_key(self, item):
        dialog = KeyBindingDialog(self)
        dialog.exec_()
        if dialog.Accepted:
            item.setText(dialog.shortcut.toString())


class KeyBindingDialog(QDialog):

    def __init__(self, parent=None):
        super(KeyBindingDialog, self).__init__(parent, Qt.WindowTitleHint | Qt.WindowSystemMenuHint)

        self.setWindowTitle("Bind new shortcut")

        self.shortcut = None

        self.main_layout = QVBoxLayout(self)
        self.setLayout(self.main_layout)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.button_box.rejected.connect(self.reject)

        self.main_layout.addWidget(QLabel("Press the shortcut you want to bind."))

        self.main_layout.addWidget(self.button_box)

    def keyPressEvent(self, event):
        if not event.text().isEmpty():
            keyInt = event.key()
            if event.modifiers() & Qt.ShiftModifier:
                keyInt += Qt.SHIFT
            if event.modifiers() & Qt.AltModifier:
                keyInt += Qt.ALT
            if event.modifiers() & Qt.ControlModifier:
                keyInt += Qt.CTRL
            if event.modifiers() & Qt.MetaModifier:
                keyInt += Qt.META
            self.shortcut = QKeySequence(keyInt)
            self.accept()
        else:
            super(KeyBindingDialog, self).keyPressEvent(event)