
from gui.settings.default import get_default
from PyQt4 import QtGui, QtCore
import copy
from gui.settings.parameters_tab import ParametersTab
from gui.settings.general_tab import GeneralTab
from core.settings import Settings as S_


class SettingsDialog(QtGui.QDialog):
    """A dialog used for settings of almost everything in the ants correction tool. Note that the QSettings name is
    'Ants correction tool'. When you need to add a settings, add it onto tab you want or add a new tab. The method
    populate of each tab is used to set initial values, the method restore_defaults populates the dialog with settings
    from default_settings file and the method harvest saves the values from dialog into settings. Should you add new
    settings, update all three methods accordingly. Also keep in mind that you need to add corresponding setting
    into default_settings.
    A propos: the construction for getting settings is:
    settings = QSettings("Ants correction tool")
    settings.value(key, default_value, type)
    """

    def __init__(self, parent=None, settable_buttons = []):
        super(SettingsDialog, self).__init__(parent, QtCore.Qt.WindowTitleHint | QtCore.Qt.WindowSystemMenuHint)

        self.setStyleSheet("""QToolTip {
                            font-size: 15px;
                           border: black solid 1px;
                           }""")

        self.setWindowTitle("Settings")

        self.buttonBox = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Cancel | QtGui.QDialogButtonBox.RestoreDefaults)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.buttonBox.button(QtGui.QDialogButtonBox.RestoreDefaults).clicked.connect(self.restore_defaults)

        self.tabWidget = QtGui.QTabWidget()
        self.general_tab = GeneralTab()
        self.tabWidget.addTab(self.general_tab, "General")
        # self.appearance_tab = AppearanceTab()
        # self.tabWidget.addTab(self.appearance_tab, "Appearance")
        # self.controls_tab = ControlsTab()
        # self.tabWidget.addTab(self.controls_tab, "Controls")
        self.key_binding_tab = KeyBindingsTab(settable_buttons)
        self.tabWidget.addTab(self.key_binding_tab, "Key bindings")
        # self.test_tab = FaultTestTab()
        # self.tabWidget.addTab(self.test_tab, "Fault tests")

        self.parameters_tab = ParametersTab()
        self.tabWidget.addTab(self.parameters_tab, "Parameters")
        # self.tabWidget.setCurrentWidget(self.parameters_tab)

        self.layout = QtGui.QVBoxLayout()
        self.layout.setSizeConstraint(QtGui.QLayout.SetNoConstraint)
        self.layout.addWidget(self.tabWidget)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)

        self.populate()
        self.tabWidget.setCurrentIndex(1)

    def populate(self, ):
        # self.controls_tab.populate()
        # self.appearance_tab.populate()
        # self.test_tab.populate()
        # self.key_binding_tab.populate()
        pass

    def harvest_results(self):
        self.general_tab.harvest()
        self.parameters_tab.harvest()
        # self.
        # self.controls_tab.harvest()
        # self.appearance_tab.harvest()
        # self.test_tab.harvest()
        self.key_binding_tab.harvest()

    def restore_defaults(self):
        self.tabWidget.currentWidget().restore_defaults()

    def done(self, p_int):
        if p_int == QtGui.QDialog.Accepted:
            self.harvest_results()
        super(SettingsDialog, self).done(p_int)


class KeyBindingDialog(QtGui.QDialog):

    def __init__(self, parent=None):
        super(KeyBindingDialog, self).__init__(parent, QtCore.Qt.WindowTitleHint | QtCore.Qt.WindowSystemMenuHint)

        self.setWindowTitle("Bind new shortcut")

        self.shortcut = None

        self.main_layout = QtGui.QVBoxLayout(self)
        self.setLayout(self.main_layout)

        self.button_box = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Cancel)
        self.button_box.rejected.connect(self.reject)

        self.main_layout.addWidget(QtGui.QLabel("Press the shortcut you want to bind."))

        self.main_layout.addWidget(self.button_box)

    def keyPressEvent(self, event):
        if not event.text().isEmpty():
            keyInt = event.key()
            if event.modifiers() & QtCore.Qt.ShiftModifier:
                keyInt += QtCore.Qt.SHIFT
            if event.modifiers() & QtCore.Qt.AltModifier:
                keyInt += QtCore.Qt.ALT
            if event.modifiers() & QtCore.Qt.ControlModifier:
                keyInt += QtCore.Qt.CTRL
            if event.modifiers() & QtCore.Qt.MetaModifier:
                keyInt += QtCore.Qt.META
            self.shortcut = QtGui.QKeySequence(keyInt)
            self.accept()
        else:
            super(KeyBindingDialog, self).keyPressEvent(event)


class KeyBindingsTab(QtGui.QWidget):

    def __init__(self, settable_buttons=[], parent=None):
        settable_buttons = [
            'show_settings',
            'next_case',
            'prev_case',
            'confirm',
            'partially_confirm'
        ]

        super(KeyBindingsTab, self).__init__(parent)
        self.buttons = settable_buttons
        self.main_layout = QtGui.QVBoxLayout(self)
        self.setLayout(self.main_layout)
        self.table = QtGui.QTableWidget(len(settable_buttons), 2, self)
        self.main_layout.addWidget(self.table)
        self.table.horizontalHeader().setVisible(False)
        self.table.verticalHeader().setVisible(False)
        for i in range(len(settable_buttons)):
            self.table.setItem(i, 0, QtGui.QTableWidgetItem(self.translate(settable_buttons[i])))
            self.table.item(i, 0).setFlags(QtCore.Qt.NoItemFlags)
            s = eval('S_.controls.'+settable_buttons[i]).toString()
            self.table.setItem(i, 1, QtGui.QTableWidgetItem(s))
            self.table.item(i, 1).setFlags(QtCore.Qt.NoItemFlags | QtCore.Qt.ItemIsEnabled)
        self.table.itemDoubleClicked.connect(self.bind_new_key)

    def bind_new_key(self, item):
        dialog = KeyBindingDialog(self)
        dialog.exec_()
        if dialog.Accepted:
            item.setText(dialog.shortcut.toString())

    def restore_defaults(self):
        for i in range(len(self.buttons)):
            self.table.item(i, 1).setText('test')

    def harvest(self):
        for i in range(len(self.buttons)):

            # s = 'S_.controls.'+self.buttons[i]+' = QtGui.QKeySequence(QtCore.Qt.Key_'+self.table.item(i, 1).text()+')'
            s = 'S_.controls.'+self.buttons[i]+' = QtGui.QKeySequence(\''+self.table.item(i, 1).text()+'\')'
            exec(str(s))
            # print S_.controls.__getattribute__('show_settings')
            # S_.controls.__setattr__(self.buttons[i], QtGui.QKeySequence(self.table.item(i, 1).text()))
            # self.buttons[i][0] = QtGui.QKeySequence(self.table.item(i, 1).text())

    def translate(self, key_name):
        k = key_name

        # if k == 'show_settings':
        #     return 'Show settings'
        # elif k == 'next_case':

        return k.replace('_', ' ')