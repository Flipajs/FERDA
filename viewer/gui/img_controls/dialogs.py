from PyQt4 import QtGui, QtCore
import default_settings
import copy


class SettingsDialog(QtGui.QDialog):
    """A dialog used for settings of almost everything in the ants results tool. Note that the QSettings name is
    'Ants results tool'. When you need to add a settings, add it onto tab you want or add a new tab. The method
    populate of each tab is used to set initial values, the method restore_defaults populates the dialog with settings
    from default_settings file and the method harvest saves the values from dialog into settings. Should you add new
    settings, update all three methods accordingly. Also keep in mind that you need to add corresponding setting
    into default_settings.
    A propos: the construction for getting settings is:
    settings = QSettings("Ants results tool")
    settings.value(key, default_value, type)
    """

    def __init__(self, parent=None, settable_buttons = []):
        super(SettingsDialog, self).__init__(parent, QtCore.Qt.WindowTitleHint | QtCore.Qt.WindowSystemMenuHint)

        self.setWindowTitle("Settings")

        self.buttonBox = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Cancel | QtGui.QDialogButtonBox.RestoreDefaults)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.buttonBox.button(QtGui.QDialogButtonBox.RestoreDefaults).clicked.connect(self.restore_defaults)

        self.tabWidget = QtGui.QTabWidget()
        self.appearance_tab = AppearanceTab()
        self.tabWidget.addTab(self.appearance_tab, "Appearance")
        self.controls_tab = ControlsTab()
        self.tabWidget.addTab(self.controls_tab, "Controls")
        self.key_binding_tab = KeyBindingsTab(settable_buttons)
        self.tabWidget.addTab(self.key_binding_tab, "Key bindings")
        self.test_tab = FaultTestTab()
        self.tabWidget.addTab(self.test_tab, "Fault tests")

        self.layout = QtGui.QVBoxLayout()
        self.layout.setSizeConstraint(QtGui.QLayout.SetNoConstraint)
        self.layout.addWidget(self.tabWidget)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)

        self.populate()

    def populate(self, ):
        self.controls_tab.populate()
        self.appearance_tab.populate()
        self.test_tab.populate()
        self.key_binding_tab.populate()

    def harvest_results(self):

        self.controls_tab.harvest()
        self.appearance_tab.harvest()
        self.test_tab.harvest()
        self.key_binding_tab.harvest()

    def restore_defaults(self):

        self.tabWidget.currentWidget().restore_defaults()

    def done(self, p_int):
        if p_int == QtGui.QDialog.Accepted:
            self.harvest_results()
        super(SettingsDialog, self).done(p_int)


class AppearanceTab(QtGui.QWidget):

    def __init__(self, parent=None):
        super(AppearanceTab, self).__init__(parent)

        main_layout = QtGui.QVBoxLayout()

        main_layout.addWidget(QtGui.QLabel("View:"))
        self.individual_button = QtGui.QRadioButton("Individual ants")
        self.group_button = QtGui.QRadioButton("Groups")
        group = QtGui.QButtonGroup()
        group.addButton(self.individual_button)
        group.addButton(self.group_button)
        group.setExclusive(True)
        main_layout.addWidget(self.individual_button)
        main_layout.addWidget(self.group_button)

        main_layout.addWidget(QtGui.QLabel("Marker opacity:"))
        self.opacity_slider = QtGui.QSlider()
        self.opacity_slider.setRange(0, 1000)
        self.opacity_slider.setOrientation(QtCore.Qt.Horizontal)
        self.opacity_slider.setSingleStep(1)
        main_layout.addWidget(self.opacity_slider)

        main_layout.addWidget(QtGui.QLabel("Center marker size:"))
        self.center_marker_edit = QtGui.QSpinBox()
        self.center_marker_edit.setMinimum(0)
        main_layout.addWidget(self.center_marker_edit)

        main_layout.addWidget(QtGui.QLabel("Head marker size:"))
        self.head_marker_edit = QtGui.QSpinBox()
        self.head_marker_edit.setMinimum(0)
        main_layout.addWidget(self.head_marker_edit)

        main_layout.addWidget(QtGui.QLabel("Tail marker size:"))
        self.tail_marker_edit = QtGui.QSpinBox()
        self.tail_marker_edit.setMinimum(0)
        main_layout.addWidget(self.tail_marker_edit)

        main_layout.addWidget(QtGui.QLabel("Bottom panel height:"))
        self.bottom_panel_edit = QtGui.QSpinBox()
        self.bottom_panel_edit.setMinimum(10)
        self.bottom_panel_edit.setMaximum(400)
        main_layout.addWidget(self.bottom_panel_edit)

        main_layout.addWidget(QtGui.QLabel("Side panel width:"))
        self.side_panel_edit = QtGui.QSpinBox()
        self.side_panel_edit.setMinimum(20)
        self.side_panel_edit.setMaximum(400)
        main_layout.addWidget(self.side_panel_edit)

        self.setLayout(main_layout)
    def populate(self):
        settings = QtCore.QSettings("Ants results tool")
        if settings.value('view_mode', default_settings.get_default('view_mode'), str) == 'individual':
            self.individual_button.setChecked(True)
        elif settings.value('view_mode', default_settings.get_default('view_mode'), str) == 'group':
            self.group_button.setChecked(True)
        self.center_marker_edit.setValue(settings.value('center_marker_size', default_settings.get_default('center_marker_size'), int))
        self.head_marker_edit.setValue(settings.value('head_marker_size', default_settings.get_default('head_marker_size'), int))
        self.tail_marker_edit.setValue(settings.value('tail_marker_size', default_settings.get_default('tail_marker_size'), int))
        self.bottom_panel_edit.setValue(settings.value('bottom_panel_height', default_settings.get_default('bottom_panel_height'), int))
        self.side_panel_edit.setValue(settings.value('side_panel_width', default_settings.get_default('side_panel_width'), int))
        self.opacity_slider.setValue(int(settings.value('marker_opacity', default_settings.get_default('marker_opacity'), float) * 1000))

    def harvest(self):
        settings = QtCore.QSettings("Ants results tool")
        if self.individual_button.isChecked():
            settings.setValue('view_mode', 'individual')
        elif self.group_button.isChecked():
            settings.setValue('view_mode', 'group')
        settings.setValue('center_marker_size', self.center_marker_edit.value())
        settings.setValue('head_marker_size', self.head_marker_edit.value())
        settings.setValue('tail_marker_size', self.tail_marker_edit.value())
        settings.setValue('bottom_panel_height', self.bottom_panel_edit.value())
        settings.setValue('side_panel_width', self.side_panel_edit.value())
        settings.setValue('marker_opacity', float(self.opacity_slider.value())/1000)

    def restore_defaults(self):
        if default_settings.get_default('view_mode') == 'individual':
            self.individual_button.setChecked(True)
        elif default_settings.get_default('view_mode') == 'group':
            self.group_button.setChecked(True)
        self.center_marker_edit.setValue(default_settings.get_default('center_marker_size'))
        self.head_marker_edit.setValue(default_settings.get_default('head_marker_size'))
        self.tail_marker_edit.setValue(default_settings.get_default('tail_marker_size'))
        self.bottom_panel_edit.setValue(default_settings.get_default('bottom_panel_height'))
        self.side_panel_edit.setValue(default_settings.get_default('side_panel_width'))
        self.opacity_slider.setValue(int(default_settings.get_default('marker_opacity')*1000))


class FaultTestTab(QtGui.QWidget):

    def __init__(self, parent=None):
        super(FaultTestTab, self).__init__(parent)

        main_layout = QtGui.QVBoxLayout()

        main_layout.addWidget(QtGui.QLabel("Correction mode:"))
        self.individual_button = QtGui.QRadioButton("Individual ants")
        self.group_button = QtGui.QRadioButton("Groups")
        group = QtGui.QButtonGroup()
        group.addButton(self.individual_button)
        group.addButton(self.group_button)
        group.setExclusive(True)
        main_layout.addWidget(self.individual_button)
        main_layout.addWidget(self.group_button)

        main_layout.addWidget(QtGui.QLabel("Length tolerance:"))
        self.length_edit = QtGui.QLineEdit()
        main_layout.addWidget(self.length_edit)

        main_layout.addWidget(QtGui.QLabel("Minimal certainty:"))
        self.certainty_edit = QtGui.QLineEdit()
        main_layout.addWidget(self.certainty_edit)

        main_layout.addWidget(QtGui.QLabel("Minimal distance between ants:"))
        self.proximity_edit = QtGui.QLineEdit()
        main_layout.addWidget(self.proximity_edit)

        main_layout.addWidget(QtGui.QLabel("Maximal angle of turning:"))
        self.angle_edit = QtGui.QSpinBox()
        self.angle_edit.setMinimum(1)
        self.angle_edit.setMaximum(180)
        main_layout.addWidget(self.angle_edit)

        main_layout.addWidget(QtGui.QLabel("Tests used:"))
        self.length_checkbox = QtGui.QCheckBox("Length different from average")
        self.certainty_checkbox = QtGui.QCheckBox("Certainty too low")
        self.proximity_checkbox = QtGui.QCheckBox("Ants too close")
        self.angular_checkbox = QtGui.QCheckBox("Ant turns too fast")
        self.lost_checkbox = QtGui.QCheckBox("Tracker has lost the ant")
        self.collision_checkbox = QtGui.QCheckBox("Ants collide")
        self.overlap_checkbox = QtGui.QCheckBox("Ants overlap")
        main_layout.addWidget(self.length_checkbox)
        main_layout.addWidget(self.certainty_checkbox)
        main_layout.addWidget(self.proximity_checkbox)
        main_layout.addWidget(self.angular_checkbox)
        main_layout.addWidget(self.lost_checkbox)
        main_layout.addWidget(self.collision_checkbox)
        main_layout.addWidget(self.overlap_checkbox)

        self.setLayout(main_layout)

    def populate(self):
        settings = QtCore.QSettings("Ants results tool")
        if settings.value('correction_mode', default_settings.get_default('correction_mode'), str) == 'individual':
            self.individual_button.setChecked(True)
        elif settings.value('correction_mode', default_settings.get_default('correction_mode'), str) == 'group':
            self.group_button.setChecked(True)
        self.length_edit.setText(QtCore.QString.number(settings.value('length_tolerance', default_settings.get_default('length_tolerance'), float)))
        self.certainty_edit.setText(QtCore.QString.number(settings.value('minimal_certainty', default_settings.get_default('minimal_certainty'), float)))
        self.proximity_edit.setText(QtCore.QString.number(settings.value('proximity_tolerance', default_settings.get_default('proximity_tolerance'), float)))
        self.angle_edit.setValue(settings.value('angular_tolerance', default_settings.get_default('angular_tolerance'), int))
        self.length_checkbox.setChecked(settings.value('len_test', default_settings.get_default('len_test'), bool))
        self.certainty_checkbox.setChecked(settings.value('certainty_test', default_settings.get_default('certainty_test'), bool))
        self.proximity_checkbox.setChecked(settings.value('proximity_test', default_settings.get_default('proximity_test'), bool))
        self.angular_checkbox.setChecked(settings.value('angular_test', default_settings.get_default('angular_test'), bool))
        self.lost_checkbox.setChecked(settings.value('lost_test', default_settings.get_default('lost_test'), bool))
        self.collision_checkbox.setChecked(settings.value('collision_test', default_settings.get_default('collision_test'), bool))
        self.overlap_checkbox.setChecked(settings.value('overlap_test', default_settings.get_default('overlap_test'), bool))

    def harvest(self):
        settings = QtCore.QSettings("Ants results tool")
        if self.individual_button.isChecked():
            settings.setValue('correction_mode', 'individual')
        elif self.group_button.isChecked():
            settings.setValue('correction_mode', 'group')

        settings.setValue('len_test', self.length_checkbox.isChecked())
        settings.setValue('certainty_test', self.certainty_checkbox.isChecked())
        settings.setValue('proximity_test', self.proximity_checkbox.isChecked())
        settings.setValue('angular_test', self.angular_checkbox.isChecked())
        settings.setValue('lost_test', self.lost_checkbox.isChecked())
        settings.setValue('collision_test', self.collision_checkbox.isChecked())
        settings.setValue('overlap_test', self.overlap_checkbox.isChecked())
        try:
            if float(self.length_edit.text()) > 1:
                settings.setValue('length_tolerance', float(self.length_edit.text()))
            else:
                settings.setValue('length_tolerance', default_settings.get_default('length_tolerance'))
        except ValueError:
            settings.setValue('length_tolerance', default_settings.get_default('length_tolerance'))
        try:
            if float(self.certainty_edit.text()) > 0:
                settings.setValue('minimal_certainty', float(self.certainty_edit.text()))
            else:
                settings.setValue('minimal_certainty',default_settings.get_default('minimal_certainty'))
        except ValueError:
            settings.setValue('minimal_certainty', default_settings.get_default('minimal_certainty'))
        try:
            if float(self.proximity_edit.text()) > 0:
                settings.setValue('proximity_tolerance', float(self.proximity_edit.text()))
            else:
                settings.setValue('proximity_tolerance', default_settings.get_default('proximity_tolerance'))
        except ValueError:
            settings.setValue('proximity_tolerance', default_settings.get_default('proximity_tolerance'))
        settings.setValue('angular_tolerance', self.angle_edit.value())

    def restore_defaults(self):
        if default_settings.get_default('correction_mode') == 'individual':
            self.individual_button.setChecked(True)
        elif default_settings.get_default('correction_mode') == 'group':
            self.group_button.setChecked(True)
        self.length_edit.setText(QtCore.QString.number(default_settings.get_default('length_tolerance')))
        self.certainty_edit.setText(QtCore.QString.number(default_settings.get_default('minimal_certainty')))
        self.proximity_edit.setText(QtCore.QString.number(default_settings.get_default('proximity_tolerance')))
        self.angle_edit.setValue(default_settings.get_default('angular_tolerance'))
        self.length_checkbox.setChecked(default_settings.get_default('len_test'))
        self.certainty_checkbox.setChecked(default_settings.get_default('certainty_test'))
        self.proximity_checkbox.setChecked(default_settings.get_default('proximity_test'))
        self.angular_checkbox.setChecked(default_settings.get_default('angular_test'))
        self.lost_checkbox.setChecked(default_settings.get_default('lost_test'))
        self.collision_checkbox.setChecked(default_settings.get_default('collision_test'))
        self.overlap_checkbox.setChecked(default_settings.get_default('overlap_test'))


class ControlsTab(QtGui.QWidget):

    def __init__(self, parent=None):
        super(ControlsTab, self).__init__(parent)

        main_layout = QtGui.QVBoxLayout()

        main_layout.addWidget(QtGui.QLabel("Markers shown in history:"))
        self.center_button = QtGui.QRadioButton("Only center")
        self.all_button = QtGui.QRadioButton("All")
        self.history_group = QtGui.QButtonGroup()
        self.history_group.setExclusive(True)
        self.history_group.addButton(self.center_button)
        self.history_group.addButton(self.all_button)
        main_layout.addWidget(self.center_button)
        main_layout.addWidget(self.all_button)

        main_layout.addWidget(QtGui.QLabel("Undo/redo mode: (all currently undone changes will be lost)"))
        self.separate_button = QtGui.QRadioButton("Separate for individual frames")
        self.global_button = QtGui.QRadioButton("Global")
        self.undo_redo_group = QtGui.QButtonGroup()
        self.undo_redo_group.setExclusive(True)
        self.undo_redo_group.addButton(self.separate_button)
        self.undo_redo_group.addButton(self.global_button)
        main_layout.addWidget(self.separate_button)
        main_layout.addWidget(self.global_button)

        main_layout.addWidget(QtGui.QLabel("Head detection:"))
        self.head_checkbox = QtGui.QCheckBox("On")
        main_layout.addWidget(self.head_checkbox)

        main_layout.addWidget(QtGui.QLabel("Automatic zoom on faults:"))
        self.zoom_checkbox = QtGui.QCheckBox("On")
        main_layout.addWidget(self.zoom_checkbox)

        main_layout.addWidget(QtGui.QLabel("Fault results order:"))
        self.switch_first_button = QtGui.QRadioButton("Possible switches first")
        self.in_order_button = QtGui.QRadioButton("In order of appearance")

        self.order_group = QtGui.QButtonGroup()
        self.order_group.setExclusive(True)
        self.order_group.addButton(self.switch_first_button)
        self.order_group.addButton(self.in_order_button)
        main_layout.addWidget(self.switch_first_button)
        main_layout.addWidget(self.in_order_button)

        main_layout.addWidget(QtGui.QLabel("Depth of history shown:"))
        self.history_edit = QtGui.QSpinBox()
        self.history_edit.setMinimum(0)
        self.history_edit.setMaximum(9999999)
        main_layout.addWidget(self.history_edit)

        main_layout.addWidget(QtGui.QLabel("Number of forward positions shown:"))
        self.forward_edit = QtGui.QSpinBox()
        self.forward_edit.setMinimum(0)
        self.forward_edit.setMaximum(9999999)
        main_layout.addWidget(self.forward_edit)

        main_layout.addWidget(QtGui.QLabel("Number of changes for autosave:"))
        self.autosave_edit = QtGui.QSpinBox()
        self.autosave_edit.setMinimum(1)
        self.autosave_edit.setMaximum(100)
        main_layout.addWidget(self.autosave_edit)

        self.setLayout(main_layout)

    def populate(self):
        settings = QtCore.QSettings("Ants results tool")
        if settings.value('markers_shown_history', default_settings.get_default('markers_shown_history'), str) == 'all':
            self.all_button.setChecked(True)
        elif settings.value('markers_shown_history', default_settings.get_default('markers_shown_history'), str) == 'center':
            self.center_button.setChecked(True)

        if settings.value('undo_redo_mode', default_settings.get_default('undo_redo_mode'), str) == 'global':
            self.global_button.setChecked(True)
        elif settings.value('undo_redo_mode', default_settings.get_default('undo_redo_mode'), str) == 'separate':
            self.separate_button.setChecked(True)

        self.switch_first_button.setChecked(settings.value('switches_first', default_settings.get_default('switches_first'), bool))
        self.in_order_button.setChecked(not settings.value('switches_first', default_settings.get_default('switches_first'), bool))
        self.head_checkbox.setChecked(settings.value('head_detection', default_settings.get_default('head_detection'), bool))
        self.zoom_checkbox.setChecked(settings.value('zoom_on_faults', default_settings.get_default('zoom_on_faults'), bool))
        self.history_edit.setValue(settings.value('history_depth', default_settings.get_default('history_depth'), int))
        self.autosave_edit.setValue(settings.value('autosave_count', default_settings.get_default('autosave_count'), int))
        self.forward_edit.setValue(settings.value('forward_depth', default_settings.get_default('forward_depth'), int))

    def harvest(self):
        settings = QtCore.QSettings("Ants results tool")
        settings.setValue('history_depth', self.history_edit.value())
        settings.setValue('forward_depth', self.forward_edit.value())
        settings.setValue('autosave_count', self.autosave_edit.value())
        settings.setValue('head_detection', self.head_checkbox.isChecked())
        settings.setValue('zoom_on_faults', self.zoom_checkbox.isChecked())
        settings.setValue('switches_first', self.switch_first_button.isChecked())
        if self.center_button.isChecked():
            settings.setValue('markers_shown_history', 'center')
        elif self.all_button.isChecked():
            settings.setValue('markers_shown_history', 'all')
        if self.global_button.isChecked():
            settings.setValue('undo_redo_mode', 'global')
        elif self.separate_button.isChecked():
            settings.setValue('undo_redo_mode', 'separate')

    def restore_defaults(self):
        if default_settings.get_default('markers_shown_history') == 'all':
            self.all_button.setChecked(True)
        elif default_settings.get_default('markers_shown_history') == 'center':
            self.center_button.setChecked(True)

        if default_settings.get_default('undo_redo_mode') == 'global':
            self.global_button.setChecked(True)
        elif default_settings.get_default('undo_redo_mode') == 'separate':
            self.separate_button.setChecked(True)

        self.switch_first_button.setChecked(default_settings.get_default('switches_first'))
        self.in_order_button.setChecked(not default_settings.get_default('switches_first'))
        self.head_checkbox.setChecked(default_settings.get_default('head_detection'))
        self.zoom_checkbox.setChecked(default_settings.get_default('zoom_on_faults'))
        self.history_edit.setValue(default_settings.get_default('history_depth'))
        self.autosave_edit.setValue(default_settings.get_default('autosave_count'))
        self.forward_edit.setValue(default_settings.get_default('forward_depth'))


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
        super(KeyBindingsTab, self).__init__(parent)
        self.buttons = settable_buttons
        self.main_layout = QtGui.QVBoxLayout(self)
        self.setLayout(self.main_layout)
        self.table = QtGui.QTableWidget(len(settable_buttons), 2, self)
        self.main_layout.addWidget(self.table)
        self.table.horizontalHeader().setVisible(False)
        self.table.verticalHeader().setVisible(False)
        for i in range(len(settable_buttons)):
            self.table.setItem(i, 0, QtGui.QTableWidgetItem(settable_buttons[i].objectName()))
            self.table.item(i, 0).setFlags(QtCore.Qt.NoItemFlags)
            self.table.setItem(i, 1, QtGui.QTableWidgetItem())
            self.table.item(i, 1).setFlags(QtCore.Qt.NoItemFlags | QtCore.Qt.ItemIsEnabled)
        self.table.itemDoubleClicked.connect(self.bind_new_key)

    def bind_new_key(self, item):
        dialog = KeyBindingDialog(self)
        dialog.exec_()
        if dialog.Accepted:
            item.setText(dialog.shortcut.toString())

    def populate(self):
        for i in range(len(self.buttons)):
            self.table.item(i, 1).setText(self.buttons[i].shortcut().toString())

    def restore_defaults(self):
        for i in range(len(self.buttons)):
            self.table.item(i, 1).setText(default_settings.get_default(str(self.buttons[i].objectName())).toString())

    def harvest(self):
        settings = QtCore.QSettings("Ants results tool")
        for i in range(len(self.buttons)):
            settings.setValue(self.buttons[i].objectName(), QtGui.QKeySequence(self.table.item(i, 1).text()))
            self.buttons[i].setShortcut(QtGui.QKeySequence(self.table.item(i, 1).text()))