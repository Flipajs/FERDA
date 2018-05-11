from PyQt4 import QtGui, QtCore
from gui.settings_widgets.parameters_tab import ParametersTab
from gui.settings_widgets.visualisation_tab import VisualisationTab


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

    def __init__(self, parent=None, settable_buttons=[]):
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

        self.visualisation_tab = VisualisationTab()
        self.tabWidget.addTab(self.visualisation_tab, "Visualisation")
        # self.tabWidget.setCurrentWidget(self.parameters_tab)

        self.layout = QtGui.QVBoxLayout()
        self.layout.setSizeConstraint(QtGui.QLayout.SetNoConstraint)
        self.layout.addWidget(self.tabWidget)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)

        self.populate()
        self.tabWidget.setCurrentIndex(3)

    def populate(self, ):
        # self.controls_tab.populate()
        # self.appearance_tab.populate()
        # self.test_tab.populate()
        # self.key_binding_tab.populate()
        pass

    def harvest_results(self):
        self.general_tab.harvest()
        self.parameters_tab.harvest()
        #self.key_binding_tab.harvest()
        self.visualisation_tab.harvest()

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
            'show_settings',  # FERDA
            'next_case',  # step by step conrrection tool
            'prev_case',  # step by step conrrection tool
            'confirm',  # step by step conrrection tool
            'partially_confirm',  # step by step conrrection tool
            'confirm_path',  # Step by step conrrection tool
            'fitting_from_left',  # Step by step conrrection tool
            'remove_region',  # Step by step conrrection tool
            'remove_chunk',  # Step by step conrrection tool, Global view
            'join_regions',  # Step by step conrrection tool
            'new_region',  # Step by step conrrection tool
            'ignore_case',  # Step by step conrrection tool, Global view

            'stop_action',  # Step by step conrrection tool
            'save',  # FERDA
            'save_only_long_enough',  # FERDA
            'undo',  # step by step conrrection tool, global view
            'get_info',  # others
            'hide_show',  # others

            'video_next',  # viewer
            'video_prev',  # viewer
            'video_play_pause',  # viewer
            'video_random_frame',  # others

            'global_view_join_chunks',  # global view
        ]

        super(KeyBindingsTab, self).__init__(parent)
        self.buttons = settable_buttons
        self.main_layout = QtGui.QVBoxLayout(self)
        self.setLayout(self.main_layout)
        self.table = QtGui.QTableWidget(len(settable_buttons), 2, self)
        self.main_layout.addWidget(self.table)
        self.table.horizontalHeader().setVisible(False)
        self.table.verticalHeader().setVisible(False)
        self.restore_defaults()
        self.table.itemDoubleClicked.connect(self.bind_new_key)

        self.main_layout.addWidget(QtGui.QLabel('To guarantee the functionality of new shortcuts, please restart \
        application after hitting OK button.'))

    def bind_new_key(self, item):
        dialog = KeyBindingDialog(self)
        dialog.exec_()
        if dialog.Accepted:
            item.setText(dialog.shortcut.toString())
        self.harvest()

    def restore_defaults(self):
        for i in range(len(self.buttons)):
            self.table.setItem(i, 0, QtGui.QTableWidgetItem(self.translate(self.buttons[i])))
            self.table.item(i, 0).setFlags(QtCore.Qt.NoItemFlags)
            s = eval('S_.controls.'+self.buttons[i]).toString()
            self.table.setItem(i, 1, QtGui.QTableWidgetItem(s))
            self.table.item(i, 1).setFlags(QtCore.Qt.NoItemFlags | QtCore.Qt.ItemIsEnabled)

    def harvest(self):
        graph = self.make_graph()
        conflicts = ConflictFinder().get_conflicts(graph)

        if len(conflicts) > 0:
            for shortcut, id in conflicts:
                self.restore_defaults()
                QtGui.QMessageBox.about(self, "Invalid shortcuts", "Sorry, the shortcut for '%s' in %s can't be set to %s. That key is already used in %s" % (shortcut.usage, shortcut.parent.name, shortcut.value, id))
                return False
        else:
            for i in range(len(self.buttons)):
                s = "S_.controls."+self.buttons[i]+" = QtGui.QKeySequence('"+self.table.item(i, 1).text()+"')"
                exec(str(s))
            return True

    def translate(self, key_name):
        k = key_name

        # if k == 'show_settings':
        #     return 'Show settings'
        # elif k == 'next_case':

        return k.replace('_', ' ')

    def make_graph(self):
        """
        load all shortcuts into a graph that describes their relations
        :return: first Node of the graph
        """

        root = Node("root", None)

        ferda = Node("FERDA", root)
        root.children.append(ferda)

        stbs = Node("Step by step", ferda)
        ferda.children.append(stbs)

        glob = Node("Global view", ferda)
        ferda.children.append(glob)

        view = Node("Viewer", ferda)
        ferda.children.append(view)

        othr = Node("Others", ferda)
        ferda.children.append(othr)

        for i in range(len(self.buttons)):
            value = self.table.item(i, 1).text()
            name = self.table.item(i, 0).text()

            ferda_commands = ["show settings", "save", "save only long enough"]
            stbs_commands = ["next case", "prev case", "confirm", "partially confirm", "confirm path", "fitting from left",
                "remove region", "remove chunk", "join regions", "new region", "ignore case", "stop action", "undo"]
            othr_commands = ["get info", "hide show", "video random frame"]
            view_commands = ["video next", "video prev", "video play pause"]
            glob_commands = ["remove chunk", "ignore case", "undo", "global view join chunks"]

            if name in ferda_commands:
                # print "Adding key %s (%s) to FERDA" % (value, name)
                ferda.shortcuts.append(Key(value, ferda, name))
            if name in stbs_commands:
                # print "Adding key %s (%s) to stbs" % (value, name)
                stbs.shortcuts.append(Key(value, stbs, name))
            if name in othr_commands:
                # print "Adding key %s (%s) to othr" % (value, name)
                othr.shortcuts.append(Key(value, othr, name))
            if name in view_commands:
                # print "Adding key %s (%s) to view" % (value, name)
                view.shortcuts.append(Key(value, view, name))
            if name in glob_commands:
                # print "Adding key %s (%s) to glob" % (value, name)
                glob.shortcuts.append(Key(value, glob, name))

        return root

class ConflictFinder():
    def __init__(self):
        self.conflicts = []

    def get_conflicts(self, graph):
        self.search_graph_(graph)
        return self.conflicts

    def contains(self, shortcuts, value):
        for sh in shortcuts:
            if sh.value == value:
                return True
        return False

    def search_graph_(self, parent):
        # temporary storage for all children's shortcuts
        short = []
        for child in parent.children:

            seen = []
            for sh in child.shortcuts:
                if sh.value not in seen:
                    seen.append(sh.value)
                else:
                    # print "Conflict! The key %s can be used only once in %s. (%s)" % (sh.value, child.name, sh.usage)
                    self.conflicts.append((sh, child.name))
                    child.shortcuts.remove(sh)

            if len(child.children) > 0:
                # load child's children
                shortcuts = self.search_graph_(child)
                # check all shortcuts
                for sh in shortcuts:
                    # check if it collides with any key in child.shortcuts
                    if self.contains(child.shortcuts, sh.value):
                        # print "Conflict! The key %s from %s can't be used also in %s. (%s)" % (sh.value, child.name, sh.parent.name, sh.usage)
                        self.conflicts.append((sh, child.name))
                    # add it to short
                    if not self.contains(short, sh):
                        short.append(sh)

            for sh in child.shortcuts:
                if not self.contains(short, sh):
                    short.append(sh)

        # short.sort()
        return short


class Node:
    def __init__(self, name, parent):
        self.name = name
        self.parent = parent
        self.shortcuts = []
        self.children = []


class Key:
    def __init__(self, value, parent, usage=""):
        self.value = value
        self.parent = parent
        self.usage = usage
