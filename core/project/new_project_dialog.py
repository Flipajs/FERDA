__author__ = 'filip@naiser.cz'

import sys
from PyQt4 import QtCore, QtGui

class NewProjectDialog(QtGui.QDialog):
    """A dialog used for new project creation"""

    def __init__(self, parent=None, settable_buttons = []):
        super(NewProjectDialog, self).__init__(parent, QtCore.Qt.WindowTitleHint | QtCore.Qt.WindowSystemMenuHint)

        self.setWindowTitle("New Project")

        self.buttonBox = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Cancel | QtGui.QDialogButtonBox.RestoreDefaults)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        # self.buttonBox.button(QtGui.QDialogButtonBox.RestoreDefaults).clicked.connect(self.restore_defaults)

        self.layout = QtGui.QFormLayout(self)
        self.layout.setSizeConstraint(QtGui.QLayout.SetNoConstraint)
        # self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)

        label = QtGui.QLabel('Project name', self)
        self.project_name = QtGui.QLineEdit(self)
        self.layout.addRow(label, self.project_name)

        label = QtGui.QLabel('Project description', self)
        self.project_description = QtGui.QLineEdit(self)
        self.layout.addRow(label, self.project_description)

        self.layout.addRow(self.buttonBox)

    def test(self):
        print self.label

    def accept(self):
        print self.project_name.text

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    d = NewProjectDialog()
    d.exec_()