from PyQt6 import QtGui, QtWidgets

class SetColorsGui(QtWidgets.QWidget):
    def __init__(self, project):
        super(SetColorsGui, self).__init__()

        self.project = project

        self.fbox = QtWidgets.QFormLayout()
        self.setLayout(self.fbox)

        self.buttons = []
        from functools import partial
        for i, a in enumerate(project.animals):
            button = QtWidgets.QPushButton('change color')
            b, g, r = project.animals[i].color_
            button.setStyleSheet("background-color: rgb({}, {}, {});".format(r, g, b))
            button.clicked.connect(partial(self.change_color, i))
            self.buttons.append(button)

            self.fbox.addRow(str(i), button)

        self.save_b = QtWidgets.QPushButton('save')
        self.save_b.clicked.connect(self.save)

        self.fbox.addRow(self.save_b)

    def change_color(self, i):
        color = QtWidgets.QColorDialog.getColor()

        self.project.animals[i].color_ = (color.blue(), color.green(), color.red())
        self.buttons[i].setStyleSheet("background-color: %s;" % color.name())

    def save(self):
        self.project.save()
