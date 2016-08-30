import numpy as np
import matplotlib.pyplot as plt
import random
from PyQt4 import QtCore
from PyQt4 import QtGui
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar

cnames = {
    'aqua': '#00FFFF',
    'aquamarine': '#7FFFD4',
    'black': '#000000',
    'blue': '#0000FF',
    'blueviolet': '#8A2BE2',
    'brown': '#A52A2A',
    'burlywood': '#DEB887',
    'cadetblue': '#5F9EA0',
    'chartreuse': '#7FFF00',
    'chocolate': '#D2691E',
    'coral': '#FF7F50',
    'cornflowerblue': '#6495ED',
    'crimson': '#DC143C',
    'cyan': '#00FFFF',
    'darkblue': '#00008B',
    'darkcyan': '#008B8B',
    'darkgoldenrod': '#B8860B',
    'darkgray': '#A9A9A9',
    'darkgreen': '#006400',
    'darkkhaki': '#BDB76B',
    'darkmagenta': '#8B008B',
    'darkolivegreen': '#556B2F',
    'darkorange': '#FF8C00',
    'darkorchid': '#9932CC',
    'darkred': '#8B0000',
    'darksalmon': '#E9967A',
    'darkseagreen': '#8FBC8F',
    'darkslateblue': '#483D8B',
    'darkslategray': '#2F4F4F',
    'darkturquoise': '#00CED1',
    'darkviolet': '#9400D3',
    'deeppink': '#FF1493',
    'deepskyblue': '#00BFFF',
    'dimgray': '#696969',
    'dodgerblue': '#1E90FF',
    'firebrick': '#B22222',
    'forestgreen': '#228B22',
    'fuchsia': '#FF00FF',
    'gold': '#FFD700',
    'goldenrod': '#DAA520',
    'gray': '#808080',
    'green': '#008000',
    'greenyellow': '#ADFF2F',
    'hotpink': '#FF69B4',
    'indianred': '#CD5C5C',
    'indigo': '#4B0082',
    'khaki': '#F0E68C',
    'lawngreen': '#7CFC00',
    'lime': '#00FF00',
    'limegreen': '#32CD32',
    'magenta': '#FF00FF',
    'maroon': '#800000',
    'midnightblue': '#191970',
    'moccasin': '#FFE4B5',
    'navy': '#000080',
    'olive': '#808000',
    'olivedrab': '#6B8E23',
    'orange': '#FFA500',
    'orangered': '#FF4500',
    'orchid': '#DA70D6',
    'palegoldenrod': '#EEE8AA',
    'palegreen': '#98FB98',
    'paleturquoise': '#AFEEEE',
    'palevioletred': '#DB7093',
    'peru': '#CD853F',
    'pink': '#FFC0CB',
    'plum': '#DDA0DD',
    'powderblue': '#B0E0E6',
    'purple': '#800080',
    'red': '#FF0000',
    'rosybrown': '#BC8F8F',
    'royalblue': '#4169E1',
    'saddlebrown': '#8B4513',
    'salmon': '#FA8072',
    'sandybrown': '#FAA460',
    'seagreen': '#2E8B57',
    'sienna': '#A0522D',
    'skyblue': '#87CEEB',
    'slateblue': '#6A5ACD',
    'slategray': '#708090',
    'springgreen': '#00FF7F',
    'steelblue': '#4682B4',
    'tan': '#D2B48C',
    'teal': '#008080',
    'thistle': '#D8BFD8',
    'tomato': '#FF6347',
    'turquoise': '#40E0D0',
    'violet': '#EE82EE',
    'yellow': '#FFFF00',
    'yellowgreen': '#9ACD32'}


class CustomSlider(QtGui.QSlider):
    SLIDE_MAX = 20
    SLIDE_MIN = -20
    SLIDE_STEP = 0.5

    def __init__(self, i, widget):
        super(CustomSlider, self).__init__(QtCore.Qt.Horizontal)
        self.setRange(self.SLIDE_MIN, self.SLIDE_MAX)
        self.setSingleStep(self.SLIDE_STEP)
        self.setTickInterval(self.SLIDE_MIN)
        self.i = i
        self.widget = widget
        widget.connect(self, QtCore.SIGNAL('valueChanged(int)'), self.valueChanged)

    def valueChanged(self, val):
        self.widget.ant[self.i] = val
        self.widget.labels[self.i].setText('{0:.2f}'.format(val))
        self.widget.plot()


class EigenViewer(FigureCanvas):
    def __init__(self, figure, slider):
        super(EigenViewer, self).__init__(figure)
        self.slider = slider

    def mousePressEvent(self, event):
        super(EigenViewer, self).mousePressEvent(event)
        dialog = QtGui.QInputDialog()
        dialog.setInputMode(QtGui.QInputDialog.DoubleInput)
        val, ok = dialog.getDouble(self, "", "Enter coefficient", min=self.slider.SLIDE_MIN,
                               max=self.slider.SLIDE_MAX)
        if ok:
            self.slider.setValue(val)
            self.slider.valueChanged(val)


class EigenWidget(QtGui.QWidget):
    def __init__(self, pca, eigens, eigen_vals, ant):
        super(EigenWidget, self).__init__()
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.ant = ant
        self.pca = pca
        self.right = QtGui.QVBoxLayout()
        self.left = QtGui.QVBoxLayout()
        self.prepare_canvas()

        self.labels = []
        self.eigens_layouts = []
        self.prepare_eigens(eigens, eigen_vals)

        l = QtGui.QHBoxLayout()
        for i in range(len(self.eigens_layouts)):
            if i % 5 == 0:
                self.right.addLayout(l)
                l = QtGui.QHBoxLayout()
            l.addLayout(self.eigens_layouts[i])
        else:
            self.right.addLayout(l)
            l = QtGui.QHBoxLayout()

        layout = QtGui.QHBoxLayout()
        layout.addLayout(self.left)
        layout.addLayout(self.right)
        self.setLayout(layout)

        self.plot()

    def prepare_canvas(self):
        self.left.addWidget(self.canvas)
        self.left.addWidget(self.toolbar)
        screen = QtGui.QDesktopWidget().screenGeometry()
        self.canvas.setFixedWidth(screen.width() / 3)

    def prepare_eigens(self, eigens, eigen_vals):
        max_val = max(eigen_vals)
        max_x = np.max(eigens[0, ::2])
        min_x = np.min(eigens[0, ::2])
        max_y = np.max(eigens[0, 1::2])
        min_y = np.min(eigens[0, 1::2])
        i = 0
        for eigen, eigen_val in zip(eigens, eigen_vals):
            layout = QtGui.QVBoxLayout()
            label = QtGui.QLabel('{0:.2f}'.format(float(self.ant[i])))
            label.setAlignment(QtCore.Qt.AlignCenter)
            self.labels.append(label)
            slider = CustomSlider(i, self)
            slider.setSliderPosition(self.ant[i])
            figure = plt.figure()
            ax = figure.add_subplot(111)
            a = random.randint(0, len(cnames) - 1)
            eigen = map (lambda x: x * (eigen_val / max_val), eigen)
            ax.plot(np.append(eigen[::2], eigen[0]), np.append(eigen[1::2], eigen[1]), c=cnames.keys()[a])
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.set_xlim([min_x, max_x])
            ax.set_ylim([min_y, max_y])
            canvas = EigenViewer(figure, slider)
            canvas.setSizePolicy(QtGui.QSizePolicy.Expanding,
                                 QtGui.QSizePolicy.Expanding)
            canvas.updateGeometry()
            canvas.draw()
            layout.addWidget(canvas)
            layout.addWidget(slider)
            layout.addWidget(label)
            self.eigens_layouts.append(layout)
            i += 1

    def plot(self):
        data = self.pca.inverse_transform(self.ant)
        ax = self.figure.add_subplot(111)
        ax.set_autoscale_on(False)
        ax.plot(np.append(data[::2], data[0]), np.append(data[1::2], data[1]))
        ax.grid(True)
        ax.set_xlim([-20, 20])
        ax.set_ylim([-40, 40])
        ax.hold(False)
        self.canvas.draw()

    def close_figures(self):
        plt.close('all')
