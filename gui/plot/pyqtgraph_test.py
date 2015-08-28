from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import pyqtgraph as pq
import numpy as np
import scripts.trajectories_data.eight_gt as data

def scatter(values):
    # show points separately
    pos = np.zeros((13000, 3))
    size = np.zeros(13000)
    colors = np.zeros((13000, 4))

    r = [1, 0, 0, 1]
    g = [0, 1, 0, 1]
    b = [0, 0, 1, 1]
    rg = [1, 1, 0, 1]
    rb = [1, 0, 1, 1]
    gb = [0, 1, 1, 1]
    rgb = [1, 1, 1, 1]
    black = [0, 0, 0, 1]

    k = 0
    for i in range(0, len(values)-1):
        for j in range(0, 8):
            pos_x = values[i][j][0]
            pos_y = values[i][j][1]
            if j == 0:
                color = r
            elif j == 1:
                color = g
            elif j == 2:
                color = b
            elif j == 3:
                color = rg
            elif j == 4:
                color = rb
            elif j == 5:
                color = gb
            elif j == 6:
                color = rgb
            else:
                color = black
            pos[k] = (pos_x, pos_y, i)
            size[k] = 2
            colors[k] = color
            k += 1
            #print "[x: %s, y: %s, z: %s]" % (pos_x, pos_y, i)

    sp1 = gl.GLScatterPlotItem(pos=pos, size=size, color=colors, pxMode=False)
    sp1.translate(-x_size/2, -y_size/2, 0)
    w.addItem(sp1)

def lines(values):
    # draw lines instead of single points
    r = (1, 0, 0, 1)
    g = (0, 1, 0, 1)
    b = (0, 0, 1, 1)
    rg = (1, 1, 0, 1)
    rb = (1, 0, 1, 1)
    gb = (0, 1, 1, 1)
    rgb = (1, 1, 1, 1)
    black = (0, 0, 0, 1)
    pos1 = np.zeros((len(values)-1, 3))
    pos2 = np.zeros((len(values)-1, 3))
    pos3 = np.zeros((len(values)-1, 3))
    pos4 = np.zeros((len(values)-1, 3))
    pos5 = np.zeros((len(values)-1, 3))
    pos6 = np.zeros((len(values)-1, 3))
    pos7 = np.zeros((len(values)-1, 3))
    pos8 = np.zeros((len(values)-1, 3))

    k = 0
    for i in range(0, len(values)-1):
        for j in range(0, 8):
            pos_x = values[i][j][0]
            pos_y = values[i][j][1]
            if j == 0:
                pos1[k] = (pos_x, pos_y, i/2)
            elif j == 1:
                pos2[k] = (pos_x, pos_y, i/2)
            elif j == 2:
                pos3[k] = (pos_x, pos_y, i/2)
            elif j == 3:
                pos4[k] = (pos_x, pos_y, i/2)
            elif j == 4:
                pos5[k] = (pos_x, pos_y, i/2)
            elif j == 5:
                pos6[k] = (pos_x, pos_y, i/2)
            elif j == 6:
                pos7[k] = (pos_x, pos_y, i/2)
            else:
                pos8[k] = (pos_x, pos_y, i/2)
        k += 1
    l1 = gl.GLLinePlotItem(pos=pos1, width=1.6, color=r)
    l2 = gl.GLLinePlotItem(pos=pos2, width=1.6, color=g)
    l3 = gl.GLLinePlotItem(pos=pos3, width=1.6, color=b)
    l4 = gl.GLLinePlotItem(pos=pos4, width=1.6, color=rg)
    l5 = gl.GLLinePlotItem(pos=pos5, width=1.6, color=gb)
    l6 = gl.GLLinePlotItem(pos=pos6, width=1.6, color=rb)
    l7 = gl.GLLinePlotItem(pos=pos7, width=1.6, color=rgb)
    l8 = gl.GLLinePlotItem(pos=pos8, width=1.6, color=black)
    l1.translate(-x_size/2, -y_size/2, 0)
    l2.translate(-x_size/2, -y_size/2, 0)
    l3.translate(-x_size/2, -y_size/2, 0)
    l4.translate(-x_size/2, -y_size/2, 0)
    l5.translate(-x_size/2, -y_size/2, 0)
    l6.translate(-x_size/2, -y_size/2, 0)
    l7.translate(-x_size/2, -y_size/2, 0)
    l8.translate(-x_size/2, -y_size/2, 0)
    w.addItem(l1)
    w.addItem(l2)
    w.addItem(l3)
    w.addItem(l4)
    w.addItem(l5)
    w.addItem(l6)
    w.addItem(l7)
    w.addItem(l8)

app = QtGui.QApplication([])
w = gl.GLViewWidget()
x_size = 1280
y_size = 1024
z_size = 1506
scale = 100
w.opts['distance'] = 2*x_size
w.show()
c = pq.mkColor(0, 0, 0)
w.setBackgroundColor(c)
w.setWindowTitle('Pyqtgraph test')


# WARN: Do not touch the grids, they seem weird, but they work
z = gl.GLGridItem()
z.setSize(y_size, x_size, 0)
z.setSpacing(scale, scale, scale)
w.addItem(z)

y = gl.GLGridItem()
y.setSize(z_size, x_size, 0)
y.setSpacing(scale, scale, scale)
y.rotate(90, 0, 1, 0)
y.translate(-y_size/2, 0, z_size/2)
w.addItem(y)

x = gl.GLGridItem()
x.setSize(y_size, z_size, 0)
x.setSpacing(scale, scale, scale)
x.rotate(90, 1, 0, 0)
x.translate(0, -x_size/2, z_size/2)
w.addItem(x)


values = data._old_vals
#scatter(values)
lines(values)

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()