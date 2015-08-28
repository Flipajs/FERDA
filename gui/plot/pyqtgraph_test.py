
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import numpy as np
import scripts.trajectories_data.eight_gt as data


def get_color(id):
    if id == 0:
        return (1, 0, 0, 1)
    if id == 1:
        return (0, 1, 0, 1)
    if id == 2:
        return (0, 0, 1, 1)
    if id == 3:
        return (1, 1, 0, 1)
    if id == 4:
        return (1, 0, 1, 1)
    if id == 5:
        return (0, 1, 1, 1)
    if id == 6:
        return (1, 1, 1, 1)
    else:
        return (0, 0, 0, 1)

app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.opts['distance'] = 100
w.show()
w.setWindowTitle('pyqtgraph example: GLScatterPlotItem')

g = gl.GLGridItem()
w.addItem(g)


##
##  First example is a set of points with pxMode=False
##  These demonstrate the ability to have points with real size down to a very small scale 
## 
pos = np.zeros((13000, 3))
size = np.zeros(13000)
colors = np.zeros((13000, 4))
#pos[0] = (1,0,0); size[0] = 0.5;   color[0] = (1.0, 0.0, 0.0, 1)
#pos[1] = (0,1,0); size[1] = 0.2;   color[1] = (0.0, 0.0, 1.0, 1)
#pos[2] = (0,0,1); size[2] = 2./3.; color[2] = (0.0, 1.0, 0.0, 1)

r = [1, 0, 0, 1]
g = [0, 1, 0, 1]
b = [0, 0, 1, 1]
rg = [1, 1, 0, 1]
rb = [1, 0, 1, 1]
gb = [0, 1, 1, 1]
rgb = [1, 1, 1, 1]
black = [0, 0, 0, 1]

values = data._old_vals
k = 0
for i in range(0, 1505):
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
        print j
        #print "[x: %s, y: %s, z: %s]" % (pos_x, pos_y, i)

sp1 = gl.GLScatterPlotItem(pos=pos, size=size, color=colors, pxMode=False)
sp1.translate(-500, -500, 0)
w.addItem(sp1)


def update():
    """
    ## update volume colors
    global phase, sp2, d2
    s = -np.cos(d2*2+phase)
    color = np.empty((len(d2),4), dtype=np.float32)
    color[:,3] = np.clip(s * 0.1, 0, 1)
    color[:,0] = np.clip(s * 3.0, 0, 1)
    color[:,1] = np.clip(s * 1.0, 0, 1)
    color[:,2] = np.clip(s ** 3, 0, 1)
    sp2.setData(color=color)
    phase -= 0.1
    
    ## update surface positions and colors
    global sp3, d3, pos3
    z = -np.cos(d3*2+phase)
    pos3[:,2] = z
    color = np.empty((len(d3),4), dtype=np.float32)
    color[:,3] = 0.3
    color[:,0] = np.clip(z * 3.0, 0, 1)
    color[:,1] = np.clip(z * 1.0, 0, 1)
    color[:,2] = np.clip(z ** 3, 0, 1)
    sp3.setData(pos=pos3, color=color)
    """
    
t = QtCore.QTimer()
t.timeout.connect(update)
t.start(50)


## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()