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
            pos[k] = (pos_x, pos_y, i/2)
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

def image(file):
    import cv2
    import pyqtgraph as pg
    im = cv2.imread(file)
    im.astype("ubyte")
    pg.makeRGBA(im)
    print im.shape[2]
    cv2.imshow("test", im)
    cv2.waitKey(0)
    im_item = gl.GLImageItem(im)
    #im_item = pq.ImageItem(image=im)
    w.addItem(im_item)

def test():
    import pyqtgraph as pg
    ## create volume data set to slice three images from
    shape = (100,100, 3)
    #data = np.array((200, 200, 4), dtype="ubyte")
    data1 = np.array(shape, dtype="ubyte")
    data1[:] = (255, 0, 0)

    ## slice out three planes, convert to RGBA for OpenGL texture
    levels = (-0.08, 0.08)
    tex1 = pg.makeRGBA(data1[2], levels=levels)[0]       # yz plane




    ## Create three image items from textures, add to view
    v1 = gl.GLImageItem(tex1)
    #v1.translate(-shape[1]/2, -shape[2]/2, 0)
    v1.rotate(90, 0,0,1)
    v1.rotate(-90, 0,1,0)
    w.addItem(v1)

    ax = gl.GLAxisItem()
    w.addItem(ax)

def test2(file):
    import pyqtgraph as pg
    import cv2

    im = cv2.imread(file)
    im = im.astype("ubyte")

    print im

    ## slice out three planes, convert to RGBA for OpenGL texture
    tex1 = pg.makeRGBA(im)[0]

    ## Create three image items from textures, add to view
    v1 = gl.GLImageItem(tex1)
    v1.translate(-im.shape[0]/2, -im.shape[1]/2, 0)
    v1.rotate(90, 0,0,1)
    v1.rotate(-90, 0,1,0)
    w.addItem(v1)
    """
    v2 = gl.GLImageItem(tex2)
    v2.translate(-shape[0]/2, -shape[2]/2, 0)
    v2.rotate(-90, 1,0,0)
    w.addItem(v2)
    v3 = gl.GLImageItem(tex3)
    v3.translate(-shape[0]/2, -shape[1]/2, 0)
    w.addItem(v3)
    """

    ax = gl.GLAxisItem()
    w.addItem(ax)

app = QtGui.QApplication([])
w = gl.GLViewWidget()
x_size = 1280
y_size = 1024
z_size = 1506
scale = 100
#w.opts['distance'] = 2*x_size
w.show()
c = pq.mkColor(0, 0, 0)
w.setBackgroundColor(c)
w.setWindowTitle('Pyqtgraph test')
w.showMaximized()


"""
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
"""

"""
# These methods do not work when there is a GridItem in the view. Also, Z axis is never shown, even when it's added
# first. Perhaps it is black so it's just invisible?
axis_x = gl.GLAxisItem()
axis_x.setSize(x=x_size, y=0, z=0)
w.addItem(axis_x)
axis_y = gl.GLAxisItem()
axis_y.setSize(x=0, y=y_size, z=0)

w.addItem(axis_y)
axis_z = gl.GLAxisItem()
axis_z.setSize(x=0, y=0, z=z_size)
w.addItem(axis_z)
"""

values = data._old_vals
# scatter(values)
# lines(values)
#image("/home/dita/PycharmProjects/sample.png")
test2("/home/dita/PycharmProjects/sample.png")

#w.setCameraPosition(elevation=20, distance=2100)
#w.pan(0, 0, z_size/3)

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()