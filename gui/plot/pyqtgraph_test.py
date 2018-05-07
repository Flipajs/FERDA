import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui

import scripts.trajectories_data.eight_gt as data
import utils.img_manager as imm
from core.project.project import Project
from gui.settings import Settings as S_


class FrameLoader(QtCore.QThread):
    proc_done = QtCore.pyqtSignal(object)
    part_done = QtCore.pyqtSignal(float)

    def __init__(self, imm, frame, limit):
        super(FrameLoader, self).__init__()

        self.imm = imm
        self.frame = frame
        self.limit = limit

    def set_frame(self, frame):
        self.frame = frame

    def run(self):
        for i in range (self.frame, self.frame+self.limit):
            self.imm.get_whole_img(i)

class MyView(QtGui.QWidget):
    def __init__(self, project):
        super(MyView, self).__init__()

        self.w = gl.GLViewWidget()
        self.imm = imm.ImgManager(project, max_size_mb=S_.cache.img_manager_size_MB)
        self.w.setCameraPosition(elevation=20, distance=2100)
        self.w.setVisible(True)

        tmp_img = self.imm.get_whole_img(0)
        self.x_size = tmp_img.shape[0]
        self.y_size = tmp_img.shape[1]
        self.z_size = 1506
        self.scale = 100

        self.frame = 0
        self.is_loading = False

        self.values = data._inverted_vals

        # color has to be black, unfortunately. White (or any other color lighter than black) hides all the plot data.
        c = pg.mkColor(0, 0, 0)
        self.w.setBackgroundColor(c)

        self.add_grids()
        self.move_image()
        self.lines()
        self.frame_scatter()

        self.setLayout(QtGui.QVBoxLayout())
        self.layout().setAlignment(QtCore.Qt.AlignBottom)
        # Frame slider
        self.slider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.slider.setFocusPolicy(QtCore.Qt.NoFocus)
        self.slider.setGeometry(30, 40, 50, 30)
        self.slider.setRange(0, self.z_size)
        self.slider.setTickInterval(50)
        self.slider.setValue(0)
        self.slider.setTickPosition(QtGui.QSlider.TicksBelow)
        self.slider.valueChanged[int].connect(self.change_frame)
        self.slider.sliderReleased.connect(self.load_frames)

        self.loading_thread = FrameLoader(self.imm, self.frame, 10)
        self.loading_thread.proc_done.connect(self.loading_done)
        self.load_frames()

        print type(self.w)
        self.w.setSizePolicy(pg.QtGui.QSizePolicy.Expanding, pg.QtGui.QSizePolicy.Expanding)

        self.layout().addWidget(self.w)
        self.layout().addWidget(self.slider)

    def scatter(self, values):
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
                pos_y = values[i][j][0]
                pos_x = values[i][j][1]
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
        #sp1.translate(-self.x_size/2, -self.y_size/2, 0)
        self.w.addItem(sp1)

    def frame_scatter(self):
        try:
            self.w.removeItem(self.scatter)
        except:
            pass
        # show points separately
        pos = np.zeros((10, 3))
        size = np.zeros(10)
        colors = np.zeros((10, 4))

        r = [1, 0, 0, 1]
        g = [0, 1, 0, 1]
        b = [0, 0, 1, 1]
        rg = [1, 1, 0, 1]
        rb = [1, 0, 1, 1]
        gb = [0, 1, 1, 1]
        rgb = [1, 1, 1, 1]
        black = [0, 0, 0, 1]
        k = 0
        for j in range(0,8):
            pos_y = self.values[self.frame][j][0]
            pos_x = self.values[self.frame][j][1]
            if j == 0:
                color = r
            elif j == 1:
                color = g
            elif j == 2:
                color = b
            elif j == 3:
                color = rg
            elif j == 4:
                color = gb
            elif j == 5:
                color = rb
            elif j == 6:
                color = rgb
            else:
                color = black
            pos[k] = (pos_x, pos_y, self.frame)
            size[k] = 20
            colors[k] = color
            k += 1

        sp1 = gl.GLScatterPlotItem(pos=pos, size=size, color=colors, pxMode=False)
        self.scatter = sp1
        self.w.addItem(sp1)

    def lines(self):
        # draw lines instead of single points
        r = (1, 0, 0, 1)
        g = (0, 1, 0, 1)
        b = (0, 0, 1, 1)
        rg = (1, 1, 0, 1)
        rb = (1, 0, 1, 1)
        gb = (0, 1, 1, 1)
        rgb = (1, 1, 1, 1)
        black = (0, 0, 0, 1)
        length = len(self.values)
        pos1 = np.zeros((length-1, 3))
        pos2 = np.zeros((length-1, 3))
        pos3 = np.zeros((length-1, 3))
        pos4 = np.zeros((length-1, 3))
        pos5 = np.zeros((length-1, 3))
        pos6 = np.zeros((length-1, 3))
        pos7 = np.zeros((length-1, 3))
        pos8 = np.zeros((length-1, 3))

        k = 0
        for i in range(0, len(self.values)-1):
            for j in range(0, 8):
                pos_y = self.values[i][j][0]
                pos_x = self.values[i][j][1]
                if j == 0:
                    pos1[k] = (pos_x, pos_y, i)
                elif j == 1:
                    pos2[k] = (pos_x, pos_y, i)
                elif j == 2:
                    pos3[k] = (pos_x, pos_y, i)
                elif j == 3:
                    pos4[k] = (pos_x, pos_y, i)
                elif j == 4:
                    pos5[k] = (pos_x, pos_y, i)
                elif j == 5:
                    pos6[k] = (pos_x, pos_y, i)
                elif j == 6:
                    pos7[k] = (pos_x, pos_y, i)
                else:
                    pos8[k] = (pos_x, pos_y, i)
            k += 1
        l1 = gl.GLLinePlotItem(pos=pos1, width=1.6, color=r)
        l2 = gl.GLLinePlotItem(pos=pos2, width=1.6, color=g)
        l3 = gl.GLLinePlotItem(pos=pos3, width=1.6, color=b)
        l4 = gl.GLLinePlotItem(pos=pos4, width=1.6, color=rg)
        l5 = gl.GLLinePlotItem(pos=pos5, width=1.6, color=gb)
        l6 = gl.GLLinePlotItem(pos=pos6, width=1.6, color=rb)
        l7 = gl.GLLinePlotItem(pos=pos7, width=1.6, color=rgb)
        l8 = gl.GLLinePlotItem(pos=pos8, width=1.6, color=black)
        self.w.addItem(l1)
        self.w.addItem(l2)
        self.w.addItem(l3)
        self.w.addItem(l4)
        self.w.addItem(l5)
        self.w.addItem(l6)
        self.w.addItem(l7)
        self.w.addItem(l8)

    def remove(self):
        print type(self.image)
        self.w.removeItem(self.image)

    def move_image(self):
        import time
        t = time.time()
        try:
            self.w.removeItem(self.image)
        except:
            pass
        #print "Time taken to delete old image: %s" % (time.time() - t)
        t = time.time()

        source_img = self.imm.get_whole_img(self.frame)
        # convert image from uint8 to ubyte
        source_img = source_img.astype("ubyte")

        # convert image to pyqtgraph-readable format and make it partially transparent
        texture = pg.makeRGBA(source_img)[0]
        texture[:,:,3] = 190
        #print "Time taken to get image %s from imm and modify it: %s" % (self.frame, time.time() - t)
        t = time.time()

        # create three image items from textures, add to view
        v1 = gl.GLImageItem(texture)
        v1.translate(0, 0, self.frame)

        self.image = v1
        self.w.addItem(v1)
        #print "Time taken to draw image to plot: %s" % (time.time() - t)

    def add_grids(self):
        x_grid = np.zeros((self.y_size, self.z_size), dtype="ubyte")
        #x_grid.astype()
        x_grid = pg.makeRGBA(x_grid)[0]
        x_grid[:,:,0] = 100
        x_grid[:,:,1] = 100
        x_grid[:,:,2] = 100
        x_grid[:,:,3] = 190

        for i in range(0, self.z_size, self.scale):
            try:
                x_grid[:, i:i+3, 0] = 0
                x_grid[:, i:i+3, 1] = 0
                x_grid[:, i:i+3, 2] = 0
            except:
                pass
        for i in range(0, self.y_size, self.scale):
            try:
                x_grid[i:i+3, :, 0] = 0
                x_grid[i:i+3, :, 1] = 0
                x_grid[i:i+3, :, 2] = 0
            except:
                pass
        # create three image items from textures, add to view
        x_g = gl.GLImageItem(x_grid)
        x_g.translate(0, 0, 0)
        x_g.rotate(90, 0, 0, 1)
        x_g.rotate(90, 0, 1, 0)

        self.w.addItem(x_g)


        y_grid = np.zeros((self.x_size, self.z_size), dtype="ubyte")
        #y_grid.astype()
        y_grid = pg.makeRGBA(y_grid)[0]
        y_grid[:,:,0] = 100
        y_grid[:,:,1] = 100
        y_grid[:,:,2] = 100
        y_grid[:,:,3] = 190

        for i in range(0, self.z_size, self.scale):
            try:
                y_grid[:, i:i+3, 0] = 0
                y_grid[:, i:i+3, 1] = 0
                y_grid[:, i:i+3, 2] = 0
            except:
                pass
        for i in range(0, self.x_size, self.scale):
            try:
                y_grid[i:i+3, :, 0] = 0
                y_grid[i:i+3, :, 1] = 0
                y_grid[i:i+3, :, 2] = 0
            except:
                pass
        # create three image items from textures, add to view
        y_g = gl.GLImageItem(y_grid)
        y_g.translate(0, 0, 0)
        y_g.rotate(90, 1, 0, 0)

        self.w.addItem(y_g)

    def add_axis(self):
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
        pass

    def change_frame(self, value):
        import time
        t = time.time()
        self.frame = value
        self.move_image()
        self.frame_scatter()
        print "Total time taken: %s" % (time.time() - t)
        # refresh text in QLabel
        #self.set_label_text()

    def load_frames(self):
        if not self.is_loading:
            self.is_loading = True
            self.loading_thread.set_frame(self.frame)
            self.loading_thread.start()

    def loading_done(self):
        self.is_loading = False

if __name__ == "__main__":
    import sys
    app = QtGui.QApplication([])
    p = Project()
    p.load('/home/dita/PycharmProjects/eight_22/eight22.fproj')
    w = MyView(p)
    #c = pg.mkColor(0, 0, 0)
    #w.w.setBackgroundColor(c)
    #w.w.setWindowTitle('Pyqtgraph test')
    w.showMaximized()
    w.show()
    app.exec_()
    app.deleteLater()
    sys.exit()