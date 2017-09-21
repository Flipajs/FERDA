from core.project.project import Project
from core.graph.region_chunk import RegionChunk
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


if __name__ == '__main__':
    p = Project()
    p.load('/Users/flipajs/Documents/wd/FERDA/Cam1_')
    step = 1

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ii = 0
    for t in p.chm.chunk_gen():
        ii += 1
        # if ii > 100:
        #     break

        rch = RegionChunk(t, p.gm, p.rm)

        a = []
        b = []
        c = []

        for i in range(0, t.length(), step):
            r = rch[i]

            a.append(r.area())
            b.append(len(r.contour()))
            c.append(r.ellipse_major_axis_length())
            # areas.append(r.area())
            # major_axes.append(r.ellipse_major_axis_length())
            # minor_axes.append(r.ellipse_minor_axis_length())

        color = 'r'
        if len(t.P) == 1:
            color = 'b'

        ax.scatter(np.mean(a), np.mean(b), np.mean(c), c=color)

    plt.show()
