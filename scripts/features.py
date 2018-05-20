from core.project.project import Project
from utils.video_manager import get_auto_video_manager
from matplotlib import pyplot as plt
import cv2
import numpy as np
from core.region.mser import get_filtered_msers
import scipy.ndimage as ndimage
import warnings
from math import acos
from core.graph.region_chunk import RegionChunk


def dataset1():
    frames = range(100, 150)

    return frames

def get_curvature_kp(cont, plot=False):
    cont = np.array(cont)

    scale = [20, 14, 10, 7, 3]
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1]]

    kps = {}

    if plot:
        plt.figure()
        plt.scatter(cont[:, 0], cont[:, 1], c=[0, 0, 0])
        plt.hold(True)

    for s, c in zip(scale, colors):
        kps[s] = []

        thetas = []
        for i in range(len(cont)):
            p1 = cont[i % len(cont)]
            p2 = cont[i-s % len(cont)]
            p3 = cont[(i+s) % len(cont)]

            a = p1 - p2
            b = p1 - p3

            d_ = (np.linalg.norm(a) * np.linalg.norm(b))
            x_ = 1
            if d_ > 0:
                x_ = np.dot(a.T, b) / (np.linalg.norm(a) * np.linalg.norm(b))

            # numerical errors fix
            x_ = min(1, max(-1, x_))

            theta = acos(x_)
            theta = abs(np.pi - theta)
            thetas.append(theta)

        for i in range(100):
            id_ = np.argmax(thetas)

            if thetas[id_] < np.pi/6:
                break

            p = cont[id_]
            kps[s].append({'point': p, 'angle': thetas[id_]})

            for j in range(id_-int(1.5*s), id_+int(1.5*s)):
                thetas[j % len(thetas)] = 0

            if plot:
                plt.scatter(p[0], p[1], c=c, s=s**2)


    if plot:
        plt.hold(False)
        plt.axis('equal')
        plt.gca().invert_yaxis()

        plt.show()
        # plt.waitforbuttonpress(0)

    return kps


if __name__ == '__main__':
    p = Project()
    name = 'Cam1/cam1.fproj'
    wd = '/Users/flipajs/Documents/wd/gt/'
    p.load(wd+name)
    vm = get_auto_video_manager(p)

    plt.ion()

    for j in range(20):
        rch = RegionChunk(p.chm[j+24], p.gm, p.rm)
        i = 1
        for r in rch:
            cont = r.contour_without_holes()
            kp = get_curvature_kp(cont, True)

            plt.savefig('/Users/flipajs/Desktop/temp/kp/'+str(j)+'_'+str(i)+'.png')
            i+=1

    for frame in dataset1():
        im = vm.get_frame(frame)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

        msers = get_filtered_msers(im, p)
        i = 0
        for r in msers:
            if r.area() < 100:
                continue

            cont = r.contour_without_holes()

            kp = get_curvature_kp(cont, True)

            plt.savefig('/Users/flipajs/Desktop/temp/kp/'+str(frame)+'_'+str(i)+'.png')
            i+=1

