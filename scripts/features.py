from core.project.project import Project
from utils.video_manager import get_auto_video_manager
from matplotlib import pyplot as plt
import cv2
import numpy as np
from core.region.mser import ferda_filtered_msers
import scipy.ndimage as ndimage

def dataset1():
    frames = range(100, 150)

    return frames

def get_curvature_kp(cont, plot=False):
    cont = np.array(cont)

    scale = [20, 14, 10, 7, 3]
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1]]

    if plot:
        plt.figure()
        plt.scatter(cont[:, 0], cont[:, 1], c=[0, 0, 0])
        plt.hold(True)

    for s, c in zip(scale, colors):
        thetas = []
        for i in range(len(cont)):
            p1 = cont[i % len(cont)]
            p2 = cont[i-s % len(cont)]
            p3 = cont[(i+s) % len(cont)]

            a = p1 - p2
            b = p1 - p3

            from math import acos
            try:
            # print np.linalg.norm(a), np.linalg.norm(b)
                x_ = np.dot(a.T, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            except:
                print np.linalg.norm(a), np.linalg.norm(b)
                pass

            # numerical errors fix
            x_ = min(1, max(-1, x_))

            theta = acos(x_)
            theta = abs(np.pi - theta)
            if theta is None:
                print "THETA is None", np.dot(a.T, b), (np.linalg.norm(a) * np.linalg.norm(b))
            thetas.append(theta)

            th = 0.2

        for i in range(20):
            id_ = np.argmax(thetas)

            if thetas[id_] < np.pi/2:
                break

            p = cont[id_]
            th = thetas[id_]
            for j in range(id_-s, id_+s):
                thetas[j % len(thetas)] = 0

            # if th > 3*np.pi/4:
            #     continue

            if plot:
                plt.scatter(p[0], p[1], c=c, s=s**2 )

    if plot:
        plt.hold(False)
        plt.axis('equal')
        plt.gca().invert_yaxis()

        plt.show()
        plt.waitforbuttonpress(0)



if __name__ == '__main__':
    p = Project()
    name = 'Cam1/cam1.fproj'
    wd = '/Users/flipajs/Documents/wd/gt/'
    p.load(wd+name)
    vm = get_auto_video_manager(p)

    plt.ion()

    from core.graph.region_chunk import RegionChunk
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

        # surf = cv2.SURF(1000)
        # kp, des = surf.detectAndCompute(gray,None)
        #
        #
        # pts = np.array([p.pt for p in kp])
        #
        # star = cv2.FeatureDetector_create("STAR")
        # kp = star.detect(gray, None)
        # pts2 = np.array([p.pt for p in kp])
        #
        #
        # corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
        # corners = np.int0(corners)
        # pts3 = np.array([p.ravel() for p in corners])


        # im = cv2.drawKeypoints(im,kp,None,(255,0,0),4)

        # gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        #
        # gray = np.float32(gray)
        # dst = cv2.cornerHarris(gray,2,3,0.04)
        #
        # #result is dilated for marking the corners, not important
        # dst = cv2.dilate(dst,None)
        #
        # # Threshold for an optimal value, it may vary depending on the image.
        # im[dst>0.1*dst.max()]=[0,0,255]

        msers = ferda_filtered_msers(im, p)
        i = 0
        for r in msers:
            if r.area() < 100:
                continue

            cont = r.contour_without_holes()

            kp = get_curvature_kp(cont)

            plt.savefig('/Users/flipajs/Desktop/temp/kp/'+str(frame)+'_'+str(i)+'.png')
            i+=1

        # plt.cla()
        # plt.imshow(im)
        # plt.hold(True)
        # # plt.scatter(pts[:, 0], pts[:, 1], c='b')
        # # plt.scatter(pts2[:, 0], pts2[:, 1], c='r')
        # # plt.scatter(pts3[:, 0], pts3[:, 1], c='g')
        # plt.hold(False)
        # plt.xlim(80, 270)
        # plt.ylim(280, 460)
        # plt.savefig('/Users/flipajs/Desktop/temp/'+str(frame)+'.png')

        # while True:
        #     k = plt.waitforbuttonpress(50)
        #     if k:
        #         break
        #
        # print k

    pass