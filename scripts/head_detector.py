import cv2
import cPickle as pickle
from core.graph.region_chunk import RegionChunk
from utils.img import get_safe_selection
from math import ceil
import matplotlib.pyplot as plt
import numpy as np
from scripts.irg_hist_demo import ColorHist3d
from scripts.irg_hist_demo import *
from utils.img_manager import ImgManager
from core.project.project import Project
import math
from ft import FASTex


def get_fastext(project):
    scaleFactor = 1.4
    nlevels = 3
    edgeThreshold = 13
    keypointTypes = 2
    kMin = 9
    kMax = 11
    erode = 0
    segmentGrad = 0
    minCompSize = 4
    process_color = 0
    segmDeltaInt = 1
    min_tupple_top_bottom_angle = math.pi / 2
    maxSpaceHeightRatio = -1
    createKeypointSegmenter = 1

    params_dict = {
        'scaleFactor': scaleFactor,
        'nlevels': nlevels,
        'edgeThreshold': edgeThreshold,
        'keypointTypes': keypointTypes,
        'kMin': kMin,
        'kMax': kMax,
        'erode': erode,
        'segmentGrad': segmentGrad,
        'minCompSize': minCompSize,
        'process_color': process_color,
        'segmDeltaInt': segmDeltaInt,
        'min_tupple_top_bottom_angle': min_tupple_top_bottom_angle,
        'maxSpaceHeightRatio': maxSpaceHeightRatio,
        'createKeypointSegmenter': createKeypointSegmenter
    }

    process_color = 0

    border = 5
    ft = FASTex(edgeThreshold=edgeThreshold,
                createKeypointSegmenter=createKeypointSegmenter,
                nlevels=nlevels,
                minCompSize=minCompSize,
                keypointTypes=keypointTypes)

    return ft


def rotate_img(img, theta):
    s_ = max(img.shape[0], img.shape[1])

    im_ = np.zeros((s_, s_, img.shape[2]), dtype=img.dtype)
    h_ = (s_ - img.shape[0]) / 2
    w_ = (s_ - img.shape[1]) / 2

    im_[h_:h_+img.shape[0], w_:w_+img.shape[1], :] = img

    center = (im_.shape[0] / 2, im_.shape[1] / 2)

    rot_mat = cv2.getRotationMatrix2D(center, -np.rad2deg(theta), 1.0)
    return cv2.warpAffine(im_, rot_mat, (s_, s_))

def centered_crop(img, new_h, new_w):
    new_h = int(new_h)
    new_w = int(new_w)

    h_ = img.shape[0]
    w_ = img.shape[1]

    y_ = (h_ - new_h) / 2
    x_ = (w_ - new_w) / 2

    if y_ < 0 or x_ < 0:
        Warning('cropped area cannot be bigger then original image!')
        return img

    return img[y_:y_+new_h, x_:x_+new_w, :]


def get_bounding_box(r, project, relative_border=1.3, fixed_border=0):
    frame = r.frame()
    img = project.img_manager.get_whole_img(frame)
    roi = r.roi()

    height2 = int(ceil((roi.height() * relative_border) / 2.0) + fixed_border)
    width2 = int(ceil((roi.width() * relative_border) / 2.0) + fixed_border)
    x = r.centroid()[1] - width2
    y = r.centroid()[0] - height2

    bb = get_safe_selection(img, y, x, height2*2, width2*2)

    return bb, np.array([y, x])

def prepare_for_fastext(img, min_size=40):
    border = 0

    d_ = min_size - img.shape[0]
    if d_ < 0:
        border = -d_

    d_ = min_size - img.shape[1]
    if d_ < 0:
        border = max(border, -d_)


    return cv2.copyMakeBorder(img, border, border, border, border, cv2.BORDER_REPLICATE )


def draw_keypoints(ft, keypoints):
    scales = ft.getImageScales()
    scs_ = []
    strokes = []
    for kp, i in zip(keypoints, range(len(keypoints))):
        s_ = ft.getKeypointStrokes(i) * (1.0 / scales[kp[2]])
        scs_.append(kp[2])
        strokes.append(s_)

    sizes_ = [10, 17, 25]

    border = 0
    if False:
        for s_, sc_ in zip(strokes, scs_):
            if sc_ != 2:
                continue

            plt.scatter(s_[:, 0] - border, s_[:, 1] - border, s=int(sizes_[int(sc_)]), c=np.random.rand(3,))

        for s_, sc_ in zip(strokes, scs_):
            if sc_ != 1:
                continue

            plt.scatter(s_[:, 0] - border, s_[:, 1] - border, s=int(sizes_[int(sc_)]), c=np.random.rand(3,))

        for s_, sc_, kp in zip(strokes, scs_, keypoints):
            if sc_ != 0:
                continue

            plt.scatter(s_[:, 0] - border, s_[:, 1] - border, s=int(sizes_[int(sc_)]), c=np.random.rand(3,))

    for s_, sc_, kp in zip(strokes, scs_, keypoints):
        plt.scatter(kp[0], kp[1], s=20)


def endpoint_rot(bb_img, pt, theta, centroid):
    rot = np.array(
        [[math.cos(theta), -math.sin(theta)],
         [math.sin(theta), math.cos(theta)]]
    )

    pt_ = pt-centroid
    pt = np.dot(rot, pt_.reshape(2, 1))
    new_pt = [int(round(pt[0][0] + bb_img.shape[0]/2)), int(round(pt[1][0] + bb_img.shape[1]/2))]

    return new_pt

def detect_head(keypoints, centroid, ep1, ep2, x_bw, x_fw, y_):
    if ep1[1] > ep2[1]:
        ep1, ep2 = ep2, ep1

    score = 0

    for kp in keypoints:
        kpy = kp[1]
        kpx = kp[0]
        if kpy < ep1[0]-y_:
            continue
        if kpy > ep1[0]+y_:
            continue

        if kpx < ep1[1]-x_fw:
            continue
        elif kpx < ep1[1] + x_bw:
            print "+"
            score += 1
            continue

        if kpx < ep2[1]-x_bw:
            continue
        elif kpx < ep2[1]+x_fw:
            print "-"
            score -= 1

    if score > 0:
        return ep1
    elif score < 0:
        return ep2

    return None


def detect_head2(keypoints, ep1, ep2, poly1, poly2):
    score = 0

    for kp in keypoints:
        x = kp[0]
        y = kp[1]
        if poly1.contains_point((y, x)):
            score += 1
        elif poly2.contains_point((y, x)):
            score -= 1

    if score > 0:
        return ep1
    elif score < 0:
        return ep2

    return None


def plot_rectangle(y, x, y2, x2):
    plt.plot([x, x2], [y, y], 'r')
    plt.plot([x, x], [y, y2],  'r')
    plt.plot([x, x2], [y2, y2], 'r')
    plt.plot([x2, x2], [y, y2], 'r')


def create_poly(ep, reversed=False):
    import matplotlib.path as mplPath

    h_ = 25
    h1_ = 20
    h2_ = 30
    h3_ = 5

    sign = -1 if reversed else 1

    x_offset = 5 * sign
    x_bw_ = 10 * sign
    x_fw_ = 17 * sign
    x_fw2_ = 5 * sign
    x_fw3_ = 25 * sign

    x_ = ep[1] + x_offset

    poly = mplPath.Path(np.array([
        [ep[0], x_],
        [ep[0] + h1_, x_ + x_bw_],
        [ep[0] + h_, x_],
        [ep[0] + h2_, x_ - x_fw_],
        [ep[0] + h1_, x_ - x_fw3_],
        [ep[0] + h3_, x_ - x_fw3_],
        [ep[0], x_ - x_fw2_], #
        [ep[0] - h3_, x_ - x_fw3_],
        [ep[0] - h1_, x_ - x_fw3_],
        [ep[0] - h2_, x_ - x_fw_],
        [ep[0] - h_, x_],
        [ep[0] - h1_, x_ + x_bw_],
    ]))

    vs2 = list(poly.vertices)
    vs2.append(poly.vertices[0, :])
    vs2 = vs2[1:]
    for v1, v2 in zip(poly.vertices, vs2):
        plt.plot([v1[1], v2[1]], [v1[0], v2[0]], 'r')

    return poly


def r_normalize(r, shape=(40, 70)):
    bb, offset = get_bounding_box(r, p, relative_border=relative_border, fixed_border=20)

    # p_ = np.array([r.a_*math.sin(-r.theta_), r.a_*math.cos(-r.theta_)])
    # endpoint1 = np.ceil(r.centroid() + p_) + np.array([1, 1])
    # endpoint2 = np.ceil(r.centroid() - p_) - np.array([1, 1])

    bb = rotate_img(bb, r.theta_)
    bb = centered_crop(bb, 6*r.b_, 2.5*r.a_)

    from scipy.misc import imresize

    cv2.imshow("img", bb)
    cv2.waitKey(25)

    bb = np.asarray(imresize(bb, shape), dtype=np.uint8)
    # endpoint1_ = endpoint_rot(bb, endpoint1, -r.theta_, r.centroid())
    # endpoint2_ = endpoint_rot(bb, endpoint2, -r.theta_, r.centroid())

    return bb


def get_sek_img(bb):
    bb_bw = cv2.cvtColor(bb, cv2.COLOR_BGR2GRAY)
    ft.getCharSegmentations(bb_bw, '', 'base')
    keypoints = ft.getLastDetectionKeypoints()

    sek_im = np.zeros((bb.shape[0], bb.shape[1]), dtype=np.uint)

    scales = ft.getImageScales()
    for i, kp in enumerate(keypoints):
        s_ = ft.getKeypointStrokes(i) * (1.0 / scales[kp[2]])
        for pt in s_:
            sek_im[round(pt[1]), round(pt[0])] += 1

    return sek_im


def img2hist(img, hSteps, wSteps):
    img = np.asarray(img, dtype=np.uint8)
    hStep = img.shape[0] / hSteps
    wStep = img.shape[1] / wSteps
    integral = cv2.integral(img)

    hist = np.zeros((hSteps*wSteps, ), dtype=np.uint)

    i = 0
    for y in range(hStep, img.shape[0] + 1, hStep):
        for x in range(wStep, img.shape[1] + 1, wStep):
            sq_sum = integral[y, x] + integral[y-hStep, x-wStep] - integral[y-hStep, x] - integral[y, x-wStep]
            hist[i] = sq_sum
            i += 1

    return hist


def get_description(r, hSteps=4, wSteps=7):
    bb = r_normalize(r)

    laplacian = cv2.Laplacian(bb[:, :, 2], cv2.CV_64F, ksize=3)
    sek_img = get_sek_img(bb)

    sek_imhist = img2hist(sek_img, hSteps, wSteps)
    laplacian_imhist = img2hist(laplacian, hSteps, wSteps)

    # plt.figure(1)
    # plt.subplot(2, 2, 1)
    # plt.imshow(sek_img)
    # plt.subplot(2, 2, 2)
    # plt.imshow(laplacian)
    # plt.subplot(2, 2, 3)
    # plt.plot(sek_imhist)
    # plt.subplot(2, 2, 4)
    # plt.plot(laplacian_imhist)
    # plt.show()
    # key = cv2.waitKey(0)

    # key = plt.waitforbuttonpress(5)



if __name__ == "__main__":
    p = Project()
    p.load('/Users/flipajs/Documents/wd/GT/Cam1/cam1.fproj')
    p.img_manager = ImgManager(p)

    ft = get_fastext(p)

    sample_step = 1
    relative_border = 5.0

    plt.ion()

    x_bw = 8
    x_fw = 20
    y_ = 25

    file_name = "../data/head_detection/data.pkl"
    with open(file_name, "rb") as f:
        data = pickle.load(f)


    skip_ids = set()
    for id_, b in data:
        print id_, b
        skip_ids.add(id_)

    # fig = plt.figure()
    # ax = fig.add_subplot(111)

    for v_id in p.gm.get_all_relevant_vertices():
        ch_id = p.gm.g.vp['chunk_start_id'][p.gm.g.vertex(v_id)]
        if ch_id > 0:
            print ch_id
            ch = p.chm[ch_id]
            r_ch = RegionChunk(ch, p.gm, p.rm)

            from_one_chunk = 10
            i = 0
            for t in range(r_ch.start_frame(), r_ch.end_frame() + 1, sample_step):
                r = r_ch[t - r_ch.start_frame()]
                if r.is_virtual:
                    continue

                if r.id_ in skip_ids:
                    break

                get_description(r)
                var = raw_input("head is Right Left? (r/l) s for skip")
                if var == 's':
                    break

                data.append((r.id_, True if var == 'r' else False))
                print "LEN: ", len(data)

                i += 1

                if i == from_one_chunk:
                    break

                with open(file_name, 'wb') as f:
                    pickle.dump(data, f)

                # bb, endpoint1_, endpoint2_ = r_normalize(r)

                # laplacian = cv2.Laplacian(bb, cv2.CV_64F, ksize=3)

                # plt.figure()
                # plt.imshow(bb)

                # if endpoint1_[1] > endpoint2_[1]:
                #     endpoint1_, endpoint2_ = endpoint2_, endpoint1_
                #
                # plt.clf()
                #
                # poly1 = create_poly(endpoint1_)
                # poly2 = create_poly(endpoint2_, True)

                # plot_rectangle(endpoint1_[0]-y_, endpoint1_[1]-x_fw, endpoint1_[0]+y_, endpoint1_[1]+x_bw)
                # plot_rectangle(endpoint1_[0]-y_, endpoint2_[1]-x_bw, endpoint1_[0]+y_, endpoint2_[1]+x_fw)

                # bb_bw = cv2.cvtColor(bb, cv2.COLOR_BGR2GRAY)
                # ft.getCharSegmentations(bb_bw, '', 'base')
                #
                # keypoints = ft.getLastDetectionKeypoints()
                # draw_keypoints(ft, keypoints)
                #
                # # head = detect_head(keypoints, np.array([bb.shape[0] / 2, bb.shape[1] / 2]), endpoint1_, endpoint2_, x_bw, x_fw, y_)
                # # head = detect_head2(keypoints, endpoint1_, endpoint2_, poly1, poly2)
                #
                # # if head:
                # #     plt.scatter(head[1], head[0], s=100, c=(1, 1,0))
                #
                # plt.show()
                # plt.waitforbuttonpress()

