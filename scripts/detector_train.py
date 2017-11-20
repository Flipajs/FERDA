import sys
import cPickle as pickle
from core.settings import Settings as S_
from utils.misc import is_flipajs_pc, is_matejs_pc
import time
from core.project.project import Project
import numpy as np
import matplotlib.pylab as plt
import montage
import math
import os.path
from core.graph.region_chunk import RegionChunk
from core.region.region import Region
from utils.video_manager import get_auto_video_manager
from utils.img import get_img_around_pts
from utils.drawing.points import get_roi
from core.region.transformableregion import TransformableRegion
from core.region.region import get_region_endpoints, get_orientation
import cv2
import fire
import tqdm
import copy
# from joblib import Parallel, delayed
import csv
from matplotlib.patches import Ellipse
import waitforbuttonpress


def p2e(projective):
    """
    Convert 2d or 3d projective to euclidean coordinates.

    :param projective: projective coordinate(s)
    :type projective: numpy.ndarray, shape=(3 or 4, n)

    :return: euclidean coordinate(s)
    :rtype: numpy.ndarray, shape=(2 or 3, n)
    """
    assert(type(projective) == np.ndarray)
    assert((projective.shape[0] == 4) | (projective.shape[0] == 3))
    return (projective / projective[-1, :])[0:-1, :]


def e2p(euclidean):
    """
    Convert 2d or 3d euclidean to projective coordinates.

    :param euclidean: projective coordinate(s)
    :type euclidean: numpy.ndarray, shape=(2 or 3, n)

    :return: projective coordinate(s)
    :rtype: numpy.ndarray, shape=(3 or 4, n)
    """
    assert(type(euclidean) == np.ndarray)
    assert((euclidean.shape[0] == 3) | (euclidean.shape[0] == 2))
    return np.vstack((euclidean, np.ones((1, euclidean.shape[1]))))


def column(vector):
    """
    Return column vector.

    :param vector: np.ndarray
    :return: column vector
    :rtype: np.ndarray, shape=(n, 1)
    """
    return vector.reshape((-1, 1))


def head_fix(tracklet_regions):
    import heapq
    q = []
    # q = Queue.PriorityQueue()
    heapq.heappush(q, (0, [False]))
    heapq.heappush(q, (0, [True]))
    # q.put((0, [False]))
    # q.put((0, [True]))

    result = []
    i = 0
    max_i = 0

    cut_diff = 10

    while True:
        i += 1

        # cost, state = q.get()
        cost, state = heapq.heappop(q)
        if len(state) > max_i:
            max_i = len(state)

        if len(state) + cut_diff < max_i:
            continue

        # print i, cost, len(state), max_i

        if len(state) == len(tracklet_regions):
            result = state
            break

        prev_r = tracklet_regions[len(state) - 1]
        r = tracklet_regions[len(state)]

        prev_c = prev_r.centroid()
        p1, p2 = get_region_endpoints(r)

        dist = np.linalg.norm
        d1 = dist(p1 - prev_c)
        d2 = dist(p2 - prev_c)

        prev_head, prev_tail = get_region_endpoints(prev_r)
        if state[-1]:
            prev_head, prev_tail = prev_tail, prev_head

        d3 = dist(p1 - prev_head) + dist(p2 - prev_tail)
        d4 = dist(p1 - prev_tail) + dist(p2 - prev_head)

        # state = list(state)
        state2 = list(state)
        state.append(False)
        state2.append(True)

        new_cost1 = d3
        new_cost2 = d4

        # TODO: param
        if dist(prev_c - r.centroid()) > 5:
            new_cost1 += d2 - d1
            new_cost2 += d1 - d2

        heapq.heappush(q, (cost + new_cost1, state))
        heapq.heappush(q, (cost + new_cost2, state2))
        # q.put((cost + new_cost1, state))
        # q.put((cost + new_cost2, state2))

    for b, r in zip(result, tracklet_regions):
        if b:
            r.theta_ += np.pi
            if r.theta_ >= 2 * np.pi:
                r.theta_ -= 2 * np.pi


class GenerateTrainingData(object):
    def __init__(self):
        self.video = None
        self.single = None
        self.multi = None
        self.project = None
        self.bg = None
        self.__i__ = 0

    def __load_project__(self):
        self.project = Project()
        # This is development speed up process (kind of fast start). Runs only on developers machines...
        # if is_flipajs_pc() and False:
        wd = None
        if is_flipajs_pc():
            # wd = '/Users/iflipajs/Documents/wd/FERDA/Cam1_rf'
            # wd = '/Users/flipajs/Documents/wd/FERDA/Cam1_playground'
            # wd = '/Users/flipajs/Documents/wd/FERDA/test6'
            # wd = '/Users/flipajs/Documents/wd/FERDA/zebrafish_playground'
            # wd = '/Users/flipajs/Documents/wd/FERDA/Camera3'
            # wd = '/Users/flipajs/Documents/wd/FERDA/Cam1_rfs2'
            # wd = '/Users/flipajs/Documents/wd/FERDA/Cam1'
            wd = '/Users/flipajs/Documents/wd/FERDA/rep1-cam2'
            # wd = '/Users/flipajs/Documents/wd/FERDA/rep1-cam3'

            # wd = '/Users/flipajs/Documents/wd/FERDA/Sowbug3'

            # wd = '/Users/flipajs/Documents/wd/FERDA/test'
        if is_matejs_pc():
            # wd = '/home/matej/prace/ferda/10-15/'
            wd = '/home/matej/prace/ferda/10-15 (copy)/'
        if wd is not None:
            self.project.load(wd)

        # img = video.get_frame(region.frame())  # ndarray bgr
        # img_region = get_img_around_pts(img, region.pts(), margin=0)
        # cv2.imshow('test', img_region[0])
        # cv2.waitKey()

        self.video = get_auto_video_manager(self.project)
        print('regions start')

        regions_filename = './out/regions_long_tracklets.pkl'
        if os.path.exists(regions_filename):
            print('regions loading...')
            with open(regions_filename, 'rb') as fr:
                self.single = pickle.load(fr)
                self.multi = pickle.load(fr)
            print('regions loaded')
        else:
            from collections import defaultdict

            self.single = defaultdict(list)
            self.multi = defaultdict(list)
            # long_moving_tracklets = []

            for tracklet in self.project.chm.chunk_gen():
                if tracklet.is_single() or tracklet.is_multi():
                    region_tracklet = RegionChunk(tracklet, self.project.gm, self.project.rm)
                    if tracklet.is_single():
                        centroids = np.array([region_tracklet.centroid_in_t(frame) for frame
                                              in
                                              range(region_tracklet.start_frame(),
                                                    region_tracklet.end_frame())])  # shape=(n,2)

                        if len(tracklet) > 20 and np.linalg.norm(np.diff(centroids, axis=0), axis=1).mean() > 1.5:
                            # long_moving_tracklets.append(tracklet)

                            regions = list(region_tracklet)
                            head_fix(regions)
                            # images = []
                            # for region in regions:
                            #     img = video.get_frame(region.frame())
                            #     head, tail = get_region_endpoints(region)
                            #     img = cv2.drawMarker(img, tuple(head[::-1].astype(int)), (0, 0, 255))
                            #     img = cv2.drawMarker(img, tuple(tail[::-1].astype(int)), (0, 255, 0))
                            #     border = 10
                            #     roi = region.roi()
                            #     images.append(img[roi.y() - border:roi.y() + roi.height() + border,
                            #                   roi.x() - border:roi.x() + roi.width() + border])
                            # montage_generator = montage.Montage((1000, 500), (10, 5))
                            # plt.imshow(montage_generator.montage(images[:50])[::-1])
                            # plt.waitforbuttonpress()
                            for region in regions:
                                self.single[region.frame()].append(region)
                        else:
                            # short single tracklets are ignored
                            pass
                    else:  # multi
                        for region in region_tracklet.regions_gen():
                            # if tracklet.is_single():
                            #     single[region.frame()].append(region)
                            # else:
                            self.multi[region.frame()].append(region)

            with open(regions_filename, 'wb') as fw:
                pickle.dump(self.single, fw)
                pickle.dump(self.multi, fw)

    def __get_out_dir_rel__(self, out_dir, out_file):
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        return os.path.relpath(os.path.abspath(out_dir), os.path.abspath(os.path.dirname(out_file)))

    def composebackground(self, video_step=100, dilation_size=100, out_filename='bg.png'):
        """
        Generates background without foreground objects.

        :param video_step: search step in video
        :param dilation_size: dilate foreground objects
        :param out_filename:
        :return:
        """
        self.__load_project__()
        # background composition
        img = self.video.get_frame(0)

        multiple_imgs = []  # np.zeros((median_len,) + img.shape)
        bg_mask = np.zeros(img.shape[:2], dtype=np.bool)
        i = 0
        print 'frame number\tmissing bg pixels'
        while not np.all(bg_mask):
            frame = i * video_step
            print frame, '\t', np.count_nonzero(~bg_mask)
            img = self.video.get_frame(frame).astype(float)
            img_bg_mask = np.zeros(img.shape[:2], dtype=np.bool)
            for region in self.single[frame] + self.multi[frame]:
                transformable_region = TransformableRegion(img)
                transformable_region.set_region(region)
                mask = transformable_region.get_mask(dilation_size)
                img[mask] = np.nan
                img_bg_mask = np.bitwise_or(img_bg_mask, mask)
            bg_mask = np.bitwise_or(bg_mask, ~img_bg_mask)
            multiple_imgs.append(img)
            i += 1

        median_img = np.nanmedian(np.stack(multiple_imgs), 0).astype(np.uint8)
        cv2.imwrite(out_filename, median_img)

    def writeforeground(self, out_dir='fg', out_file='fg.txt'):
        """
        Write all foreground objects as images and list them into a text file for training.

        :param out_dir:
        :param out_file:
        :return:
        """
        self.__load_project__()
        # write fg images and txt file
        out_dir_rel = self.__get_out_dir_rel__(out_dir, out_file)

        with open(out_file, 'w') as fw:
            for frame in tqdm.tqdm(sorted(self.single.keys())):
                img = self.video.get_frame(frame)
                filename = '%05d.jpg' % frame
                cv2.imwrite(os.path.join(out_dir, filename), img)
                line = os.path.join(out_dir_rel, filename) + ' ' + str(len(self.single[frame])) + ' '
                for region in self.single[frame]:
                    roi = region.roi()
                    line += '%d %d %d %d ' % (roi.x(), roi.y(), roi.width(), roi.height())
                line += '\n'
                fw.write(line)

    def meanimage(self, out_filename='mean_foreground.png', width=70, height=100, num_regions=80):
        """
        Creates mean "single" foreground object.

        :param out_filename:
        :param width:
        :param height:
        :param num_regions:
        :return:
        """
        self.__load_project__()
        frames_regions = []
        for _ in range(num_regions):
            region = np.random.choice(self.single[np.random.choice(self.single.keys())])
            frames_regions.append((region.frame(), region))
        frames_regions = sorted(frames_regions, key=lambda fr: fr[0])

        # single_non_empty = {f: s for f, s in self.single.iteritems() if s != []}
        #
        # for i in range(num_regions):
        #     region = np.random.choice(single_non_empty[single_non_empty.keys()[i]])
        #     frames_regions.append((region.frame(), region))

        mean_img = np.zeros((height, width, 3))
        for frame, region in tqdm.tqdm(frames_regions):
            # img = cv2.cvtColor(self.video.get_frame(region.frame()), cv2.COLOR_BGR2GRAY)
            tregion = TransformableRegion(self.video.get_frame(frame))
            tregion.rotate(-math.degrees(region.theta_) + 90, rotation_center_yx=region.centroid())
            img_aligned = tregion.get_img()
            hw = np.array((height, width), dtype=float)
            tlbr = np.hstack((region.centroid() - hw / 2, region.centroid() + hw / 2)).round().astype(int)
            try:
                mean_img += img_aligned[tlbr[0]:tlbr[2], tlbr[1]:tlbr[3], :]
            except ValueError:
                # out of image bounds
                num_regions -= 1

        cv2.imwrite(out_filename, (mean_img / float(num_regions)).astype(np.uint8))

    def writebackgroundaligned(self, out_dir='bg', out_file='bg.txt', bg_angle_max_deviation_deg=20):
        """
        Write backgrounds for training an aligned object detector: the
        backgrounds include non-vertical foreground objects.

        :param out_dir:
        :param out_file:
        :param bg_angle_max_deviation_deg:
        :return:
        """
        self.__load_project__()
        out_dir_rel = self.__get_out_dir_rel__(out_dir, out_file)
        images = []
        i = 0
        bb_fixed_border_xy = (20, 10)

        fw = open(out_file, 'w')

        for frame in tqdm.tqdm(sorted(self.single.keys())):  # [::10]:
            img = self.video.get_frame(frame)

            for region in self.single[frame]:  # [::10]:

                if not (-bg_angle_max_deviation_deg < math.degrees(region.theta_) - 90 < bg_angle_max_deviation_deg):
                    filename = '%05d.png' % i
                    roi = region.roi()
                    crop = img[roi.y(): roi.y() + roi.height() + 1,
                               roi.x(): roi.x() + roi.width() + 1]
                    cv2.imwrite(os.path.join(out_dir, filename), crop)
                    fw.write(os.path.join(out_dir_rel, filename) + '\n')
                    # plt.imshow(crop)
                    # plt.waitforbuttonpress()


                # tcorners = np.int32(np.dot(c
                #  rot, img.shape[:2])[roi.y():roi.y() + roi.height(), roi.x():roi.x() + roi.width()])
                #                             #  (roi.height() + 10, roi.width() + 10)))
                # size = crop.shape[:2]
                # half = (size / 2.)[::-1]
                # crop = cv2.arrowedLine(crop, half, half + min(half) * []
                # head, tail = get_region_endpoints(region)
                # img = cv2.drawMarker(img, tuple(head[::-1].astype(int)), (0, 0, 255))
                # img = cv2.drawMarker(img, tuple(tail[::-1].astype(int)), (0, 255, 0))
                # border = 10
                # images.append(img[roi.y() - border:roi.y() + roi.height() + border,
                #                   roi.x() - border:roi.x() + roi.width() + border])
                i += 1
                # if i > 200:
                #     plt.imshow(montage.montage(images[:200])[::-1])
                #     plt.waitforbuttonpress()
                #     i = 0
                #     images = []

        fw.close()

    def writeforegroundaligned(self, out_dir='fg', out_file='fg.txt', fixed_size=True):
        """
        Write foreground objects for training an aligned objects detector. The foregrounds include
        only vertical objects.

        :param out_dir:
        :param out_file:
        :param fixed_size:
        :return:
        """
        self.__load_project__()
        out_dir_rel = self.__get_out_dir_rel__(out_dir, out_file)
        images = []
        i = 0
        bb_fixed_border_xy = (20, 10)

        if fixed_size:
            import itertools

            all_regions = list(itertools.chain(*self.single.values()))
            major_axes = np.array([r.a_ for r in all_regions]) * 2
            bb_major_px = np.median(major_axes)
            minor_axes = np.array([r.b_ for r in all_regions]) * 2
            bb_minor_px = np.median(minor_axes)
            bb_size = (round(bb_minor_px + 2 * bb_fixed_border_xy[0]),
                       round(bb_major_px + 2 * bb_fixed_border_xy[1]))

        fw = open(out_file, 'w')
        # with open('./out/regions.pkl', 'wb') as fw:
        #     frames = sorted(single.keys())
        #     i = 0
        #     r1 = single[frames[i]][0]
        #     img1 = video.get_frame(frames[i])
        #     i = 1
        #     r2 = single[frames[i]][0]
        #     img2 = video.get_frame(frames[i])
        #     pickle.dump(r1, fw)
        #     pickle.dump(img1, fw)
        #     pickle.dump(r2, fw)
        #     pickle.dump(img2, fw)

        for frame in tqdm.tqdm(sorted(self.single.keys())):  # [::10]:
            img = self.video.get_frame(frame)

            for region in self.single[frame]:  # [::10]:

                centroid_crop = tuple(region.centroid()[::-1] - cropxy)
                rot = cv2.getRotationMatrix2D(centroid_crop,
                                              -math.degrees(region.theta_) + 90, 1.)
                crop_rot = cv2.warpAffine(crop, rot, crop.shape[:2][::-1])

                if not fixed_size:
                    head, tail = np.array(get_region_endpoints(region))
                    head_tail_xy = np.vstack((head[::-1] - cropxy, tail[::-1] - cropxy))
                    head_rotated, tail_rotated = rot.dot(np.hstack((head_tail_xy, np.ones((2, 1)))).T).T.astype(int)
                    border_xy = (30, 10)
                    img_aligned = crop_rot[max(head_rotated[1] - border_xy[1], 0):
                    min(tail_rotated[1] + border_xy[1], crop_rot.shape[0]),
                                  max(head_rotated[0] - border_xy[0], 0):
                                  min(tail_rotated[0] + border_xy[0], crop_rot.shape[1])]
                else:
                    img_aligned = crop_rot[max(int(round(centroid_crop[1] - bb_size[1] / 2)), 0):
                    min(int(round(centroid_crop[1] + bb_size[1] / 2)), crop_rot.shape[0]),
                                  max(int(round(centroid_crop[0] - bb_size[0] / 2)), 0):
                                  min(int(round(centroid_crop[0] + bb_size[0] / 2)), crop_rot.shape[1])]

                # [tl, tr, br, bl]
                # corners = np.array([[roi.x() - border, roi.y() - border],
                #            [roi.x() + roi.width() + border, roi.y() - border],
                #            [roi.x() + roi.width() + border, roi.y() + roi.height() + border],
                #            [roi.x() - border, roi.y() + roi.height() + border]])
                # corners_rotated = rot.dot(np.hstack((corners, np.ones((4,1)))).T).T
                # x, y, w, h = cv2.boundingRect(corners_rotated.astype(int).reshape(1, -1, 2))
                # img = cv2.warpAffine(img, rot, img.shape[:2])
                # images.append(img[y:y+h, x:x+w])
                # plt.imshow(crop_rot)
                # plt.imshow(crop_rot[head_rotated[1] - border_xy[1]:tail_rotated[1] + border_xy[1],
                # head_rotated[0] - border_xy[0]:tail_rotated[0] + border_xy[0]])

                # images.append()
                filename = '%05d.png' % i
                cv2.imwrite(os.path.join(out_dir, filename), img_aligned)
                line = os.path.join(out_dir_rel, filename) + ' 1 0 0 %d %d\n' % (img_aligned.shape[1], img_aligned.shape[0])
                fw.write(line)

                # tcorners = np.int32(np.dot(corners, rot.T))
                # x, y, w, h = cv2.boundingRect(tcorners.reshape(1, -1, 2))
                # images.append(cv2.warpAffine(img, rot, img.shape[:2])[roi.y():roi.y() + roi.height(), roi.x():roi.x() + roi.width()])
                #                             #  (roi.height() + 10, roi.width() + 10)))
                # size = crop.shape[:2]
                # half = (size / 2.)[::-1]
                # crop = cv2.arrowedLine(crop, half, half + min(half) * []
                # head, tail = get_region_endpoints(region)
                # img = cv2.drawMarker(img, tuple(head[::-1].astype(int)), (0, 0, 255))
                # img = cv2.drawMarker(img, tuple(tail[::-1].astype(int)), (0, 255, 0))
                # border = 10
                # images.append(img[roi.y() - border:roi.y() + roi.height() + border,
                #                   roi.x() - border:roi.x() + roi.width() + border])
                i += 1
                # if i > 200:
                #     plt.imshow(montage.montage(images[:200])[::-1])
                #     plt.waitforbuttonpress()
                #     i = 0
                #     images = []

        fw.close()

    def test(self):
        with open('./out/regions.pkl', 'rb') as fr:
            r1 = pickle.load(fr)
            img1 = pickle.load(fr)
            r2 = pickle.load(fr)
            img2 = pickle.load(fr)

        region1 = TransformableRegion(img1)
        region1.set_region(r1)
        region2 = TransformableRegion(img2)
        region2.set_region(r2)
        region2.use_background = False
        plt.imshow(region1.compose(region2.rotate(-30).move((-15, -15))))
        # plt.imshow(region1.move((0, 100)).get_mask())

    def synthetize_double_regions(self, count=100, out_dir='./out', out_csv='./out/doubleregions.csv'):
        # angles: positive clockwise, zero direction to right
        self.__load_project__()
        from core.bg_model.median_intensity import MedianIntensity
        self.bg = MedianIntensity(self.project)
        self.bg.compute_model()

        single_regions = [item for sublist in self.single.values() for item in sublist]
        BATCH_SIZE = 250

        with open(out_csv, 'w') as csv_file:
            fieldnames = ['filename', 'ant1_x', 'ant1_y', 'ant1_major', 'ant1_minor', 'ant1_angle_deg',
                                      'ant2_x', 'ant2_y', 'ant2_major', 'ant2_minor', 'ant2_angle_deg',
                                      'ant1_id', 'ant2_id', 'theta_rad', 'phi_rad', 'overlap_px', 'video_file']
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writeheader()

            i = 0
            for i1 in tqdm.tqdm(np.arange(0, count, BATCH_SIZE), desc='batch'):
                i2 = min(i1 + BATCH_SIZE, count)
                n = i2 - i1
                regions = np.random.choice(single_regions, 2 * n)
                frames = [r.frame() for r in regions]
                sort_idx = np.argsort(frames)
                sort_idx_reverse = np.argsort(sort_idx)
                images_sorted = [self.video.get_frame(r.frame()) for r in tqdm.tqdm(regions[sort_idx],
                                                                                    desc='images reading')]
                images = [images_sorted[idx] for idx in sort_idx_reverse]

                with tqdm.tqdm(total=n, desc='synthetize') as progress_bar:
                    for region1, region2, img1, img2 in zip(regions[:n], regions[n:], images[:n], images[n:]):
                        img_filename = '%06d.jpg' % i
                        img = None
                        while True:
                            # border point angle with respect to object centroid, 0 rad is from the centroid rightwards, positive ccw
                            theta_rad = np.random.uniform(-math.pi, math.pi)
                            # approach angle, 0 rad is direction from the object centroid
                            phi_rad = np.clip(np.random.normal(scale=(math.pi / 2) / 2), math.radians(-80), math.radians(80))
                            overlap_px = int(round(np.random.gamma(1, 5)))

                            try:
                                img, ant1, ant2 = self.synthetize(region1, region2,
                                                                  theta_rad, phi_rad, overlap_px,
                                                                  img1, img2)
                            except IndexError:
                                print('%s: IndexError, repeating' % img_filename)
                            if img is not None:
                                break

                        cv2.imwrite(os.path.join(out_dir, img_filename), img)
                        csv_writer.writerow({
                            'filename': img_filename,
                            'ant1_x': round(ant1['xy'][0], 1),
                            'ant1_y': round(ant1['xy'][1], 1),
                            'ant1_major': round(ant1['major'], 1),
                            'ant1_minor': round(ant1['minor'], 1),
                            'ant1_angle_deg': round(ant1['angle_deg'], 1),
                            'ant2_x': round(ant2['xy'][0], 1),
                            'ant2_y': round(ant2['xy'][1], 1),
                            'ant2_major': round(ant2['major'], 1),
                            'ant2_minor': round(ant2['minor'], 1),
                            'ant2_angle_deg': round(ant2['angle_deg'], 1),
                            'ant1_id': region1.id(),
                            'ant2_id': region2.id(),
                            'theta_rad': round(theta_rad, 1),
                            'phi_rad': round(phi_rad, 1),
                            'overlap_px': round(overlap_px, 1),
                            'video_file': os.path.basename(self.video.video_path),
                        })
                        i += 1
                        progress_bar.update()
                progress_bar.close()


        # # montage bounding box
        # plt.imshow(img_synthetic_upright)
        # x1, x2 = np.nonzero(mask_double.sum(axis=0))[0][[0, -1]]
        # y1, y2 = np.nonzero(mask_double.sum(axis=1))[0][[0, -1]]
        # bb_xywh = [x1, y1, x2 - x1 + 1, y2 - y1 + 1]
        #
        # ax = plt.gca()
        # from matplotlib.patches import Rectangle
        # ax.add_patch(Rectangle(bb_xywh[:2], bb_xywh[2], bb_xywh[3], linewidth=1, edgecolor='r', facecolor='none'))

    def show_double_regions_csv(self, csv_file, image_dir):
        # waitforbuttonpress.figure()
        with open(csv_file, 'r') as fr:
            csv_reader = csv.DictReader(fr)
            for i, row in enumerate(csv_reader):
                img = plt.imread(os.path.join(image_dir, row['filename']))
                fig = plt.figure()
                plt.imshow(img)
                ax = plt.gca()
                ax.add_patch(Ellipse(xy=(float(row['ant1_x']), float(row['ant1_y'])),
                                     width=float(row['ant1_major']),
                                     height=float(row['ant1_minor']),
                                     angle=-float(row['ant1_angle_deg']),
                                     edgecolor='r',
                                     facecolor='none'))
                ax.add_patch(Ellipse(xy=(float(row['ant2_x']), float(row['ant2_y'])),
                                     width=float(row['ant2_major']),
                                     height=float(row['ant2_minor']),
                                     angle=-float(row['ant2_angle_deg']),
                                     edgecolor='r',
                                     facecolor='none'))
                # plt.draw()
                fig.savefig('out/debug/%03d.png' % i, transparent=True, bbox_inches='tight', pad_inches=0)
                plt.close(fig)
                # waitforbuttonpress.wait()
                # if waitforbuttonpress.is_closed():
                #     break
                plt.clf()

    def synthetize(self, region1, region2, theta_rad, phi_rad, overlap_px, img1=None, img2=None):
        # angles: positive clockwise, zero direction to right

        if img1 is None:
            img1 = self.video.get_frame(region1.frame())
        tregion1 = TransformableRegion(img1)
        tregion1.set_region(region1)

        border_point_xy = region1.get_border_point(theta_rad, shift_px=-overlap_px)

        # # mask based on background subtraction
        # fg_mask = bg.get_fg_mask(img)
        # fg_mask = cv2.dilate(fg_mask.astype(np.uint8), kernel=np.ones((3, 3), np.uint8), iterations=1)
        # mask_labels = cv2.connectedComponents(fg_mask)[1]  # .astype(np.uint8)
        # center_xy_rounded = center_xy.round().astype(int)
        # mask = (mask_labels == mask_labels[center_xy_rounded[1], center_xy_rounded[0]]).astype(np.uint8)
        # region.set_mask(mask)

        if img2 is None:
            img2 = self.video.get_frame(region2.frame())
        head_yx, tail_yx = get_region_endpoints(region2)

        ##

        # constructing img2 alpha channel
        bg_diff = (self.bg.bg_model.astype(np.float) - img2).mean(axis=2).clip(5, 100)
        alpha = ((bg_diff - bg_diff.min()) / np.ptp(bg_diff))
        # plt.imshow(alpha)
        # plt.jet()
        # plt.colorbar()

        ##

        img2_rgba = np.concatenate((img2, np.expand_dims(alpha * 255, 2).astype(np.uint8)), 2)
        tregion2 = TransformableRegion(img2_rgba)
        tregion2.set_region(region2)

        # # background subtraction: component analysis
        # fg_mask2 = bg.get_fg_mask(img2)
        # # fg_mask2 = cv2.dilate(fg_mask2.astype(np.uint8), kernel=np.ones((3, 3), np.uint8), iterations=1)
        # fg_mask2 = fg_mask2.astype(np.uint8)
        # _, labels, stats, centroids = cv2.connectedComponentsWithStats(fg_mask2)  # .astype(np.uint8)
        # center_xy2 = single2.centroid()[::-1]
        # center_xy_rounded2 = center_xy2.round().astype(int)
        # component_id = labels[center_xy_rounded2[1], center_xy_rounded2[0]]
        # bb_xywh = stats[component_id][:4]
        # from matplotlib.patches import Rectangle
        # # fig, ax = plt.subplots()
        # # ax = plt.gca()
        # mask2 = (labels == component_id).astype(np.uint8)
        # ax.imshow(mask2)
        # ax.add_patch(Rectangle(bb_xywh[:2], bb_xywh[2], bb_xywh[3], linewidth=1, edgecolor='r', facecolor='none'))

        tregion2.set_elliptic_mask()

        tregion2.use_background = False
        region2_main_axis_angle_rad = -region2.theta_ + math.pi - (phi_rad + theta_rad)
        tregion2.move(-head_yx).rotate(math.degrees(region2_main_axis_angle_rad)).move(border_point_xy[::-1])

        img2_rgba_trans = tregion2.get_img()
        alpha_trans = img2_rgba_trans[:, :, 3]
        mask_trans = tregion2.get_mask()
        alpha_trans[mask_trans == 0] = 0  # apply mask
        alpha_trans = alpha_trans.astype(float) / 255

        # plt.figure()
        # plt.imshow(img2)
        # plt.plot(*single2.centroid()[::-1], marker='+')

        # plt.figure()
        # plt.imshow(img2_rgba_trans[:,:, :3])
        # plt.plot(*xy, marker='+')
        from matplotlib.patches import Ellipse

        img_synthetic = (img1.astype(float) * (1 - np.expand_dims(alpha_trans, 2)) +
                         img2_rgba_trans[:, :, :3].astype(float) * np.expand_dims(alpha_trans, 2)).astype(np.uint8)

        synth_center_xy1 = region1.centroid()[::-1].astype(int)

        # plt.imshow(img_synthetic)
        # ax = plt.gca()
        # zoomed_size = 300
        # _ = plt.axis([synth_center_xy1[0] - zoomed_size / 2, synth_center_xy1[0] + zoomed_size / 2,
        #               synth_center_xy1[1] - zoomed_size / 2, synth_center_xy1[1] + zoomed_size / 2])

        ##

        # ax.add_patch(Ellipse(xy=synth_center_xy1,
        #                      width=4 * single.major_axis_,
        #                      height=4 * single.minor_axis_,
        #                      angle=-math.degrees(single.theta_),
        #                      edgecolor='r',
        #                      facecolor='none'))
        #
        synth_center_xy2 = tregion2.get_transformed_coords(region2.centroid()[::-1])
        # ax.add_patch(Ellipse(xy=synth_center_xy2,
        #                      width=4 * single2.major_axis_,
        #                      height=4 * single2.minor_axis_,
        #                      angle=-math.degrees(single2.theta_ + region2_main_axis_angle_rad),
        #                      edgecolor='r',
        #                      facecolor='none'))

        ##

        tregion1.set_elliptic_mask()
        mask = np.logical_or(tregion1.get_mask().astype(bool), tregion2.get_mask().astype(np.bool))

        # plt.figure()
        # # plt.imshow(mask)
        # zoomed_size = 300
        # _ = plt.axis([center_xy[0] - zoomed_size / 2, center_xy[0] + zoomed_size / 2,
        #               center_xy[1] - zoomed_size / 2, center_xy[1] + zoomed_size / 2])

        moments = cv2.moments(mask.astype(np.uint8), True)
        centroid_xy = (moments['m10'] / moments['m00'], moments['m01'] / moments['m00'])
        moments['muprime20'] = moments['mu20'] / moments['m00']
        moments['muprime02'] = moments['mu02'] / moments['m00']
        moments['muprime11'] = moments['mu11'] / moments['m00']
        theta = 0.5 * math.atan2(2 * moments['muprime11'],
                                 (moments['muprime20'] - moments['muprime02']))

        tregion_synthetic = TransformableRegion(img_synthetic)
        # tregion_synthetic.set_region(single)
        tregion_synthetic.set_mask(mask.astype(np.uint8))
        tregion_synthetic.rotate(-math.degrees(theta), centroid_xy[::-1])
        img_synthetic_upright = tregion_synthetic.get_img()
        # mask_synthetic_upright = tregion_synthetic.get_mask()

        ##
        ant1 = {'xy': tregion_synthetic.get_transformed_coords(synth_center_xy1),
                'major': 4 * region1.major_axis_,
                'minor': 4 * region1.minor_axis_,
                'angle_deg': tregion_synthetic.get_transformed_angle(math.degrees(region1.theta_))}
        ant2 = {'xy': tregion_synthetic.get_transformed_coords(synth_center_xy2),
                'major': 4 * region2.major_axis_,
                'minor': 4 * region2.minor_axis_,
                'angle_deg': tregion_synthetic.get_transformed_angle(
                    math.degrees(region2.theta_ + region2_main_axis_angle_rad))}

        # fig = plt.figure()
        # plt.imshow(img_synthetic_upright)
        # ax = plt.gca()
        # ax.add_patch(Ellipse(xy=ant1['xy'],
        #                      width=ant1['major'],
        #                      height=ant1['minor'],
        #                      angle=-ant1['angle_deg'],
        #                      edgecolor='r',
        #                      facecolor='none'))
        # ax.add_patch(Ellipse(xy=ant2['xy'],
        #                      width=ant2['major'],
        #                      height=ant2['minor'],
        #                      angle=-ant2['angle_deg'],
        #                      edgecolor='r',
        #                      facecolor='none'))
        # fig.savefig('out/debug/%03d.png' % self.__i__, transparent=True, bbox_inches='tight', pad_inches=0)
        # plt.close(fig)

        ##
        img_size = 200
        img_crop = np.zeros((img_size, img_size, 3), dtype=np.uint8)

        dest_top_left = -np.clip(np.array(centroid_xy[::-1]) - img_size / 2, None, 0).round().astype(int)
        dest_bot_right = np.clip(
            img_size - (np.array(centroid_xy[::-1]) + img_size / 2 - img_synthetic_upright.shape[:2]),
            None, img_size).round().astype(int)
        x_range = np.clip((centroid_xy[0] - img_size / 2, centroid_xy[0] + img_size / 2),
                          0, img_synthetic_upright.shape[1]).round().astype(int)
        y_range = np.clip((centroid_xy[1] - img_size / 2, centroid_xy[1] + img_size / 2),
                          0, img_synthetic_upright.shape[0]).round().astype(int)

        img_crop[dest_top_left[0]:dest_bot_right[0], dest_top_left[1]:dest_bot_right[1]] = \
            img_synthetic_upright[slice(*y_range), slice(*x_range)]

        delta_xy = np.array((x_range[0] - dest_top_left[1], y_range[0] - dest_top_left[0]))
        ant1_crop = copy.deepcopy(ant1)
        ant1_crop['xy'] -= delta_xy
        ant2_crop = copy.deepcopy(ant2)
        ant2_crop['xy'] -= delta_xy

        # fig = plt.figure()
        # plt.imshow(img_crop)
        # ax = plt.gca()
        # ax.add_patch(Ellipse(xy=ant1_crop['xy'],
        #                      width=ant1_crop['major'],
        #                      height=ant1_crop['minor'],
        #                      angle=-ant1_crop['angle_deg'],
        #                      edgecolor='r',
        #                      facecolor='none'))
        # ax.add_patch(Ellipse(xy=ant2_crop['xy'],
        #                      width=ant2_crop['major'],
        #                      height=ant2_crop['minor'],
        #                      angle=-ant2_crop['angle_deg'],
        #                      edgecolor='r',
        #                      facecolor='none'))
        # fig.savefig('out/debug/%03d_crop.png' % self.__i__, transparent=True, bbox_inches='tight', pad_inches=0)
        # plt.close(fig)

        self.__i__ += 1

        return img_crop, ant1_crop, ant2_crop


if __name__ == '__main__':
    fire.Fire(GenerateTrainingData)