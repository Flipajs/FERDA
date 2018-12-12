from __future__ import print_function

"""
use: $ python -m core.interactions.generate_data
"""

#TODO change self.__video.get_frame() for cached self._project.img_manager.get_whole_img()

import sys
import pickle
import numpy as np
import math
import os.path
import cv2
import fire
import tqdm
import csv
import h5py
import pandas as pd
import errno
import hashlib
import random
from joblib import Memory
from os.path import join
import matplotlib.pylab as plt
import itertools
import yaml

from core.project.project import Project
from core.graph.region_chunk import RegionChunk
from utils.video_manager import get_auto_video_manager
from core.region.transformableregion import TransformableRegion
from core.region.region import get_region_endpoints
from core.region.ellipse import Ellipse
from core.interactions.visualization import save_prediction_img
from core.interactions.io import read_gt
from utils.img import safe_crop

memory = Memory('out/cache', verbose=1)


IMAGE_SIZE_PX = 200


#TODO: check for possible bug
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

    # fix tracklet
    for state, r in zip(result, tracklet_regions):
        if state:
            r.theta_ += np.pi
            if r.theta_ >= 2 * np.pi:
                r.theta_ -= 2 * np.pi


class DataGenerator(object):
    def __init__(self):
        self._video = None
        self._single = None
        self._multi = None
        self._project = None
        self.bg_model = None
        self.params = {'single_min_frames': 0,
                       'single_min_average_speed_px': 0,
                       'regression_tracking_image_size_px': 150,
                       'single_tracklet_min_speed': 0,
                       'single_tracklet_remove_fraction': 0,
#                       'augmentation_elliptic_mask_multipliers': (1, 1), # fishes, sowbugs
                       'augmentation_elliptic_mask_multipliers': (1.5, 4),  # ants
                       }
        # self.__i = 0  # used for visualizations commented out
        # self._get_single_region_tracklets_cached = memory.cache(
        #     self._get_single_region_tracklets_cached,
        #     ignore=['self'])  # uncomment to enable caching, disabled due to extremely slow speed

    def _load_project(self, project_dir=None, video_file=None):
        self._project = Project()
        self._project.load(project_dir, video_file=video_file)
        self._video = get_auto_video_manager(self._project)

    def _write_params(self, out_dir):
        yaml.dump(self.params, open(join(out_dir, 'parameters.yaml'), 'w'))

    @staticmethod
    def _get_hash(*args):
        m = hashlib.md5()
        for arg in args:
            m.update(str(arg))
        return m.hexdigest()

    def _init_regions(self, cache_dir='../data/cache'):
        hash = self._get_hash(self._project.video_paths, self.params)
        regions_filename = join(cache_dir, hash + '.pkl')
        if os.path.exists(regions_filename):
            print('regions loading...')
            with open(regions_filename, 'rb') as fr:
                self._single = pickle.load(fr)
                self._multi = pickle.load(fr)
            print('regions loaded')
        else:
            self._single, self._multi = self._collect_regions()
            with open(regions_filename, 'wb') as fw:
                pickle.dump(self._single, fw)
                pickle.dump(self._multi, fw)

    def _collect_regions(self):
        p = self.params
        from collections import defaultdict
        single = defaultdict(list)
        multi = defaultdict(list)
        # long_moving_tracklets = []
        for tracklet in tqdm.tqdm(self._project.chm.chunk_gen(), desc='collecting regions', total=len(self._project.chm)):
            if tracklet.is_single() or tracklet.is_multi():
                region_tracklet = RegionChunk(tracklet, self._project.gm, self._project.rm)
                if tracklet.is_single():
                    centroids = np.array([region_tracklet.centroid_in_t(frame) for frame
                                          in
                                          range(region_tracklet.start_frame(),
                                                region_tracklet.end_frame())])  # shape=(n,2)

                    if len(tracklet) > p['single_min_frames'] and \
                            np.linalg.norm(np.diff(centroids, axis=0), axis=1).mean() > p['single_min_average_speed_px']:
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
                            single[region.frame()].append(region)
                    else:
                        # short single tracklets are ignored
                        pass
                else:  # multi
                    for region in region_tracklet.regions_gen():
                        # if tracklet.is_single():
                        #     single[region.frame()].append(region)
                        # else:
                        multi[region.frame()].append(region)

        return single, multi

    def _get_out_dir_rel(self, out_dir, out_file):
        try:
            os.makedirs(out_dir)
        except OSError:
            pass
        return os.path.relpath(os.path.abspath(out_dir), os.path.abspath(os.path.dirname(out_file)))

    def write_regions_for_testing(self, project_dir, out_dir, n, multi=True, single=False):
        """
        Write n randomly chosen cropped region images in hdf5 file.

        :param project_dir: gather regions from the FERDA project file
        :param out_dir: output directory where images.h5 and parameters.txt will be placed
        :param n: number of region images to write
        :param multi: include multi animal regions
        :param single: include single animal regions
        """
        # write-regions-for-testing /home/matej/prace/ferda/projects/camera1_10-15/10-15.fproj /home/matej/prace/ferda/data/interactions/180129_camera1_10-15_multi_test 100
        self._load_project(project_dir)
        self._write_argv(out_dir)
        # load regions
        self._init_regions()
        regions = []
        if multi:
            regions.extend([item for sublist in self._multi.values() for item in sublist])
        if single:
            regions.extend([item for sublist in self._single.values() for item in sublist])
        n_regions = np.random.choice(regions, n)
        frames = np.array([r.frame() for r in n_regions])
        sort_idx = np.argsort(frames)
        # sort_idx_reverse = np.argsort(sort_idx)

        # initialize hdf5 output file
        out_hdf5 = join(out_dir, 'images.h5')
        dataset = 'test'
        if os.path.exists(out_hdf5):
            raise OSError(errno.EEXIST, 'HDF5 file %s already exists.' % out_hdf5)
        hdf5_file = h5py.File(out_hdf5, mode='w')
        hdf5_file.create_dataset(dataset, (n, IMAGE_SIZE_PX, IMAGE_SIZE_PX, 3), np.uint8)

        # load images, crop regions and save to hdf5
        for i, (frame, region) in tqdm.tqdm(enumerate(zip(frames[sort_idx], n_regions[sort_idx])), total=n):
            img = self._project.img_manager.get_whole_img(frame)
            img_crop, delta_xy = safe_crop(img, region.centroid()[::-1], IMAGE_SIZE_PX)
            hdf5_file[dataset][i, ...] = img_crop
        hdf5_file.close()

    def _write_argv(self, out_dir):
        with open(join(out_dir, 'parameters.txt'), 'w') as fw:
            fw.writelines('\n'.join(sys.argv))

    def write_annotated_blobs_groundtruth(self, project_dir, blobs_filename, n_objects, out_dir):
        """
        Load annotated blobs and write interaction ground truth to hdf5 and csv files.

        For annotated blobs see data.GT.region_annotation_tools.ant_blob_gt_manager.
        """
        # write_annotated_blobs_groundtruth '/home/matej/prace/ferda/projects/camera1/Camera 1.fproj' ../data/annotated_blobs/Camera1_blob_gt.pkl 2 ../data/interactions/180126_test_real_2_ants
        self._write_argv(out_dir)
        import scripts.region_annotation_tools.region_annotation_tool as ant_blob_gt_manager
        self._load_project(project_dir)
        img_shape = self._project.img_manager.get_whole_img(0).shape[:2]
        blob_manager = ant_blob_gt_manager.AntBlobGtManager(blobs_filename, self._project)
        blobs = blob_manager.get_ant_blobs()  # see AntBlobs() docs
        gt = pd.DataFrame(ant_blob_gt_manager.blobs_to_dict(blobs, img_shape, self._project.rm))
        gt_n = gt[gt.n_objects == n_objects].copy()

        out_hdf5 = join(out_dir, 'images.h5')
        dataset = 'test'
        if os.path.exists(out_hdf5):
            raise OSError(errno.EEXIST, 'HDF5 file %s already exists.' % out_hdf5)
        hdf5_file = h5py.File(out_hdf5, mode='w')
        hdf5_file.create_dataset(dataset, (len(gt_n), IMAGE_SIZE_PX, IMAGE_SIZE_PX, 3), np.uint8)  # , compression='szip')
        for i, (_, item) in enumerate(tqdm.tqdm(gt_n.iterrows(), desc='writing images')):
            img = self._project.img_manager.get_whole_img(item['frame'])
            region = self._project.rm[item['region_id']]
            img_crop, delta_xy = safe_crop(img, region.centroid()[::-1], IMAGE_SIZE_PX)
            for j in range(n_objects):
                gt_n.loc[gt_n.index[i], '%d_x' % j] -= delta_xy[0]
                gt_n.loc[gt_n.index[i], '%d_y' % j] -= delta_xy[1]
            hdf5_file[dataset][i, ...] = img_crop
        hdf5_file.close()

        out_csv = join(out_dir, 'test.csv')
        columns = []
        for i in range(n_objects):
            gt_n.loc[:, '%d_region_id' % i] = gt_n['region_id']
            columns.extend([
                '%d_x' % i, '%d_y' % i, '%d_major' % i, '%d_minor' % i,
                '%d_angle_deg' % i, '%d_region_id' % i
            ])
        gt_out = gt_n.loc[:, columns]
        gt_out['video_file'] = os.path.basename(self._video.video_path)
        gt_out.to_csv(out_csv, index=False, float_format='%.2f')
        # gt.to_hdf(join(out_dir, 'test.h5'), )

    def _write_segmentation_dataset(self, h5_group, frames, write_masks=True, single_color=True):
        # img_shape = self._project.img_manager.get_whole_img(0).shape
        img_shape = (512, 512)
        if h5_group is not None:
            h5_group.create_dataset('img', (len(frames), img_shape[0], img_shape[1], 3), np.uint8)
            h5_group.create_dataset('mask', (len(frames), img_shape[0], img_shape[1]), np.uint8)

        for i, frame in enumerate(frames):
            img = cv2.cvtColor(self._project.img_manager.get_whole_img(frame), cv2.COLOR_BGR2GRAY)
            if img.shape[:2] != tuple(img_shape[:2]):
                img = cv2.resize(img, img_shape, interpolation=cv2.INTER_LANCZOS4)
            if write_masks:
                mask = np.zeros(shape=img_shape[:2], dtype=np.uint8)
                for t in self._project.chm.tracklets_in_frame(frame):
                    r = t.get_region_in_frame(self._project.gm, frame)
                    yx = r.contour_without_holes()
                    if single_color:
                        color = 255
                    else:
                        color = random.randint(1, 255)
                    cv2.drawContours(mask, [yx[:, ::-1]], -1, color, cv2.FILLED)
            if h5_group is not None:
                h5_group['img'][i] = img
                if write_masks:
                    h5_group['mask'][i] = mask
            else:
                plt.imshow(img)
                plt.show()
                if write_masks:
                    plt.imshow(mask)
                    plt.show()

    def write_segmentation_data(self, project_dir, n, out_h5_filename=None, test_fraction=0.1, overwrite=False,
                                n_validation=0):
        self._load_project(project_dir)

        if out_h5_filename is not None:
            out_dir = os.path.dirname(out_h5_filename)
            self._makedirs(out_dir)
            if not overwrite:
                if os.path.exists(out_h5_filename):
                    raise OSError(errno.EEXIST, 'HDF5 file %s already exists.' % out_h5_filename)
            h5_file = h5py.File(out_h5_filename, mode='w')
            h5_group_train = h5_file.create_group('train')
            h5_group_test = h5_file.create_group('test')
            if n_validation != 0:
                h5_group_valid = h5_file.create_group('valid')
        else:
            h5_group_train = None
            h5_group_test = None

        multi_tracklets = [t for t in self._project.chm.chunk_gen() if t.is_multi()]
        frames_with_multi = set(itertools.chain(*[range(t.start_frame(self._project.gm),
                                                        t.end_frame(self._project.gm) + 1) for t in multi_tracklets]))
        frames_without_multi = list(set(range(self._project.video_start_t,
                                              self._project.video_end_t + 1)) - frames_with_multi)
        frames_selected = random.sample(frames_without_multi, int(n * (1 + test_fraction)))
        print('Selected {} out of {} frames with all objects correctly segmented.'.format(len(frames_selected),
                                                                                          len(frames_without_multi)))
        self._write_segmentation_dataset(h5_group_train, frames_selected[:n])
        self._write_segmentation_dataset(h5_group_test, frames_selected[n:])
        if n_validation != 0:
            frames_selected = random.sample(frames_with_multi, n_validation)
            print('Selected {} out of {} frames with multi tracklets.'.format(len(frames_selected),
                                                                              len(frames_with_multi)))
            self._write_segmentation_dataset(h5_group_valid, frames_selected, write_masks=False)

        if out_h5_filename is not None:
            sample_dir = join(out_dir, 'sample')
            DataGenerator._makedirs(sample_dir)
            for i in range(20):
                cv2.imwrite(join(sample_dir, '%02d.png' % i), h5_group_train['img'][i])
                cv2.imwrite(join(sample_dir, '%02d_mask.png' % i), h5_group_train['mask'][i])
            if n_validation != 0:
                for i in range(20):
                    cv2.imwrite(join(sample_dir, 'validation_%02d.png' % i), h5_group_valid['img'][i])
            h5_file.close()

    def write_regression_tracking_data(self, project_dir, count, out_dir=None, test_fraction=0.1, augmentation=True,
                                       overwrite=False):
        """
        Write regression tracking training and testing data with optional augmentation.

        Example script usage arguments:
        --write-regression-tracking-data /home/matej/prace/ferda/projects/2_temp/180713_1633_Cam1_clip_initial . 10

        :param project_dir: project to use for training data extraction
        :param count: number of generated training samples
        :param out_dir: output directory for images.h5, train.csv, test.csv or None to display the generated data
        :param test_fraction: fraction of test samples with respect to training samples
        :param augmentation: augment training samples with a disruptor object
        :param overwrite: enable to silently overwrite existing data
        """
        self._load_project(project_dir)

        if out_dir is not None:
            self._makedirs(out_dir)
            self._write_params(out_dir)
            h5_filename = join(out_dir, 'images.h5')
            train_csv_filename = join(out_dir, 'train.csv')
            test_csv_filename = join(out_dir, 'test.csv')

            if not overwrite:
                if os.path.exists(h5_filename):
                    raise OSError(errno.EEXIST, 'HDF5 file %s already exists.' % h5_filename)
                if os.path.exists(train_csv_filename):
                    raise OSError(errno.EEXIST, 'file %s already exists.' % train_csv_filename)
                if os.path.exists(test_csv_filename):
                    raise OSError(errno.EEXIST, 'file %s already exists.' % test_csv_filename)
            h5_file = h5py.File(h5_filename, mode='w')
            h5_group_train = h5_file.create_group('train')
            h5_group_test = h5_file.create_group('test')
        else:
            train_csv_filename = None
            test_csv_filename = None
            h5_group_train = None
            h5_group_test = None

        all_regions_idx, single_region_tracklets = self._get_single_region_tracklets()

        print('Total regions in single tracklets: {}'.format(len(all_regions_idx)))

        # with open('single_regions.pkl', 'wb') as fw:
        #     pickle.dump(single_region_tracklets, fw)
        #     pickle.dump(all_regions_idx, fw)

        idxs = random.sample(all_regions_idx, int(count * (1 + test_fraction)))
        if augmentation:
            idxs_augmentation = random.sample(all_regions_idx, int(count * (1 + test_fraction)))
        else:
            idxs_augmentation = None
        self._write_regression_tracking_dataset(h5_group_train, idxs[:count],
                                                single_region_tracklets, train_csv_filename,
                                                None if not augmentation else idxs_augmentation[:count])
        self._write_regression_tracking_dataset(h5_group_test, idxs[count:],
                                                single_region_tracklets, test_csv_filename,
                                                None if not augmentation else idxs_augmentation[count:])
        if out_dir is not None:
            h5_file.close()
            self.show_ground_truth(train_csv_filename, join(out_dir, 'sample'), h5_filename + ':train/img1', n=20)

    def _get_single_region_tracklets_cached(self, project_working_directory):
        """
        :param project_working_directory: used only for caching
        :return:
        """
        single_region_tracklets = []
        all_regions_idx = []  # [(i1, j1), (i2, j2), ... ] where j in <0, n-2> where n is tracklet length
        i = 0
        for tracklet in tqdm.tqdm(self._project.chm.chunk_gen(), total=len(self._project.chm),
                                  desc='gathering single tracklets'):
            if len(tracklet) > self.params['single_min_frames'] and tracklet.is_single():
                region_tracklet = RegionChunk(tracklet, self._project.gm, self._project.rm)
                if 'single_tracklet_min_speed' in self.params:
                    distances = np.linalg.norm(np.diff(
                        np.array([r.centroid() for r in region_tracklet.regions_gen()]), axis=0), axis=1)
                    if distances.mean() < self.params['single_tracklet_min_speed']:
                        continue
                region_tracklet_fixed = list(region_tracklet)
                if 'single_tracklet_remove_fraction' in self.params:
                    n = len(region_tracklet_fixed)
                    region_tracklet_fixed = region_tracklet_fixed[
                        int(n * self.params['single_tracklet_remove_fraction'] / 2):
                        int(n * (1 - self.params['single_tracklet_remove_fraction'] / 2))]
                head_fix(region_tracklet_fixed)
                single_region_tracklets.append(region_tracklet_fixed)
                all_regions_idx.extend([(i, j) for j in range(len(region_tracklet_fixed) - 1)])
                i += 1
            # if i == 10:  # for debugging
            #     break
        return all_regions_idx, single_region_tracklets

    def _get_single_region_tracklets(self):
        return self._get_single_region_tracklets_cached(self._project.working_directory)

    def _write_regression_tracking_dataset(self, h5_group, idx, tracklets, out_csv, idxs_augmentation=None):
        """
        Generate regression tracking training data.

        :param h5_group: h5 group or dataset or None to display the images
        :param idx: tracklet / region indices, [(tracklet_idx, region_idx), (tracklet_idx, region_idx), ...]
        :param tracklets: tracklets regions, [[Region, Region, ....], [Region, ...], ...]
        :param out_csv: out csv filename
        :param idxs_augmentation: tracklet / region indices for augmentation
        """

        dataset_names = ['img0', 'img1']
        img_size_px = self.params['regression_tracking_image_size_px']
        if h5_group is not None:
            for name in dataset_names:
                h5_group.create_dataset(name, (len(idx), img_size_px, img_size_px, 3), np.uint8)

        if out_csv is not None:
            # initialize csv output file
            COLUMNS = ['x', 'y', 'major', 'minor', 'angle_deg_cw']  # , 'region1_id', 'region2_id']
            csv_file = open(out_csv, 'w')
            csv_writer = csv.DictWriter(csv_file, fieldnames=COLUMNS)
            csv_writer.writeheader()

        for i, (tracklet_idx, region_idx) in enumerate(tqdm.tqdm(idx)):
            # a region in the frame n and n + 1
            consecutive_regions = [tracklets[tracklet_idx][j] for j in (region_idx, region_idx + 1)]
            if idxs_augmentation:
                aug_tracklet_idx, aug_region_idx = idxs_augmentation[i]
                aug_consecutive_regions = [tracklets[aug_tracklet_idx][j] for j in (aug_region_idx, aug_region_idx + 1)]
            else:
                aug_consecutive_regions = [None, None]

            if idxs_augmentation is not None:
                # generate parameters for augmentation
                # border point angle with respect to object centroid, 0 rad is from the centroid rightwards, positive ccw
                theta_deg = np.random.uniform(-180, 180)
                # approach angle, 0 rad is direction from the object centroid
                # phi_deg = np.clip(np.random.normal(scale=90 / 2, size=n), -80, 80)
                phi_deg = np.random.uniform(-90, 90)
                # shift along ellipse_a major axis
                aug_shift_px = int(round(np.random.normal(scale=5, loc=-15)))
                # img1 rotation around object center
                rot_deg = np.random.normal(scale=30)

            for region, aug_region, dataset in zip(consecutive_regions, aug_consecutive_regions, dataset_names):
                timg = TransformableRegion()
                timg.set_img(self._project.img_manager.get_whole_img(region.frame()))
                timg.set_model(Ellipse.from_region(region))
                # undo the first frame rotation
                timg.rotate(-consecutive_regions[0].angle_deg_cw, consecutive_regions[0].centroid())

                if idxs_augmentation is not None:
                    if dataset == 'img1':
                        timg.rotate(rot_deg, region.centroid())  # simulate rotation movement
                    aug_img = self._project.img_manager.get_whole_img(aug_region.frame())
                    img = self._augment(timg.get_img(), timg.get_model_copy(), aug_img, Ellipse.from_region(aug_region),
                                        theta_deg, phi_deg, aug_shift_px)
                else:
                    img = timg.get_img()
                img_crop, delta_xy = safe_crop(img, consecutive_regions[0].centroid()[::-1], img_size_px)

                if h5_group is not None:
                    h5_group[dataset][i] = img_crop
                else:
                    ##
                    plt.imshow(save_prediction_img(None, 1, img_crop,
                                                   timg.get_model_copy().move(-delta_xy).to_dict(),
                                                   title=dataset))
                    plt.show()
                    ##

            if out_csv is not None:
                model = timg.get_model_copy().move(-delta_xy)
                csv_writer.writerow({
                    'x': model.x,
                    'y': model.y,
                    'major': model.major,
                    'minor': model.minor,
                    'angle_deg_cw': model.angle_deg,
                })

        if out_csv is not None:
            csv_file.close()

    def _get_bg_model(self):
        from core.bg_model.median_intensity import MedianIntensity
        if self.bg_model is None:
            self.bg_model = MedianIntensity(self._project)
            self.bg_model.compute_model()
        return self.bg_model.bg_model

    def _augment(self, img, ellipse, img_a, ellipse_a, theta_deg, phi_deg, aug_shift_px):
        # base_tregion = TransformableRegion(img)
        # # base_tregion.set_region(region)
        # base_tregion.set_ellipse(ellipse)
        # base_tregion.set_elliptic_mask()
        # center_xy = region.centroid()[::-1].astype(int)

        # construct alpha channel
        bg_diff = (self._get_bg_model().astype(np.float) - img_a).mean(axis=2).clip(5, 100)
        alpha = ((bg_diff - bg_diff.min()) / np.ptp(bg_diff))
        img_rgba = np.concatenate((img_a, np.expand_dims(alpha * 255, 2).astype(np.uint8)), 2)

        tregion = TransformableRegion(img_rgba)
        tregion.set_model(ellipse_a)
        multipliers = self.params['augmentation_elliptic_mask_multipliers']
        tregion.set_elliptic_mask(*multipliers)

        head_xy, _ = ellipse_a.get_vertices()
        tregion.move(-head_xy[::-1])  # move head of object 2 to (0, 0)
        tregion.rotate(-ellipse_a.angle_deg, (0, 0))  # normalize the rotation to 0 deg, tail is to left, head to right
        tregion.rotate(phi_deg, (0, 0))  # approach angle
        tregion.move((0, aug_shift_px))  # positioning closer (even overlapping) or further from object 1
        # object 2 will be positioned head first under angle theta with respect to the main object 1
        tregion.rotate(180 + theta_deg, (0, 0))
        tregion.move(ellipse.get_point(theta_deg)[::-1])  # move the object 2 to the object 1 border

        alpha_trans = tregion.get_img()[:, :, -1].astype(float)
        alpha_trans *= tregion.get_mask(alpha=True) / 255
        alpha_trans /= 255
        alpha_trans = np.expand_dims(alpha_trans, 2)

        img_synthetic = (img.astype(float) * (1 - alpha_trans) +
                         tregion.get_img()[:, :, :-1].astype(float) * alpha_trans).astype(np.uint8)

        return img_synthetic

    @staticmethod
    def show_ground_truth(csv_file, out_dir, image_hdf5='images.h5:train/img1', n=None):
        """
        Save images with visualized ground truth.

        :param csv_file: input ground truth
        :param out_dir: directory for output images
        :param image_hdf5: input images, image_dir or image_hdf5 is required
        """
        hf = h5py.File(image_hdf5.split(':')[0], 'r')
        images = hf[image_hdf5.split(':')[1]]
        n_ids, _, df = read_gt(csv_file)
        if n is not None:
            df = df[:n]
        DataGenerator._makedirs(out_dir)
        for i, row in tqdm.tqdm(df.iterrows(), total=len(df), desc='saving ground truth samples'):
            save_prediction_img(join(out_dir, '%05d.png' % i), n_ids, images[i], pred=None, gt=row)

    @staticmethod
    def _makedirs(dir_name):
        try:
            os.makedirs(dir_name)
        except OSError:
            pass

    def _get_moments(self, mask):
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
        major_deg = math.degrees(0.5 * math.atan2(2 * moments['muprime11'],
                                                  (moments['muprime20'] - moments['muprime02'])))
        return centroid_xy, major_deg


if __name__ == '__main__':
    fire.Fire(DataGenerator)
