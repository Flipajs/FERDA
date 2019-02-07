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
from core.region.ellipse import Ellipse
from core.interactions.visualization import save_prediction_img
from core.interactions.io import read_gt
from utils.img import safe_crop

memory = Memory('out/cache', verbose=1)


IMAGE_SIZE_PX = 200


# TODO: check for possible bug

class ImageIOFile(object):
    def __init__(self, filename_template):
        self.filename_template = filename_template
        self.next_idx = 0

    def add_item(self, image):
        cv2.imwrite(self.filename_template.format(self.next_idx), image)
        self.next_idx += 1

    def read_item(self):
        return cv2.imread(self.filename_template.format(self.next_idx))
        self.next_idx += 1

    def close(self):
        pass


class MultiImageIOFile(object):
    def __init__(self, filename_template, names):
        self.filename_template = filename_template
        self.names = names
        self.next_idx = 0

    def add_item(self, images):
        for name, image in zip(self.names, images):
            cv2.imwrite(
                self.filename_template.format(idx=self.next_idxm, name=name),
                image)
        self.next_idx += 1

    def read_item(self):
        images = []
        for name in self.names:
            images.append(cv2.imread(self.filename_template.format(
                idx=self.next_idx, name=name)))
        self.next_idx += 1
        return images

    def close(self):
        pass


class ImageIOHdf5(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.next_idx = 0

    def add_item(self, image):
        self.dataset[self.next_idx] = image
        self.next_idx += 1

    def read_item(self):
        image = self.dataset[self.next_idx]
        self.next_idx += 1
        return image


class MultiImageIOHdf5(object):
    def __init__(self, h5group, dataset_names, shapes):
        self.datasets = []
        if len(shapes) != len(dataset_names) or not isinstance(shapes, list):
            shapes = [shapes] * len(dataset_names)
        for name, shape in zip(dataset_names, shapes):
            self.datasets.append(h5group.create_dataset(name, shape, np.uint8))
        self.next_idx = 0

    def add_item(self, images):
        for dataset, image in zip(self.datasets, images):
            dataset[self.next_idx] = image
        self.next_idx += 1

    def read_item(self):
        images = [dataset[self.next_idx] for dataset in self.datasets]
        self.next_idx += 1
        return images

    def close(self):
        pass


class DataIOCSV(object):
    def __init__(self, filename, columns=None):
        if columns is not None:
            self.csv_file = open(filename, 'w')
            self.writer = csv.DictWriter(self.csv_file, fieldnames=columns)
            self.writer.writeheader()
            self.reader = None
        else:
            assert False
            # df = pd.read_csv(filename)
            self.reader = None
            self.writer = None

    def add_item(self, data):
        self.writer.writerow(data)

    def read_item(self):
        assert False

    def close(self):
        self.csv_file.close()


class DataIOVot(object):
    def __init__(self, filename_template, image_filename_template=None, image_shape=None):
        self.template = '<annotation><folder>GeneratedData_Train</folder>' \
                 '<filename>{filename}</filename>' \
                 '<path>{path}</path>' \
                 '<source>' \
                 '<database>Unknown</database>' \
                 '</source>' \
                 '<size>' \
                 '<width>{width}</width>' \
                 '<height>{height}</height>' \
                 '<depth>{depth}</depth>' \
                 '</size>' \
                 '<segmented>0</segmented>' \
                 '{annotation}</annotation>'
        self.bbox_template = '<object><name>{name}</name><bndbox>' \
                             '<xmin>{xmin}</xmin><xmax>{xmax}</xmax>' \
                             '<ymin>{ymin}</ymin><ymax>{ymax}</ymax></bndbox></object>'
        self.next_idx = 0
        self.filename_template = filename_template
        if image_filename_template is not None and image_shape is not None:
            # writing
            self.image_filename_template = image_filename_template
            self.template_data = {'height': image_shape[0], 'width': image_shape[1],
                                  'depth': image_shape[2], 'path': ''}
        else:
            assert False

    def add_item(self, data):
        if isinstance(data, dict):
            data = [data]
        assert isinstance(data, list)
        bboxes_str = ''
        for i, bbox in enumerate(data):
            bbox['name'] = i
            bboxes_str += self.bbox_template.format(**bbox)
        annotation = self.template.format(annotation=bboxes_str,
                                          filename=self.image_filename_template.format(self.next_idx),
                                          **self.template_data)
        open(self.filename_template.format(self.next_idx), 'w').write(annotation)
        self.next_idx += 1

    def read_item(self):
        annotation = open(self.filename_template.format(self.next_idx), 'w').read()
        # parse xml
        assert False

    def close(self):
        pass


class Dataset(object):
    def __init__(self, image_io, data_io):
        self.image_io = image_io
        self.data_io = data_io

    def add_item(self, image, data):
        self.image_io.add_item(image)
        self.data_io.add_item(data)

    def read_item(self):
        return self.image_io.read_item(), self.data_io.read_item()

    def close(self):
        self.image_io.close()
        self.data_io.close()


class DummyDataset(object):
    def add_item(self, image, data):
        pass

    def close(self):
        pass


class DataGenerator(object):
    def __init__(self):
        self._video = None
        self._single = None
        self._multi = None
        self._project = None
        self.bg_model = None
        self.params = {'single_min_frames': 0,
                       'single_min_average_speed_px': 0,
                       'regression_tracking_image_size_px': 224,
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
                    r = t.get_region_in_frame(frame)
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
        frames_with_multi = set(itertools.chain(*[range(t.start_frame(),
                                                        t.end_frame() + 1) for t in multi_tracklets]))
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
                                       overwrite=False, forward=True, backward=False, foreground_layer=False,
                                       data_format='csv', image_format='hdf5'):
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
        :param bidi: write bidirectional data, doubles the number of generated samples
        TODO
        :param foreground_layer: add foreground alpha mask as a fourth layer
        """
        self._load_project(project_dir)
        if out_dir is not None:
            img_shape = (self.params['regression_tracking_image_size_px'],
                         self.params['regression_tracking_image_size_px'],
                         3 if not foreground_layer else 4)
            padding = ':0{}d'.format(len(str(count)))
            self._makedirs(out_dir)
            self._write_params(out_dir)
            if image_format == 'hdf5':
                if forward:
                    n_train = count
                    n_test = int(count * test_fraction)
                else:
                    n_train = 0
                    n_test = 0
                if backward:
                    n_train += count
                    n_test += int(count * test_fraction)
                h5_filename = join(out_dir, 'images.h5')
                if not overwrite:
                    if os.path.exists(h5_filename):
                        raise OSError(errno.EEXIST, 'HDF5 file %s already exists.' % h5_filename)
                h5_file = h5py.File(h5_filename, mode='w')
                train_image_io = MultiImageIOHdf5(h5_file.create_group('train'), ['img0', 'img1'],
                                                  (n_train, ) + img_shape)
                test_image_io = MultiImageIOHdf5(h5_file.create_group('test'), ['img0', 'img1'],
                                                 (n_test,) + img_shape)
            elif image_format == 'file':
                out_dir_train = join(out_dir, 'train')
                self._makedirs(out_dir_train)
                train_image_io = MultiImageIOFile(join(out_dir_train, '{idx' + padding + '}_{name}.jpg'), ['0', '1'])
                out_dir_test = join(out_dir, 'test')
                self._makedirs(out_dir_test)
                test_image_io = MultiImageIOFile(join(out_dir_test, '{idx' + padding + '}_{name}.jpg'), ['0', '1'])
            else:
                assert False, 'unknown image_format'
            if data_format == 'csv':
                train_csv_filename = join(out_dir, 'train.csv')
                test_csv_filename = join(out_dir, 'test.csv')
                if not overwrite:
                    if os.path.exists(train_csv_filename):
                        raise OSError(errno.EEXIST, 'file %s already exists.' % train_csv_filename)
                    if os.path.exists(test_csv_filename):
                        raise OSError(errno.EEXIST, 'file %s already exists.' % test_csv_filename)
                columns = ['x', 'y', 'major', 'minor', 'angle_deg_cw']
                train_data_io = DataIOCSV(train_csv_filename, columns)
                test_data_io = DataIOCSV(test_csv_filename, columns)
            elif data_format == 'vot':
                out_dir_train = join(out_dir, 'train')
                self._makedirs(out_dir_train)
                train_data_io = DataIOVot(join(out_dir_train, '{idx' + padding + '}.xml'),
                                          join(out_dir_train, '{idx' + padding + '}.jpg'),
                                          img_shape)
                out_dir_test = join(out_dir, 'test')
                self._makedirs(out_dir_test)
                test_data_io = DataIOVot(join(out_dir_test, '{idx' + padding + '}.xml'),
                                         join(out_dir_test, '{idx' + padding + '}.jpg'),
                                         img_shape)
            else:
                assert False, 'uknown data_format'
            train_dataset = Dataset(train_image_io, train_data_io)
            test_dataset = Dataset(test_image_io, test_data_io)
        else:
            train_dataset = DummyDataset()
            test_dataset = DummyDataset()

        all_regions_idx, single_region_tracklets = self._get_single_region_tracklets()  # limit=20)  # for debugging

        print('Total regions in single tracklets: {}'.format(len(all_regions_idx)))

        # with open('single_regions.pkl', 'wb') as fw:
        #     pickle.dump(single_region_tracklets, fw)
        #     pickle.dump(all_regions_idx, fw)

        idxs = random.sample(all_regions_idx, int(count * (1 + test_fraction)))
        if augmentation:
            idxs_augmentation = random.sample(all_regions_idx, int(count * (1 + test_fraction)))
        else:
            idxs_augmentation = None
        self._write_regression_tracking_dataset(train_dataset, idxs[:count],
                                                single_region_tracklets,
                                                None if not augmentation else idxs_augmentation[:count],
                                                forward, backward, foreground_layer)
        self._write_regression_tracking_dataset(test_dataset, idxs[count:],
                                                single_region_tracklets,
                                                None if not augmentation else idxs_augmentation[count:],
                                                forward, backward, foreground_layer)
        if out_dir is not None:
            train_dataset.close()
            test_dataset.close()
            h5_file.close()
            self.show_ground_truth(train_csv_filename, join(out_dir, 'sample'), h5_filename + ':train/img1', n=20)

    def _get_single_region_tracklets(self, limit=None):
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
                single_region_tracklets.append(region_tracklet_fixed)
                all_regions_idx.extend([(i, j) for j in range(len(region_tracklet_fixed) - 1)])
                i += 1
            if limit is not None and i == limit:  # for debugging
                break
        return all_regions_idx, single_region_tracklets

    def _write_regression_tracking_dataset(self, dataset, idx, tracklets, idxs_augmentation=None,
                                           forward=True, backward=False, foreground_layer=False):
        """
        Generate regression tracking training data.

        :param dataset: Dataset()
        :param idx: tracklet / region indices, [(tracklet_idx, region_idx), (tracklet_idx, region_idx), ...]
        :param tracklets: tracklets regions, [[Region, Region, ....], [Region, ...], ...]
        :param idxs_augmentation: tracklet / region indices for augmentation
        :param bidi: write bidirectional data, doubles the number of generated samples
        """
        if forward:
            self._write_regression_tracking_images(tracklets, idx, dataset, idxs_augmentation,
                                                   forward=True, foreground_layer=foreground_layer)
        if backward:
            self._write_regression_tracking_images(tracklets, idx, dataset, idxs_augmentation,
                                                   forward=False, foreground_layer=foreground_layer)

    def _write_regression_tracking_images(self, tracklets, idx, dataset,
                                          idxs_augmentation=None, forward=True,
                                          foreground_layer=False):
        for i, (tracklet_idx, region_idx) in enumerate(tqdm.tqdm(idx)):
            if forward:
                region_indices = (region_idx, region_idx + 1)
            else:
                region_indices = (region_idx + 1, region_idx)
            consecutive_regions = [tracklets[tracklet_idx][j] for j in region_indices]
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
                # shift along synthetic object's major axis,
                # negative means further from augmented object, positive increase overlap with the augmented object
                # 0 is located at "head" tip of the synthetic object
                aug_shift_px = int(round(np.random.normal(scale=10, loc=5)))
                # img1 rotation around object center
                rot_deg = np.random.normal(scale=30)

            # generate both images
            images = []
            for region, aug_region, image_idx in zip(consecutive_regions, aug_consecutive_regions, (0, 1)):
                timg = TransformableRegion()
                img = self._project.img_manager.get_whole_img(region.frame())
                if not foreground_layer:
                    timg.set_img(img)
                else:
                    timg.set_img(np.dstack((img, self._project.img_manager.get_foreground(region.frame()))))
                timg.set_model(Ellipse.from_region(region))
                # undo the first frame rotation
                timg.rotate(-consecutive_regions[0].angle_deg_cw, consecutive_regions[0].centroid())

                if idxs_augmentation is not None:
                    if image_idx == 1:
                        timg.rotate(rot_deg, region.centroid())  # simulate rotation movement
                    aug_img = self._project.img_manager.get_whole_img(aug_region.frame())
                    if foreground_layer:
                        aug_img = np.dstack((aug_img, self._project.img_manager.get_foreground(aug_region.frame())))
                    img = self._augment(timg.get_img(), timg.get_model_copy(),
                                        aug_img, Ellipse.from_region(aug_region),
                                        theta_deg, phi_deg, aug_shift_px)
                else:
                    img = timg.get_img()
                img_crop, delta_xy = safe_crop(img, consecutive_regions[0].centroid()[::-1],
                                               self.params['regression_tracking_image_size_px'])

                images.append(img_crop)

                if isinstance(dataset, DummyDataset):
                    plt.imshow(save_prediction_img(None, 1, img_crop,
                                                   timg.get_model_copy().move(-delta_xy).to_dict(),
                                                   title=dataset))
                    plt.show()

            model = timg.get_model_copy().move(-delta_xy)
            dataset.add_item(images, {
                    'x': model.x,
                    'y': model.y,
                    'major': model.major,
                    'minor': model.minor,
                    'angle_deg_cw': model.angle_deg,
                })

    def _get_bg_model(self):
        from core.bg_model.median_intensity import MedianIntensity
        if self.bg_model is None:
            self.bg_model = MedianIntensity(self._project)
            self.bg_model.compute_model()
        return self.bg_model.bg_model

    def _draw_elliptic_mask(self, shape, ellipse, major_multiplier=1, minor_multiplier=1, dtype=np.uint8):
        img = np.zeros(shape=shape, dtype=dtype)
        if dtype == np.uint8:
            color = 255
        else:
            color = 1.
        cv2.ellipse(img, tuple(ellipse.xy.astype(int)),
                    (int(major_multiplier * ellipse.major / 2.) + 4,
                     int(minor_multiplier * ellipse.minor / 2.) + 4),
                    int(ellipse.angle_deg), 0, 360, color, -1)
        return cv2.GaussianBlur(img, (5, 5), -1)

    def _augment(self, img, ellipse, img_aug, ellipse_aug, theta_deg, phi_deg, aug_shift_px):
        timg_aug = TransformableRegion(img_aug)
        timg_aug.set_model(ellipse_aug)

        # construct augmented image alpha channel
        bg_diff = (self._get_bg_model().astype(np.float) - img_aug[:, :, :3]).mean(axis=2).clip(5, 100)
        img_aug_alpha = (bg_diff - bg_diff.min()) / np.ptp(bg_diff)
        elliptic_mask = self._draw_elliptic_mask(img_aug_alpha.shape, ellipse_aug, dtype=float,
                                                 *self.params['augmentation_elliptic_mask_multipliers'])
        timg_aug.set_mask(img_aug_alpha * elliptic_mask)

        head_xy, _ = ellipse_aug.get_vertices()
        timg_aug.move(-head_xy[::-1])  # move head of synthetic object to (0, 0)
        timg_aug.rotate(-ellipse_aug.angle_deg, (0, 0))  # normalize the rotation to 0 deg, tail is to left, head to right
        timg_aug.rotate(phi_deg, (0, 0))  # approach angle
        timg_aug.move((0, aug_shift_px))  # positioning closer (even overlapping) or further from object 1
        # object 2 will be positioned head first under angle theta with respect to the main object 1
        timg_aug.rotate(180 + theta_deg, (0, 0))
        timg_aug.move(ellipse.get_point(theta_deg)[::-1])  # move the object 2 to the object 1 border

        img_aug_alpha = timg_aug.get_mask(alpha=True)[:, :, None]  # range <0, 1>, dtype=float
        img_augmented = (img.astype(float) * (1 - img_aug_alpha) +
                         timg_aug.get_img().astype(float) * img_aug_alpha).astype(np.uint8)

        return img_augmented

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
