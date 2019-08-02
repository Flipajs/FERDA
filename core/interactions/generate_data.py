from __future__ import print_function
# import matplotlib
# matplotlib.use('qt4agg')
"""
Generate augmented training and testing data from FERDA project for

- detection of fixed number of interacting objects
- regression tracking
- segmentation  

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
import h5py
import pandas as pd
import errno
import hashlib
import random
# from joblib import Memory
from os.path import join
import matplotlib.pylab as plt
import itertools
import yaml
import warnings
from collections import defaultdict
from itertools import izip_longest

from core.project.project import Project
from core.graph.region_chunk import RegionChunk
from utils.video_manager import get_auto_video_manager
from shapes.transformableregion import TransformableRegion
from shapes.ellipse import Ellipse
from shapes.bbox import BBox
from shapes.point import Point
from core.interactions.visualization import save_prediction_img, save_img_with_objects
from core.interactions.io import read_gt
from utils.img import safe_crop
from utils.dataset_io import ImageIOFile, ImageIOHdf5, DataIOCSV, DataIOVot, Dataset
from utils.gt.gt import GT
from utils.gt.gt_project import GtProjectMixin
from utils.misc import makedirs

# memory = Memory('out/cache', verbose=1)

class GtProject(GtProjectMixin, GT):
    pass


class TrainingDataset(Dataset):
    def __init__(self, out_dir=None, count=None, image_format=None, data_format=None,
                 img_shape=None, overwrite=False, two_images=False, name='train', csv_name='train.csv',
                 csv_columns=('x', 'y', 'major', 'minor', 'angle_deg_cw'), idx_start=0):
        if out_dir is not None:
            assert count is not None
            padding = ':0{}d'.format(len(str(count)))
            makedirs(out_dir)
            if image_format == 'hdf5':
                assert img_shape is not None
                h5_filename = join(out_dir, 'images.h5')
                h5_file = h5py.File(h5_filename, mode='a')
                if not overwrite:
                    if name in h5_file:
                        raise OSError(errno.EEXIST, 'HDF5 file {} already contains {} dataset/group.'.
                                      format(h5_filename, name))
                else:
                    if name in h5_file:
                        del h5_file[name]
                if two_images:
                    h5group = h5_file.create_group(name)
                    image_io = [
                        ImageIOHdf5(h5group.create_dataset('img0', (count,) + img_shape, np.uint8)),
                        ImageIOHdf5(h5group.create_dataset('img1', (count,) + img_shape, np.uint8)),
                    ]
                else:
                    image_io = [ImageIOHdf5(h5_file.create_dataset(name, (count,) + img_shape, np.uint8))]
            elif image_format == 'file':
                out_dir_imgs = join(out_dir, 'imgs')
                makedirs(out_dir_imgs)
                if two_images:
                    image_io = [
                        ImageIOFile(join(out_dir_imgs, name + '{idx' + padding + '}_img0.jpg')),
                        ImageIOFile(join(out_dir_imgs, name + '{idx' + padding + '}_img1.jpg')),
                    ]
                else:
                    image_io = [ImageIOFile(join(out_dir_imgs, name + '{idx' + padding + '}.jpg'))]
            elif image_format is None:
                image_io = None
            else:
                assert False, 'unknown image_format'

            if data_format == 'csv':
                csv_filename = join(out_dir, csv_name)
                if not overwrite:
                    if os.path.exists(csv_filename):
                        raise OSError(errno.EEXIST, 'file %s already exists.' % csv_filename)
                data_io = DataIOCSV(csv_filename, csv_columns)
            elif data_format == 'vot':
                out_dir_annotations = join(out_dir, 'annotations')
                makedirs(out_dir_annotations)
                data_io = DataIOVot(join(out_dir_annotations, '{idx' + padding + '}.xml'),
                                    image_filename_template='{idx' + padding + '}.jpg',
                                    image_shape=img_shape)
            elif data_format is None:
                data_io = None
            else:
                assert False, 'uknown data_format'
        else:
            image_io = None
            data_io = None

        super(TrainingDataset, self).__init__(image_io, data_io)
        self.next_idx = idx_start


class DataGenerator(object):
    def __init__(self):
        self._video = None
        self._single = None
        self._multi = None
        self._project = None
        self.bg_model = None
        self.params = {'tracklet_min_frames': 0,
                       'tracklet_min_speed_px': 0,
                       'tracklet_remove_fraction': 0,
                       'regression_tracking_image_size_px': 224,
                       'detection_image_size_px': 224,
#                       'augmentation_elliptic_mask_multipliers': (1, 1), # fishes, sowbugs
                       'augmentation_elliptic_mask_multipliers': (1.5, 4),  # ants
                       }
        # self.__i = 0  # used for visualizations commented out
        # self._get_single_region_tracklets_cached = memory.cache(
        #     self._get_single_region_tracklets_cached,
        #     ignore=['self'])  # uncomment to enable caching, disabled due to extremely slow speed

    def _load_project(self, project_dir=None):
        # self._project = Project()
        # self._project.load(project_dir, video_file=video_file)
        self._project = Project(project_dir)
        self._video = get_auto_video_manager(self._project)

    def _write_params(self, out_dir):
        yaml.dump(self.params, open(join(out_dir, 'parameters.yaml'), 'w'))

    @staticmethod
    def _get_hash(*args):
        m = hashlib.md5()
        for arg in args:
            m.update(str(arg))
        return m.hexdigest()

    def _init_regions(self, cache_dir='../data/cache', gt=None):
        hash = self._get_hash(self._project.video_paths, self.params)
        regions_filename = join(cache_dir, hash + '.pkl')
        if os.path.exists(regions_filename):
            print('regions loading...')
            with open(regions_filename, 'rb') as fr:
                self._regions = pickle.load(fr)
            print('regions loaded')
        else:
            self._regions = self._collect_regions(gt)
            with open(regions_filename, 'wb') as fw:
                pickle.dump(self._regions, fw)

    def _collect_regions(self, single=True, multi=True, gt=None, limit=None):
        p = self.params
        regions = []
        regions_in_tracklets = []
        for tracklet in tqdm.tqdm(self._project.chm.chunk_gen(), desc='processing tracklets', total=len(self._project.chm)):
            if tracklet.is_origin_interaction():
                continue
            if tracklet.is_single():
                if not single:
                    continue
                if 'tracklet_min_frames' in p and not (len(tracklet) >= p['tracklet_min_frames']):
                    continue
                if 'tracklet_min_speed' in p and not tracklet.speed >= p['tracklet_min_speed']:
                    continue
                cardinality = 'single'
            if tracklet.is_multi():
                if not multi:
                    continue
                cardinality = 'multi'

            region_tracklet = RegionChunk(tracklet, self._project.gm, self._project.rm)
            region_tracklet_fixed = list(region_tracklet)
            if 'tracklet_remove_fraction' in p:
                n = len(region_tracklet_fixed)
                region_tracklet_fixed = region_tracklet_fixed[
                    int(n * p['tracklet_remove_fraction'] / 2):
                    int(n * (1 - p['tracklet_remove_fraction'] / 2))]

            gt_id = None
            if tracklet.is_single() and gt is not None:
                ids = gt.tracklet_id_set(tracklet, self._project)
                if (ids is not None) and len(ids) == 1:
                    gt_id = ids[0]
                else:
                    warnings.warn('Tracklet id {} has no unique gt match.'.format(tracklet.id()))
                    continue

            tracklet_regions = []
            for r in region_tracklet_fixed:
                r.cardinality = cardinality
                r.gt_id = gt_id
                tracklet_regions.append(r)
            regions.extend(tracklet_regions)
            regions_in_tracklets.append(tracklet_regions)
            if limit is not None and len(regions) > limit:  # for debugging
                break
        return regions, regions_in_tracklets

    def _get_out_dir_rel(self, out_dir, out_file):
        makedirs(out_dir)
        return os.path.relpath(os.path.abspath(out_dir), os.path.abspath(os.path.dirname(out_file)))

    def write_regions_for_testing(self, project_dir, count, out_dir=None, multi=True, single=False, image_format='hdf5',
                                  sort=False, gt_filename=None):
        """
        Write n randomly chosen cropped region images in hdf5 file.

        :param project_dir: gather regions from the FERDA project file
        :param out_dir: output directory where images.h5 and parameters.txt will be placed
        :param count: number of region images to write
        :param multi: include multi animal regions
        :param single: include single animal regions
        """
        # write-regions-for-testing /home/matej/prace/ferda/projects/camera1_10-15/10-15.fproj /home/matej/prace/ferda/data/interactions/180129_camera1_10-15_multi_test 100
        self._load_project(project_dir)
        img_shape = (self.params['detection_image_size_px'], self.params['detection_image_size_px'], 3)
        dataset = TrainingDataset(out_dir, count, image_format, data_format=None, img_shape=img_shape, name='test')
        self._write_params(out_dir)
        regions, _ = self._collect_regions(single, multi)  # , 100)
        print('Total regions: {}'.format(len(regions)))
        selected_regions = np.random.choice(regions, count, replace=False)
        if sort:
            selected_regions = sorted(selected_regions, key=lambda x: x.frame())
        shift_max_px = self.params['detection_image_size_px'] / 2 * 0.7
        for region in tqdm.tqdm(selected_regions):
            shift_xy_px = np.random.uniform(-shift_max_px, shift_max_px, 2)
            img_crop, _ = safe_crop(self._project.img_manager.get_whole_img(region.frame()),
                                    region.centroid()[::-1] + shift_xy_px,
                                    self.params['detection_image_size_px'])
            dataset.add_item(img_crop)
        dataset.close()

    def write_detections(self, project_dir, frame_range=None, out_dir=None, image_format='file', data_format='csv'):
        """
        Write n randomly chosen cropped region images and metadata from a FERDA project.

        write-detections ../projects/2_temp/180810_2359_Cam1_ILP_cardinality_dense/ [100,150] /datagrid/ferda/datasets/tracking/detections_Cam1_clip_range_100_200

        :param project_dir: gather regions from the FERDA project file
        :param frame_range: tuple, inclusive
        :param out_dir: output directory where images.h5 and parameters.txt will be placed
        :param count: number of region images to write
        :param multi: include multi animal regions
        :param single: include single animal regions
        """
        # write-detections ../projects/2_temp/180810_2359_Cam1_ILP_cardinality_dense/ [100,150] /datagrid/ferda/datasets/tracking/detections_Cam1_clip_range_100_200
        self._load_project(project_dir)
        # img_shape = (self.params['detection_image_size_px'], self.params['detection_image_size_px'], 3)
        if frame_range is None:
            frame_range = (self._project.video_start_t, self._project.video_end_t + 1)
        dataset = TrainingDataset(out_dir, (frame_range[1] - frame_range[0] + 1),
                                  image_format, data_format, csv_columns=('frame', 'x', 'y', 'width', 'height'),
                                  idx_start=1, overwrite=True, name='', csv_name='detections.csv')  # , img_shape=img_shape, name='test')
        self._write_params(out_dir)
        regions, _ = self._collect_regions(single=True, multi=False)  # , 100)
        frame_regions = defaultdict(list)
        for r in regions:
            if frame_range[0] <= r.frame() <= frame_range[1]:
                frame_regions[r.frame()].append(r)
        for i, (frame, region_in_frame) in tqdm.tqdm(enumerate(frame_regions.iteritems(), start=1), total=len(frame_regions)):
            img = self._project.img_manager.get_whole_img(frame)
            data = []
            for r in region_in_frame:
                r_dict = r.roi().as_dict()
                r_dict['frame'] = i
                data.append(r_dict)
            dataset.add_item(img, data)
        dataset.close()

    def show_detections(self, detections_csv, image_dir, out_dir):
        df = pd.read_csv(detections_csv, index_col='frame')
        padding = ':0{}d'.format(len(str(df.index.nunique())))
        for i, (frame, df_bboxes) in enumerate(df.groupby(level=0)):
            img = plt.imread(join(image_dir, '{' + padding + '}.jpg').format(frame))
            plt.figure()
            plt.imshow(img)  # [:, :, ::-1]
            for a, bbox in df_bboxes.iterrows():
                BBox.from_dict(bbox.to_dict()).draw()
            plt.show()
            if i == 10:
                break

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
            makedirs(out_dir)
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
            makedirs(sample_dir)
            for i in range(20):
                cv2.imwrite(join(sample_dir, '%02d.png' % i), h5_group_train['img'][i])
                cv2.imwrite(join(sample_dir, '%02d_mask.png' % i), h5_group_train['mask'][i])
            if n_validation != 0:
                for i in range(20):
                    cv2.imwrite(join(sample_dir, 'validation_%02d.png' % i), h5_group_valid['img'][i])
            h5_file.close()

    def write_detection_data(self, project_dir, count, out_dir=None, augmentation=True, overwrite=False,
                             foreground_layer=False, data_format='csv', image_format='hdf5', gt_filename=None):
        """
        Write detection training data with optional augmentation.

        Example script usage arguments:
        write-detection-data ../projects/2_temp/180810_2359_Cam1_ILP_cardinality_dense/ 20 out/dataset --data_format=vot --image_format=file

        :param project_dir: project to use for training data extraction
        :param count: number of generated training samples
        :param out_dir: output directory for images.h5, train.csv, test.csv or None to display the generated data
        :param augmentation: augment training samples with a disruptor object
        :param overwrite: enable to silently overwrite existing data
        :param foreground_layer: add foreground alpha mask as a fourth layer
        :param data_format: csv, vot
        :param image_format: hdf5, file
        :param gt_filename: MOT ground truth file, object labels will be written to the vot annotation file
        """
        self._load_project(project_dir)
        if gt_filename is not None:
            gt = GtProject(gt_filename)
            gt.set_project_offsets(self._project)
            gt.break_on_inconsistency = True
        else:
            gt = None

        img_shape = (self.params['detection_image_size_px'],
                     self.params['detection_image_size_px'],
                     3 if not foreground_layer else 4)
        dataset = TrainingDataset(out_dir, count, image_format, data_format, img_shape, overwrite, name='')
        self._write_params(out_dir)
        padding = ':0{}d'.format(len(str(count)))

        regions, _ = self._collect_regions(single=True, multi=True, gt=gt) # , limit=500)
        regions_by_frame = defaultdict(list)
        regions_by_cardinality = defaultdict(list)
        for r in regions:
            regions_by_frame[r.frame()].append(r)
            regions_by_cardinality[r.cardinality].append(r)

        print('Total regions in single tracklets: {}'.format(len(regions_by_cardinality['single'])))

        random_regions = random.sample(regions_by_cardinality['single'], count)
        if augmentation:
            random_regions_augmentation = random.sample(regions_by_cardinality['single'], count)

        for i, region in enumerate(tqdm.tqdm(random_regions)):
            if augmentation:
                aug_region = random_regions_augmentation[i]
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
                # # img1 rotation around object center
                # rot_deg = np.random.normal(scale=30)

            timg = TransformableRegion()
            img = self._project.img_manager.get_whole_img(region.frame())
            if not foreground_layer:
                timg.set_img(img)
            else:
                timg.set_img(np.dstack((img, self._project.img_manager.get_foreground(region.frame()))))
            timg.set_model(Ellipse.from_region(region))

            if augmentation:
                aug_img = self._project.img_manager.get_whole_img(aug_region.frame())
                if foreground_layer:
                    aug_img = np.dstack((aug_img, self._project.img_manager.get_foreground(aug_region.frame())))
                img, timg_aug = self._augment(timg.get_img(), timg.get_model_copy(),
                                    aug_img, Ellipse.from_region(aug_region),
                                    theta_deg, phi_deg, aug_shift_px)
            else:
                img = timg.get_img()

            # random shift and image crop
            shift_max_px = self.params['detection_image_size_px'] / 2 - timg.model.major / 2 * 2.5
            shift_xy_px = np.random.uniform(-shift_max_px, shift_max_px, 2)
            img_crop, delta_xy, crop_range = safe_crop(img, region.centroid()[::-1] + shift_xy_px,
                                                       self.params['detection_image_size_px'],
                                                       return_src_range=True)

            # bboxes for first object and synthetic object
            model = timg.get_model_copy().move(-delta_xy)
            # model_dict = model.to_dict()
            bbox_models = [BBox.from_planar_object(model)]
            bbox_ids = [region.gt_id]
            points = [[Point(x, y, region.frame()).move(-delta_xy)] for y, x in region.get_head_tail()]
            if augmentation:
                model_aug = timg_aug.get_model_copy().move(-delta_xy)
                # model_dict.update(model_aug.to_dict(1))
                bbox_models.append(BBox.from_planar_object(model_aug))
                bbox_ids.append(aug_region.gt_id)
                for pointlist, yx in zip(points, aug_region.get_head_tail()):
                    x, y = timg_aug.get_transformed_coords(yx[::-1])
                    pointlist.append(Point(x, y, aug_region.frame()).move(-delta_xy))

            # bboxes for other present objects
            size_px = self.params['detection_image_size_px']
            discard = False
            for r in regions_by_frame[region.frame()]:
                bbox = BBox.from_planar_object(Ellipse.from_region(r)).move(-delta_xy)
                if r != region and not bbox.is_outside_bounds(0, 0, size_px, size_px):
                    if r.cardinality == 'multi':
                        discard = True
                        break
                    else:
                        bbox_models.append(bbox)
                        bbox_ids.append(r.gt_id)
                        for pointlist, (y, x) in zip(points, r.get_head_tail()):
                            pointlist.append(Point(x, y, r.frame()).move(-delta_xy))
            if discard:
                continue

            if True:  # isinstance(dataset, DummyDataset):
                makedirs(join(out_dir, 'examples'))
                filename_template = join(out_dir, 'examples', '{idx' + padding + '}.jpg')
                save_img_with_objects(filename_template.format(idx=dataset.next_idx), img_crop[:, :, ::-1],
                                      bbox_models + points[0] + points[1],
                                      bbox_ids + ['head'] * len(points[0]) + ['tail'] * len(points[1]))

            def merge_two_dicts(x, y):
                z = x.copy()  # start with x's keys and values
                z.update(y)  # modifies z with y's keys and values & returns None
                return z

            # [{**bbox.to_dict(), **{'name': 'ant'}} for bbox in bbox_models]  # works in python 3.5
            dataset.add_item(img_crop, [merge_two_dicts(bbox.to_dict(),
                                                       {'name': gt_id if gt_id is not None else 'ant',
                                                        'p0_x': p0.x, 'p0_y': p0.y,
                                                        'p1_x': p1.x, 'p1_y': p1.y})
                                        for bbox, p0, p1, gt_id in izip_longest(bbox_models, points[0], points[1], bbox_ids)] )

        if out_dir is not None:
            dataset.close()

    def write_regression_tracking_data(self, project_dir, count, out_dir=None, test_fraction=0.1, augmentation=True,
                                       overwrite=False, forward=True, backward=False, foreground_layer=False,
                                       data_format='csv', image_format='hdf5'):
        """
        Write regression tracking training and testing data with optional augmentation.

        Example script usage arguments:
        write-regression-tracking-data /home/matej/prace/ferda/projects/2_temp/180713_1633_Cam1_clip_initial 10 out/

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
        img_shape = (self.params['regression_tracking_image_size_px'],
                     self.params['regression_tracking_image_size_px'],
                     3 if not foreground_layer else 4)
        if forward:
            n_train = count
            n_test = int(count * test_fraction)
        else:
            n_train = 0
            n_test = 0
        if backward:
            n_train += count
            n_test += int(count * test_fraction)
        dataset_train = TrainingDataset(out_dir, n_train, image_format, data_format, img_shape, overwrite,
                                        two_images=True, csv_name='train.csv', name='train')
        dataset_test = TrainingDataset(out_dir, n_test, image_format, data_format, img_shape, overwrite,
                                       two_images=True, csv_name='test.csv', name='test')
        self._write_params(out_dir)
        # padding = ':0{}d'.format(len(str(count)))

        regions, regions_in_tracklets = self._collect_regions(single=True, multi=False)  # , limit=500)

        print('Total regions in single tracklets: {}'.format(len(regions)))
        all_indices = []
        for i, rt in enumerate(regions_in_tracklets):
            # indices (i, j) and (i, j+1) will be used
            all_indices.extend([(i, j) for j in xrange(0, len(rt) - 1)])

        indices = random.sample(all_indices, int(count * (1 + test_fraction)))
        if augmentation:
            indices_augmentation = random.sample(all_indices, int(count * (1 + test_fraction)))
        else:
            indices_augmentation = None
        self._write_regression_tracking_dataset(dataset_train, indices[:count],
                                                regions_in_tracklets,
                                                None if not augmentation else indices_augmentation[:count],
                                                forward, backward, foreground_layer)
        self._write_regression_tracking_dataset(dataset_test, indices[count:],
                                                regions_in_tracklets,
                                                None if not augmentation else indices_augmentation[count:],
                                                forward, backward, foreground_layer)
        if out_dir is not None:
            dataset_train.close()
            dataset_test.close()
            self.show_ground_truth(join(out_dir, 'train.csv'), join(out_dir, 'sample'),
                                   join(out_dir, 'images.h5') + ':train/img1', n=20)

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
                    # if image_idx == 1:
                    #     timg.rotate(rot_deg, region.centroid())  # simulate rotation movement
                    aug_img = self._project.img_manager.get_whole_img(aug_region.frame())
                    if foreground_layer:
                        aug_img = np.dstack((aug_img, self._project.img_manager.get_foreground(aug_region.frame())))
                    img, _ = self._augment(timg.get_img(), timg.get_model_copy(),
                                        aug_img, Ellipse.from_region(aug_region),
                                        theta_deg, phi_deg, aug_shift_px)
                else:
                    img = timg.get_img()
                img_crop, delta_xy = safe_crop(img, consecutive_regions[0].centroid()[::-1],
                                               self.params['regression_tracking_image_size_px'])

                images.append(img_crop)

                if dataset.is_dummy():
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

        return img_augmented, timg_aug

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
        makedirs(out_dir)
        for i, row in tqdm.tqdm(df.iterrows(), total=len(df), desc='saving ground truth samples'):
            save_prediction_img(join(out_dir, '%05d.png' % i), n_ids, images[i], pred=None, gt=row)

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
    from os import umask
    os.umask(007)  # all files created will be rw by group
    fire.Fire(DataGenerator)
