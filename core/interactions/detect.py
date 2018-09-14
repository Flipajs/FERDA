import tempfile

import fire
import imageio
from keras.applications.mobilenet import mobilenet
import numpy as np
import pandas as pd
import yaml
from keras.models import model_from_yaml, model_from_json
import os.path
from os.path import join
from os import remove
from scipy.special import expit
from tqdm import tqdm

from core.interactions.visualization import save_prediction_img, show_prediction
from utils.angles import angle_absolute_error_direction_agnostic
from utils.img import safe_crop
from utils.objectsarray import ObjectsArray
from core.interactions.generate_data import DataGenerator
from core.interactions.train import TrainInteractions
from core.region.transformableregion import TransformableRegion


class InteractionDetector:
    def __init__(self, model_dir):
        """
        Load interaction detector keras model.

        :param model_dir: directory with model.{yaml or json}, weights.h5 and config.yaml
        """
        self.TRACKING_COST_WEIGHT = 0.8
        self.TRACKING_CONFIDENCE_LOC = 20
        self.TRACKING_CONFIDENCE_SCALE = 0.2
        keras_custom_objects = {
            'relu6': mobilenet.relu6,
        }
        with open(join(model_dir, 'config.yaml'), 'r') as fr:
            self.config = yaml.load(fr)
        self.ti = TrainInteractions(self.config['num_objects'], detector_input_size_px=self.config['input_size_px'])  # TODO: remove dependency

        if os.path.exists(join(model_dir, 'model.yaml')):
            with open(join(model_dir, 'model.yaml'), 'r') as fr:
                self.m = model_from_yaml(fr.read(), custom_objects=keras_custom_objects)
        elif os.path.exists(join(model_dir, 'model.json')):
            with open(join(model_dir, 'model.json'), 'r') as fr:
                self.m = model_from_json(fr.read(), custom_objects=keras_custom_objects)
        else:
            # temporary handling of obsolete models
            self.m = self.ti.model_6conv_3dense_legacy()
            self.config['properties'] = ['x', 'y', 'major', 'minor', 'angle_deg']
            # assert False, '{} or {} not found'.format(join(model_dir, 'model.yaml'), join(model_dir, 'model.json'))
        self.m.load_weights(join(model_dir, 'weights.h5'))
        self.predictions = ObjectsArray(self.config['properties'], self.config['num_objects'])

    def detect(self, img, xy):
        """
        Detect objects in interaction.

        :param img: input image
        :param xy: position of the interaction
        :return: dict, properties of the detected objects
                       keys: '0_x', '0_y', '0_angle_deg', .... 'n_x', 'n_y', 'n_angle_deg'
        """
        img_crop, delta_xy = safe_crop(img, xy, self.config['input_size_px'])
        # img_crop, delta_xy = safe_crop(img, xy, 200)
        # img_crop = self.ti.resize_images([img_crop], (224, 224, 3))[0]
        img = np.expand_dims(img_crop, 0).astype(np.float) / 255.
        # import matplotlib.pylab as plt
        # plt.imshow(img[0])
        pred = self.m.predict(img)
        pred_ = pred.copy().flatten()
        pred = self.ti.postprocess_predictions(pred)
        pred_dict = self.predictions.array_to_dict(pred)
        for i in range(self.config['num_objects']):
            pred_dict['{}_x'.format(i)] += delta_xy[0]
            pred_dict['{}_y'.format(i)] += delta_xy[1]
        return pred_dict, pred_, delta_xy

    def region_to_dict(self, r):
        return({
            '0_x': r.centroid()[1],
            '0_y': r.centroid()[0],
            '0_major': 2 * r.major_axis_,
            '0_minor': 2 * r.minor_axis_,
            '0_angle_deg': np.rad2deg(r.theta_),
        })

    def detect_single(self, img, prev_detection):
        """
        Detect objects in interaction.

        :param img: input image
        :param xy: position of the interaction
        :return: dict, properties of the detected objects
                       keys: '0_x', '0_y', '0_angle_deg', .... 'n_x', 'n_y', 'n_angle_deg'
        """
        timg = TransformableRegion()
        prev_xy = (prev_detection['0_x'], prev_detection['0_y'])
        timg.rotate(-prev_detection['0_angle_deg'], prev_xy[::-1])
        timg.set_img(img)
        img_rotated = timg.get_img()

        img_crop, delta_xy = safe_crop(img_rotated, prev_xy, self.config['input_size_px'])
        img = np.expand_dims(img_crop, 0).astype(np.float) / 255.
        pred = self.m.predict(img)
        pred_ = pred.copy().flatten()
        pred = self.ti.postprocess_predictions(pred)
        # pred_dict = {
        #     '0_x':
        # }
        pred_dict = self.predictions.array_to_dict(pred)
        pred_dict['0_x'] += delta_xy[0]
        pred_dict['0_y'] += delta_xy[1]
        pred_dict['0_major'] = prev_detection['0_major']
        pred_dict['0_minor'] = prev_detection['0_minor']
        # timg.get_transformed_angle()
        # timg.get_transformed_coords()
        pred_dict['0_angle_deg'] = -timg.get_inverse_transformed_angle(pred_dict['0_angle_deg'])
        pred_dict['0_x'], pred_dict['0_y'] = timg.get_inverse_transformed_coords(
            np.array((pred_dict['0_x'], pred_dict['0_y'])))

        return pred_dict, pred_, delta_xy

    def draw_detections(self, img, detections):
        ax = show_prediction(img, self.config['num_objects'], detections)
        return ax

    def track(self, detections):
        """
        Find tracks in detections.

        :param detections: see detect() return value
        :return:
        """
        def cost(prev, cur):
            """
            Cost for connecting two consecutive object positions.
            """
            w = self.TRACKING_COST_WEIGHT
            return w * np.sqrt((cur['x'] - prev['x']) ** 2 + (cur['y'] - prev['y']) ** 2) + \
                   (1 - w) * angle_absolute_error_direction_agnostic(cur['angle_deg'], prev['angle_deg'])

        def object_from_dict(dct, idx):
            out = {}
            for key, val in dct.iteritems():
                i, prop = key.split('_', 1)
                i = int(i)
                if i == idx:
                    out[prop] = val
            return out

        def swap(dct):
            out = {}
            for key, val in dct.iteritems():
                i, prop = key.split('_', 1)
                if i == '0':
                    i = '1'
                else:
                    i = '0'
                out['{}_{}'.format(i, prop)] = val
            return out

        tracks = []
        costs = []
        costs_columns = ['0', '0_swap', '1', '1_swap']
        for i, cur in enumerate(detections):
            if i == 0:
                tracks.append(cur)
                costs.append((np.nan, np.nan, np.nan, np.nan))
            else:
                current_costs = {(j, k): cost(object_from_dict(prev, j), object_from_dict(cur, k)) for j, k in
                                 ((0, 0), (1, 1), (0, 1), (1, 0))}
                if current_costs[(0, 0)] + current_costs[(1, 1)] < current_costs[(0, 1)] + current_costs[(1, 0)]:
                    tracks.append(cur)
                    costs.append(
                        (current_costs[(0, 0)], current_costs[(0, 1)], current_costs[(1, 1)], current_costs[(1, 0)]))
                else:
                    tracks.append(swap(cur))
                    costs.append(
                        (current_costs[(0, 1)], current_costs[(0, 0)], current_costs[(1, 0)], current_costs[(1, 1)]))
            prev = tracks[-1]

        pred_fixed = pd.DataFrame(tracks, columns=self.predictions.columns())
        costs = pd.DataFrame(costs, columns=costs_columns)
        min_swap_diff = min((costs['1_swap'] - costs['1']).abs().min(), (costs['0_swap'] - costs['0']).abs().min())
        confidence = expit((min_swap_diff - self.TRACKING_CONFIDENCE_LOC) * self.TRACKING_CONFIDENCE_SCALE)
        return pred_fixed, confidence, costs

    def write_interaction_movie(self, images, tracks, out_filename):
        _, tmp_file = tempfile.mkstemp(suffix='.png')
        out_images = []
        if isinstance(tracks, pd.DataFrame):
            tracks_items = [t for _, t in tracks.iterrows()]
        elif isinstance(tracks, list):
            tracks_items = tracks
        for img, props in tqdm(zip(images, tracks_items), desc='interaction movie'):
    #         for obj_i in range(2):
    #             predictions['{}_major'.format(obj_i)] = 60
    #             predictions['{}_minor'.format(obj_i)] = 15
            save_prediction_img(tmp_file, 2, img, pred=props, scale=0.8)
            out_images.append(imageio.imread(tmp_file))

        imageio.mimwrite(out_filename, out_images)
        os.unlink(tmp_file)


def train(project_dir, output_model_dir, temp_dataset_dir, num_objects=2, delete_data=False, video_file=None):  # model_dir=None,
    """

    Creates interaction detection model

    :param project_dir:
    :param output_model_dir: the model weights are saved in weights.h5
    :param temp_dataset_dir:
    :param model_dir:
    :param num_objects:
    :return:
    """
    dg = DataGenerator()
    dg.write_synthetized_interactions(project_dir, 1080, num_objects,
                                      join(temp_dataset_dir, 'train.csv'), 10,
                                      join(temp_dataset_dir, 'images.h5'), 'train',
                                      write_masks=True, video_file=video_file)
    dg.write_synthetized_interactions(project_dir, 100, num_objects,
                                      join(temp_dataset_dir, 'test.csv'), 'random',
                                      join(temp_dataset_dir, 'images.h5'), 'test',
                                      write_masks=True, video_file=video_file)
    ti = TrainInteractions(num_objects, num_input_layers=1)
    ti.train_and_evaluate(temp_dataset_dir, 0.42, 10, model='mobilenet', input_layers=1,
                          experiment_dir=output_model_dir)
    if delete_data:
        remove(join(temp_dataset_dir, 'images.h5'))
        remove(join(temp_dataset_dir, 'train.csv'))
        remove(join(temp_dataset_dir, 'test.csv'))


def detect_and_visualize(model_dir, in_img, x, y, out_img=None):
    """
    Run detector of objects in interaction on single image.

    :param model_dir: directory with trained detector model (config.yaml, model.yaml)
    :param in_img: input image file
    :param x: position of the interaction
    :param y: position of the interaction
    :param out_img: output image file with results visualization
    """
    img = imageio.imread(in_img)
    detector = InteractionDetector(model_dir)
    detections = detector.detect(img, (x, y))
    for obj_i in range(2):
        detections['{}_major'.format(obj_i)] = 60
        detections['{}_minor'.format(obj_i)] = 15
    detector.draw_detections(img, detections)
    if out_img is None:
        root, ext = os.path.splitext(in_img)
        out_img = root + '_detections' + ext
    save_prediction_img(out_img, 2, img, detections)


if __name__ == '__main__':
    fire.Fire({
      'detect_and_visualize': detect_and_visualize,
      'train': train,
    })
