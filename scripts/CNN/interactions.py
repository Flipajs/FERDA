import os.path
from os.path import join
from keras.models import model_from_yaml, model_from_json
from utils.img import safe_crop
import yaml
from utils.objectsarray import ObjectsArray
from scripts.CNN.interactions_results import show_prediction
import keras.applications.mobilenet as mobilenet
import numpy as np
from scripts.CNN.train_interactions import TrainInteractions
import pandas as pd
import imageio
import tempfile
from tqdm import tqdm
from scripts.CNN.interactions_results import save_prediction_img
from scipy.special import expit


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
            'DepthwiseConv2D': mobilenet.DepthwiseConv2D
        }
        with open(join(model_dir, 'config.yaml'), 'r') as fr:
            self.config = yaml.load(fr)
        self.ti = TrainInteractions(self.config['num_objects'])  # TODO: remove dependency

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
        # img_crop, delta_xy = safe_crop(img, xy, self.config['input_size_px'])
        img_crop, delta_xy = safe_crop(img, xy, 200)
        img_crop = self.ti.resize_images([img_crop], (224, 224, 3))[0]
        img = np.expand_dims(img_crop, 0).astype(np.float) / 255.
        # import matplotlib.pylab as plt
        # plt.imshow(img[0])
        pred = self.m.predict(img)
        pred = self.ti.postprocess_predictions(pred)
        pred_dict = self.predictions.array_to_dict(pred)
        for i in range(self.config['num_objects']):
            pred_dict['{}_x'.format(i)] += delta_xy[0]
            pred_dict['{}_y'.format(i)] += delta_xy[1]
        return pred_dict

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
            w = self.TRACKING_COST_WEIGHT
            return w * np.sqrt((cur['x'] - prev['x']) ** 2 + (cur['y'] - prev['y']) ** 2) + \
                   (1 - w) * self.ti.angle_absolute_error_direction_agnostic(cur['angle_deg'], prev['angle_deg'])

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