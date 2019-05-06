from __future__ import print_function
import cPickle as pickle
import os.path
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from utils.drawing.points import draw_points
from collections import OrderedDict
import tqdm

region_features = OrderedDict([
    ('area', lambda r: r.area()),
    ('major axis length', lambda r: r.ellipse_major_axis_length()),
    ('minor axis length', lambda r: r.ellipse_minor_axis_length()),
    ('min intensity', lambda r: r.min_intensity_),
    ('max intensity', lambda r: r.max_intensity_),
    ('margin', lambda r: r.margin_),
    ('contour length', lambda r: len(r.contour())),
    ('ellipse area ratio', lambda r: r.ellipse_area_ratio()),
    ])

labels = ['single', 'multi', 'noise', 'part']

# id_to_label = dict(enumerate(labels))
# label_to_id = dict([x[::-1] for x in enumerate(labels)])


class RegionSample(object):
    def __init__(self, region, frame_image=None):
        self.region = region
        self.label = None
        self.confidence = None
        self.widget = None

        self.features = self.compute_features(self.region)
        self.image = None
        if frame_image is not None:
            self._draw_region(frame_image)

    def __lt__(self, other):
        return self.confidence < other.confidence

    @staticmethod
    def compute_features(region):
        return [fun(region) for fun in region_features.values()]

    def _draw_region(self, frame_img):
        img_copy = frame_img.copy()
        draw_points(img_copy, self.region.contour(), color=(255, 0, 0, 255))
        draw_points(img_copy, self.region.pts(), color=(255, 0, 0, 20))
        roi = self.region.roi().safe_expand(30, img_copy)
        self.image = img_copy[roi.slices()].copy()


class RegionCardinality:
    def __init__(self):
        # self.storage_directory = os.path.join(project.working_directory, 'temp')
        # self.samples_filename = os.path.join(self.storage_directory, 'region_cardinality_samples.pkl')

        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

        self.active_features_mask = np.ones(len(region_features), dtype=np.bool)
        self.classifier = KNeighborsClassifier(n_neighbors=1)

    def init_scaler(self, samples):
        X = np.array([s.features for s in samples])
        self.scaler.fit(X)

    def gather_samples(self, n, project, progress_update_fun=None):
        # if len(self.project.chm) > 0:
        #     regions, vertices = self.project.chm.get_random_regions(n, self.project.gm)
        # else:
        samples = get_random_segmented_regions(n, project, progress_update_fun)
        X = np.array([s.features for s in samples])
        self.scaler.fit(X)
        return samples
        # self.data = np.array([s.features for s in self.samples])

    def gather_diverse_samples(self, n_gather, n_final, project, progress_update_fun=None):
        samples = get_random_segmented_regions(n_gather, project, progress_update_fun, get_diverse_samples=True)
        X = np.array([s.features for s in samples])
        self.scaler.fit(X)
        X_norm = self.scaler.transform(X)

        from sklearn.cluster.k_means_ import _k_init
        X_norm_active = X_norm[:, self.active_features_mask]
        X_selected = _k_init(X_norm_active, n_final,
                             x_squared_norms=(X_norm_active ** 2).sum(axis=1),
                             random_state=np.random.RandomState())
        indices = [np.argmin(((X_norm_active - x) ** 2).sum(axis=1)) for x in X_selected]
        self.scaler.fit(X[indices])
        return [samples[i] for i in indices]

    def set_active_features(self, features_mask):
        self.active_features_mask = features_mask

    def train(self, labeled_samples):
        if labeled_samples:
            X = np.array([s.features for s in labeled_samples])
            X = self.scaler.transform(X)
            labels = [s.label for s in labeled_samples]
            self.label_encoder.fit(labels)
            X_active = X[:, self.active_features_mask]
            y = self.label_encoder.transform(labels)
            self.classifier.fit(X_active, y)

    def classify_samples(self, samples):
        assert len(samples) > 0
        X = np.array([s.features for s in samples])
        X = self.scaler.transform(np.atleast_2d(X))
        X_active = X[:, self.active_features_mask]
        y = self.classifier.predict(X_active)
        p = self.classifier.predict_proba(X_active)
        for sample, label_id, class_probs in zip(samples, y, p):
            sample.label = self.label_encoder.inverse_transform(label_id)
            assert class_probs[label_id] == max(class_probs)
            sample.confidence = class_probs[label_id]
        return samples

    def classify(self, region):
        features = np.atleast_2d(RegionSample.compute_features(region))
        features_scaled = self.scaler.transform(features)
        return self.label_encoder.inverse_transform(
            self.classifier.predict(features_scaled[:, self.active_features_mask]))[0]

    def classify_tracklet(self, region_chunk):
        """
        Classify cardinality of a tracklet.

        :param region_chunk: core.graph.region_chunk.RegionChunk
        :return: int, label 0, 1, 2, 3, 4
        """
        freq = np.zeros(len(labels), dtype=np.int)
        for region in region_chunk.regions_gen():
            label = self.classify(region)
            freq[labels.index(label)] += 1

        return np.argmax(freq)

    def classify_project(self, project):
        """
        Classify tracklet cardinality for all tracklets in a project.

        :param project: core.project.Project
        """
        from core.graph.region_chunk import RegionChunk

        for i, tracklet in enumerate(tqdm.tqdm(project.chm.chunk_gen(),
                                               total=len(project.chm),
                                               desc='Classifying tracklets (single/multi/part/no-ID)')):
            region_chunk = RegionChunk(tracklet, project.gm, project.rm)
            tracklet.segmentation_class = self.classify_tracklet(region_chunk)


def is_project_cardinality_classified(project):
    if not project.chm:
        return False
    for tracklet in project.chm.chunk_gen():
        if tracklet.segmentation_class == -1:
            return False
    return True


def get_random_segmented_regions(n, project, progress_update_fun=None,
                                 get_diverse_samples=False, complete_frames=False):
    from utils.img import prepare_for_segmentation
    from core.region.mser import get_filtered_regions

    if get_diverse_samples:
        assert project.animals is not None
    samples = []
    vm = project.get_video_manager()

    if progress_update_fun is None:
        pbar = tqdm.tqdm(total=n, desc='segmenting regions for learning cardinality classification')
        progress_update_fun = pbar.update

    used_frames = []
    while len(samples) < n:
        img_rgb, frame = vm.random_frame()
        if frame in used_frames:
            continue
        used_frames.append(frame)
        img = prepare_for_segmentation(img_rgb, project)
        regions = get_filtered_regions(img, project, frame)
        if get_diverse_samples and len(regions) == len(project.animals):
            # TODO: guard against infinite loop when not enough diverse samples
            continue
        samples.extend([RegionSample(r, img_rgb) for r in regions])
        progress_update_fun(len(regions))

    if complete_frames:
        return samples
    else:
        return samples[:n]


if __name__ == '__main__':
    from core.project.project import Project

    p = Project()
    p.load('/home/matej/prace/ferda/projects/Sowbug_deleteme2/')
    # p.load_semistate('/Users/flipajs/Documents/wd/FERDA/zebrafish_playground')

    p.region_cardinality_classifier.classify_project(p)
