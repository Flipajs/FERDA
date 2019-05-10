import pickle
import string
import time
import numpy as np
import os
from os.path import join
import tqdm
import json
import errno

from core.graph.solver import Solver
from core.graph.graph_manager import GraphManager
from core.graph.chunk_manager import ChunkManager
from core.region.region_manager import RegionManager
from core.log import Log
from core.project.mser_parameters import MSERParameters
from core.project.other_parameters import OtherParameters
from core.project.solver_parameters import SolverParameters
from core.config import config
from utils.img_manager import ImgManager
import jsonpickle


class ProjectNotFoundError(OSError):
    def __init__(self, *args, **kwargs):
        super(ProjectNotFoundError, self).__init__(*args, **kwargs)


class Project(object):
    """
    This class encapsulates one experiment using FERDA
    """
    def __init__(self, project_directory=None, video_file=None):
        self.working_directory = ''

        self.name = ''
        self.description = ''
        self.video_paths = []
        self.video_start_t = 0
        self.video_end_t = None  # inclusive
        self.date_created = -1
        self.date_last_modification = -1
        self.next_processing_stage = 'segmentation'  # initialization, segmentation, assembly, cardinality_classification,
                                           # re_identification,


        self.video_crop_model = None
        # {'x1': ..., 'x2':..., 'y1':..., 'y2':...}
        # cropped image: img[cm['y1']:cm['y2'], cm['x1']:cm['x2']]
        self.stats = None  # TODO: review if needed
        self.mser_parameters = MSERParameters()  # TODO: change to dict, initialized from config
        self.other_parameters = OtherParameters()  # TODO: change to dict, initialized from config
        self.solver_parameters = SolverParameters()  # TODO: change to dict, initialized from config
        self.version = "3.1.0"

        self.rm = RegionManager()
        self.chm = ChunkManager()
        self.gm = GraphManager()
        self._solver = Solver(self)
        set_managers(self, self.rm, self.chm, self.gm)

        self.arena_model = None
        self.img_manager = None
        self.region_cardinality_classifier = None
        self.bg_model = None

        if project_directory is not None:
            self.load(project_directory, video_file)


        # TODO: remove
        self.log = Log()
        self.snapshot_id = 0
        self.active_snapshot = -1
        # so for new projects it is True as default but it will still works for the older ones without this support...
        self.other_parameters.store_area_info = True
        self.color_manager = None
        self.use_colormarks = False
        self.colormarks_model = None
        self.animals = None

    def version_is_le(self, ver):
        # returns true if self.version is lower or equal then version
        l1 = string.split(self.version, '.')
        l2 = string.split(ver, '.')

        for a, b in zip(l1, l2):
            if int(a) > int(b):
                return False

        return True

    def __getstate__(self):
        d = self.__dict__.copy()
        if 'working_directory' in d:
            del d['working_directory']
        # saved separately
        del d['chm']
        del d['rm']
        del d['gm']
        del d['region_cardinality_classifier']

        # reinitialized
        del d['_solver']
        del d['bg_model']
        del d['color_manager']
        return d

    def set_rm(self, rm):
        self.rm = rm
        self.gm.rm = rm

    def save(self, directory=None):
        with tqdm.tqdm(total=5, desc='saving project') as pbar:
            if directory is None:
                directory = self.working_directory
            try:
                os.makedirs(directory)
            except OSError:
                pass

            open(join(directory, 'project.json'), 'w').write(jsonpickle.encode(self, keys=True, warn=True))
            if self.arena_model is not None:
                self.arena_model.save_mask(join(directory, 'mask.png'))
            pbar.update()

            if self.gm is not None:
                self.gm.save(directory)
            pbar.update()
            if self.rm is not None:
                self.rm.save(directory)
            pbar.update()
            if self.chm is not None:
                self.chm.save(directory)
            pbar.update()
            # TODO: remove after automatic cardinality estimation is integrated
            if self.region_cardinality_classifier:
                with open(join(directory, 'region_cardinality_clustering.pkl'), 'wb') as f:
                    pickle.dump(self.region_cardinality_classifier, f)
            pbar.update()

    def load(self, directory, video_file=None,
             regions_optional=False, graph_optional=False, tracklets_optional=False):
        with tqdm.tqdm(total=5, desc='loading project') as pbar:
            self.__dict__.update(jsonpickle.decode(open(join(directory, 'project.json'), 'r').read(), keys=True).__dict__)
            if self.arena_model is not None:
                self.arena_model.load_mask(join(directory, 'mask.png'))
            # check for video file
            if video_file is not None:
                self.video_paths = video_file
            self.img_manager.set_project(self)
            pbar.update()
            pbar.set_description('loading regions')
            if os.path.exists(join(directory, 'regions.csv')) and os.path.exists(join(directory, 'regions.h5')):
                self.rm = RegionManager.from_dir(directory)
            elif not regions_optional:
                raise ProjectNotFoundError('regions.{csv,h5} not found and regions not optional')
            else:
                self.rm = RegionManager()
            pbar.update()
            pbar.set_description('loading graph')
            if os.path.exists(join(directory, 'graph.json')):
                self.gm = GraphManager.from_dir(directory)
            elif not graph_optional:
                raise ProjectNotFoundError('graph.json not found and graph not optional')
            else:
                self.gm = GraphManager()
            pbar.update()
            pbar.set_description('loading tracklets')
            if os.path.exists(join(directory, 'tracklets.json')):
                self.chm = ChunkManager.from_dir(directory)
            elif not tracklets_optional:
                raise ProjectNotFoundError('tracklets.json not found and tracklets not optional')
            else:
                self.chm = ChunkManager()
            pbar.update()
            pbar.set_description('loading cardinality classifier')
            cardinality_model_filename = join(directory, 'region_cardinality_clustering.pkl')
            if os.path.exists(cardinality_model_filename):
                self.region_cardinality_classifier = pickle.load(open(cardinality_model_filename, 'r'))
            pbar.update()

            self.solver = Solver(self)
            set_managers(self, self.rm, self.chm, self.gm)
            self.working_directory = directory

    @classmethod
    def from_dir(cls, directory, video_file=None,
             regions_optional=False, graph_optional=False, tracklets_optional=False):
        project = cls()
        project.load(directory, video_file, regions_optional, graph_optional, tracklets_optional)
        return project

    @property
    def solver(self):
        return self._solver

    @solver.setter
    def solver(self, solver):
        self._solver = solver
        if self.gm is not None:
            self.gm.assignment_score = self._solver.assignment_score

    def video_exists(self):
        if isinstance(self.video_paths, list):
            for path in self.video_paths:
                if os.path.isfile(path):
                    return True
            return False
        return os.path.isfile(self.video_paths)

    def get_video_manager(self):
        from utils.video_manager import get_auto_video_manager
        return get_auto_video_manager(self)

    def num_frames(self):
        return self.get_video_manager().total_frame_count()

    def get_results_trajectories(self):
        """
        Return resulting single id centroid trajectories.

        :return: ndarray, shape=(n_frames, n_animals, 2); coordinates are in yx order, nan when id not present
        """
        assert self.video_start_t != -1
        n_frames = self.video_end_t + 1
        results = np.ones(shape=(n_frames, len(self.animals), 2)) * np.nan
        for frame in tqdm.tqdm(range(self.video_start_t, n_frames), desc='gathering trajectories'):
            for t in self.chm.tracklets_in_frame(frame - self.video_start_t):
                if len(t.P) == 1:
                    id_ = list(t.P)[0]
                    if id_ >= len(self.animals):
                        import warnings
                        warnings.warn("id_ > num animals t_id: {} id: {}".format(t.id(), id_))
                        continue
                    results[frame, id_] = self.rm[t.r_id_in_t(frame - self.video_start_t)].centroid()  # yx

        if self.video_crop_model is not None:
            results[:, :, 0] += self.video_crop_model['y1']
            results[:, :, 1] += self.video_crop_model['x1']

        return results

    def fix_regions_orientation(self):
        from core.graph.region_chunk import RegionChunk
        n_swaps = 0
        for tracklet in tqdm.tqdm(self.chm.chunk_gen(), total=len(self.chm),
                                  desc='fixing orientation of regions in single tracklets'):
            if tracklet.is_single():
                n_swaps += RegionChunk(tracklet, self.gm, self.rm).fix_regions_orientation()
        return n_swaps


def dummy_project():
    from core.classes_stats import dummy_classes_stats
    from core.region.region_manager import RegionManager

    p = Project()
    p.stats = dummy_classes_stats()
    p.rm = RegionManager()

    return p


def set_managers(project=None, rm=None, chm=None, gm=None):
    if rm is not None:
        if project is not None:
            project.rm = rm
        if gm is not None:
            gm.rm = rm
    if chm is not None:
        if project is not None:
            project.chm = chm
        if gm is not None:
            gm.chm = chm
    if gm is not None:
        if project is not None:
            project.gm = gm
            gm.assignment_score = project.solver.assignment_score
        if chm is not None:
            chm.set_graph_manager(gm)


if __name__ == "__main__":
    # experiments...
    p = Project('../projects/2_temp/190404_Sowbug3_cut_open')
    # t = p.chm[4]
    # from core.graph.region_chunk import RegionChunk
    # import matplotlib.pylab as plt
    # rt = RegionChunk(t, p.gm, p.rm)
    # rt.draw()
    # plt.show()


