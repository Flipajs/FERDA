__author__ = 'filip@naiser.cz'
import cPickle as pickle
import string
import time
import numpy as np
import os
from os.path import join
import tqdm
import json

from core.graph.solver import Solver
from core.log import Log
from core.project.mser_parameters import MSERParameters
from core.project.other_parameters import OtherParameters
from core.project.solver_parameters import SolverParameters
from core.config import config
from utils.img_manager import ImgManager


class Project:
    """
    This class encapsulates one experiment using FERDA
    """
    def __init__(self, path=None):
        self.working_directory = ''

        self.name = ''
        self.description = ''
        self.video_paths = []
        self.video_start_t = 0
        self.video_end_t = None
        self.date_created = -1
        self.date_last_modification = -1
        self.video_crop_model = None
        self.classes = None
        self.groups = None
        self.stats = None
        self.mser_parameters = MSERParameters()
        self.other_parameters = OtherParameters()
        self.solver_parameters = SolverParameters()
        self.use_colormarks = False
        self.colormarks_model = None
        self.version = "3.1.0"

        # REGION MANAGER
        self.rm = None
        # CHUNK MANAGER
        self.chm = None
        # GRAPH MANAGER
        self.gm = None

        self.arena_model = None
        self.solver = None
        self.img_manager = None
        self.region_cardinality_classifier = None
        self.bg_model = None
        self.animals = None
        self.color_manager = None
        self.log = Log()

        self.snapshot_id = 0
        self.active_snapshot = -1

        # so for new projects it is True as default but it will still works for the older ones without this support...
        self.other_parameters.store_area_info = True

        if path is not None:
            self.load(path)

    def version_is_le(self, ver):
        # returns true if self.version is lower or equal then version
        l1 = string.split(self.version, '.')
        l2 = string.split(ver, '.')

        for a, b in zip(l1, l2):
            if int(a) > int(b):
                return False

        return True

    # def save_project_file_(self, dir_path=None):
    #     if dir_path is None:
    #         dir_path = self.working_directory
    #
    #     p = Project()
    #     p.name = self.name
    #     p.description = self.description
    #     p.video_paths = self.video_paths
    #     p.working_directory = self.working_directory
    #     p.video_start_t = self.video_start_t
    #     p.video_end_t = self.video_end_t
    #
    #     p.mser_parameters = self.mser_parameters
    #     p.other_parameters = self.other_parameters
    #     p.solver_parameters = self.solver_parameters
    #     p.version = self.version
    #
    #     p.date_created = self.date_created
    #     p.use_colormarks = self.use_colormarks
    #     p.colormarks_model = self.colormarks_model
    #     p.color_manager = self.color_manager
    #
    #     p.date_last_modifiaction = time.time()
    #
    #     p.snapshot_id = self.snapshot_id
    #     p.active_snapshot = self.active_snapshot
    #
    #     try:
    #         p.GT_file = self.GT_file
    #     except AttributeError:
    #         pass
    #
    #     try:
    #         p.video_crop_model = self.video_crop_model
    #     except AttributeError:
    #         pass
    #
    #     self.project_file = join(dir_path, '{}.fproj'.format(self.name))
    #     with open(self.project_file, 'wb') as f:
    #         pickle.dump(p.__dict__, f, 2)

    def to_json(self):
        d = self.__dict__.copy()
        del d['working_directory']
        del d['rm']
        del d['chm']
        del d['gm']
        del d['arena_model']
        del d['solver']
        del d['img_manager']
        del d['region_cardinality_classifier']
        del d['bg_model']
        del d['color_manager']
        del d['log']
        return json.dumps(d, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def from_json(self, json_content):
        d = json.loads(json_content)
        d['mser_parameters'] = MSERParameters(d['mser_parameters'])
        d['solver_parameters'] = SolverParameters(d['solver_parameters'])
        d['other_parameters'] = OtherParameters(d['other_parameters'])
        self.__dict__.update(d)

    def from_pkl(self, pkl_file):
        with open(pkl_file, 'rb') as f:
            tmp_dict = pickle.load(f)
        del tmp_dict['working_directory']
        self.__dict__.update(tmp_dict)

    def save(self, path=None):
        if path is None:
            path = self.working_directory

        # BG MODEL
        try:
            from core.bg_model.model import Model
            if self.bg_model:
                if isinstance(self.bg_model, Model):
                    if self.bg_model.is_computed():
                        self.bg_model = self.bg_model.get_model()

                        with open(path+ '/bg_model.pkl', 'wb') as f:
                            pickle.dump(self.bg_model, f)
                else:
                    with open(path+ '/bg_model.pkl', 'wb') as f:
                        pickle.dump(self.bg_model, f)
        except:
            pass

        # ARENA MODEL
        if self.arena_model:
            with open(path+ '/arena_model.pkl', 'wb') as f:
                pickle.dump(self.arena_model, f)

        # CLASSES
        if self.classes:
            with open(path+ '/classes.pkl', 'wb') as f:
                pickle.dump(self.classes, f)

        # GROUPS
        if self.groups:
            with open(path+ '/groups.pkl', 'wb') as f:
                pickle.dump(self.groups, f)

        # ANIMALS
        if self.animals:
            with open(path+ '/animals.pkl', 'wb') as f:
                pickle.dump(self.animals, f)

        # STATS
        if self.stats:
            with open(path+ '/stats.pkl', 'wb') as f:
                pickle.dump(self.stats, f)

        if self.region_cardinality_classifier:
            with open(join(path, 'region_cardinality_clustering.pkl'), 'wb') as f:
                pickle.dump(self.region_cardinality_classifier, f)

        # # Region Manager
        # if self.rm:
        #     with open(self.working_directory+'/region_manager.pkl', 'wb') as f:
        #         pickle.dump(self.rm, f, -1)

        self.save_chm_(path + '/chunk_manager.pkl')

        self.save_gm_(path + '/graph_manager.pkl')

        # self.save_project_file_(path)
        with open(join(path, 'project.json'), 'w') as fw:
            fw.write(self.to_json())

    def save_gm_(self, file_path):
        print "saving GM"
        # Graph Manager
        if self.gm:
            self.gm.project = None
            self.gm.rm = None
            ac = self.gm.assignment_score
            self.gm.assignment_score = None

            with open(file_path, 'wb') as f:
                pickle.dump(self.gm, f, -1)

            self.gm.project = self
            self.gm.rm = self.rm
            self.gm.assignment_score = ac

    def save_chm_(self, file_path):
        print "saving chm"
        import os

        try:
            os.rename(file_path, file_path+'__')
        except:
            pass

        # Chunk Manager
        if self.chm:
            for _, ch in self.chm.chunks_.iteritems():
                ch.project = None

            with open(file_path, 'wb') as f:
                pickle.dump(self.chm, f, -1)

    def save_snapshot(self):
        # print self.snapshot_id, self.active_snapshot
        import os

        if not os.path.exists(self.working_directory + '/.auto_save'):
            os.mkdir(self.working_directory + '/.auto_save')

        self.save_chm_(self.working_directory+'/.auto_save/'+str(self.snapshot_id)+'__chunk_manager.pkl')

        self.save_gm_(self.working_directory+'/.auto_save/'+str(self.snapshot_id)+'__graph_manager.pkl')

        self.snapshot_id += 1
        self.active_snapshot = -1

    def load_semistate(self, path, state='isolation_score', one_vertex_chunk=False, update_t_nodes=False):
        self.load(path)

        with open(self.working_directory + '/temp/'+state+'.pkl', 'rb') as f:
            up = pickle.Unpickler(f)
            self.gm.g = up.load()
            up.load()
            self.chm = up.load()

        from core.region.region_manager import RegionManager
        self.rm = RegionManager(self.working_directory + '/temp', db_name='part0_rm.sqlite3')
        self.gm.rm = self.rm

        if one_vertex_chunk:
            self.chm.add_single_vertices_chunks(self)

        if update_t_nodes:
            self.gm.update_nodes_in_t_refs()

    def save_semistate(self, state):
        with open(self.working_directory + '/temp/'+state+'.pkl', 'wb') as f:
            p = pickle.Pickler(f)
            p.dump(self.gm.g)
            p.dump(None)
            p.dump(self.chm)

    # @staticmethod
    # def get_project_dir_and_file(filename_or_dir):
    #     if os.path.splitext(filename_or_dir)[1] == '.json':
    #         filename = filename_or_dir
    #         dirname = os.path.dirname(filename)
    #     else:
    #         dirname = filename_or_dir
    #         filename = os.path.join(dirname, 'project.json')
    #     return dirname, filename

    def load(self, path, snapshot=None, lightweight=False, video_file=None):
        self.working_directory = path  # , self.project_file = self.get_project_dir_and_file(path)

        project_json = join(self.working_directory, 'project.json')
        project_fproj = join(self.working_directory, 'project.fproj')
        if os.path.isfile(project_json):
            with open(project_json, 'r') as fr:
                self.from_json(fr.read())
        elif os.path.isfile(project_fproj):
            self.from_pkl(project_fproj)
        else:
            assert False, 'no project file found at {} or {}'.format(project_json, project_fproj)

        # check for video file
        if video_file is not None:
            self.video_paths = video_file

        # BG MODEL
        try:
            with open(self.working_directory+'/bg_model.pkl', 'rb') as f:
                self.bg_model = pickle.load(f)
        except:
            pass

        # ARENA MODEL
        try:
            with open(self.working_directory+'/arena_model.pkl', 'rb') as f:
                self.arena_model = pickle.load(f)
        except:
            pass

        # # CLASSES
        # try:
        #     with open(self.working_directory+'/classes.pkl', 'rb') as f:
        #         self.classes = pickle.load(f)
        # except:
        #     pass

        # # GROUPS
        # try:
        #     with open(self.working_directory+'/groups.pkl', 'rb') as f:
        #         self.groups = pickle.load(f)
        # except:
        #     pass

        # ANIMALS
        try:
            with open(self.working_directory+'/animals.pkl', 'rb') as f:
                self.animals = pickle.load(f)
        except:
            pass

        # STATS
        try:
            with open(self.working_directory+'/stats.pkl', 'rb') as f:
                self.stats = pickle.load(f)
        except:
            pass

        # SEGMENTATION MODEL (core.segmentation_helper)
        try:
            with open(self.working_directory+'/segmentation_model.pkl', 'rb') as f:
                self.segmentation_model = pickle.load(f)
        except:
            pass

        try:
            with open(join(self.working_directory, 'region_cardinality_clustering.pkl'), 'rb') as f:
                self.region_cardinality_classifier = pickle.load(f)
        except:
            pass

        # # Region Manager
        # try:
        #     with open(self.working_directory+'/region_manager.pkl', 'rb') as f:
        #         self.rm = pickle.load(f)
        # except:
        #     pass

        self.load_snapshot(snapshot)

        # if not lightweight:
        # SAVED CORRECTION PROGRESS
        try:
            with open(self.working_directory+'/progress_save.pkl', 'rb') as f:
                up = pickle.Unpickler(f)
                g = up.load()
                log = up.load()
                if isinstance(log, list):
                    log = Log()

                ignored_nodes = {}
                try:
                    ignored_nodes = up.load()
                except:
                    pass

                # self.solver = Solver(self)
                # self.gm.g = g
                # self.solver.g = g
                self.solver.ignored_nodes = ignored_nodes
                # solver.update_nodes_in_t_refs()
                self.log = log

                if self.gm:
                    self.gm.assignment_score = self.solver.assignment_score
        except:
            pass

        # reconnect...
        if not self.gm:
            from core.graph.graph_manager import GraphManager
            self.gm = GraphManager(self, None)

        from core.region.region_manager import RegionManager
        self.solver = Solver(self)
        self.gm.assignment_score = self.solver.assignment_score

        self.rm = RegionManager(db_wd=self.working_directory,
                                cache_size_limit=config['cache']['region_manager_num_of_instances'])

        self.gm.project = self
        self.gm.rm = self.rm
        # self.gm.update_nodes_in_t_refs()

        if not lightweight:
            # fix itree in chm...
            if self.chm is not None and self.gm is not None and self.rm is not None:
                if not hasattr(self.chm, 'itree'):
                    from libs.intervaltree.intervaltree import IntervalTree
                    self.chm.itree = IntervalTree()
                    self.chm.eps1 = 0.01
                    self.chm.eps2 = 0.1

                    for ch in self.chm.chunk_gen():
                        self.chm._add_ch_itree(ch, self.gm)

                for ch in self.chm.chunk_gen():
                    if hasattr(ch, 'color') and ch.color is not None:
                        break
                    ch.set_random_color()

                for ch in self.chm.chunk_gen():
                    if hasattr(ch, 'N'):
                        break

                        ch.N = set()
                        ch.P = set()

                self.save()

        self.img_manager = ImgManager(self, max_num_of_instances=500)

        self.active_snapshot = -1

        # if self.chm is not None:
        #     self.solver.one2one(check_tclass=True)

    def video_exists(self):
        if isinstance(self.video_paths, list):
            for path in self.video_paths:
                if os.path.isfile(path):
                    return True
            return False
        return os.path.isfile(self.video_paths)

    def load_snapshot(self, snapshot):
        chm_path = self.working_directory+'/chunk_manager.pkl'
        gm_path = self.working_directory+'/graph_manager.pkl'

        if snapshot:
            chm_path = snapshot['chm']
            gm_path = snapshot['gm']

        # Chunk Manager
        try:
            with open(chm_path, 'rb') as f:
                self.chm = pickle.load(f)
        except Exception as e:
            print e
            print "CHM not loaded"
            pass

        # Graph Manager
        try:
            with open(gm_path, 'rb') as f:
                self.gm = pickle.load(f)
                self.gm.project = self
                self.gm.assignment_score = self.solver.assignment_score
        except:
            pass

        self.img_manager = ImgManager(self, max_num_of_instances=500)

    def snapshot_undo(self):
        if self.active_snapshot < 0:
            self.active_snapshot = self.snapshot_id - 2
        else:
            self.active_snapshot -= 1

        if self.active_snapshot < 0:
            print "No more undo possible!"

        # print "UNDO", self.snapshot_id, self.active_snapshot

        self.load_snapshot({'chm': self.working_directory+'/.auto_save/'+str(self.active_snapshot)+'__chunk_manager.pkl',
                           'gm': self.working_directory+'/.auto_save/'+str(self.active_snapshot)+'__graph_manager.pkl'})

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
        n_frames = self.video_end_t
        results = np.ones(shape=(n_frames, len(self.animals), 2)) * np.nan
        for frame in tqdm.tqdm(range(self.video_start_t, n_frames), desc='gathering trajectories'):
            for t in self.chm.tracklets_in_frame(frame - self.video_start_t):
                if len(t.P) == 1:
                    id_ = list(t.P)[0]
                    if id_ >= len(self.animals):
                        import warnings
                        warnings.warn("id_ > num animals t_id: {} id: {}".format(t.id(), id_))
                        continue
                    results[frame, id_] = self.rm[t.r_id_in_t(frame - self.video_start_t, self.gm)].centroid()  # yx

        if self.video_crop_model is not None:
            results[:, :, 0] += self.video_crop_model['y1']
            results[:, :, 1] += self.video_crop_model['x1']

        return results


def project_video_file_exists(project_dir):
    project = Project()
    with open(join(project_dir, 'project.json'), 'r') as fr:
        project.from_json(fr.read())
    for path in project.video_paths:
        if os.path.isfile(path):
            return True
    return False


def dummy_project():
    from core.classes_stats import dummy_classes_stats
    from core.region.region_manager import RegionManager

    p = Project()
    p.stats = dummy_classes_stats()
    p.rm = RegionManager()

    return p


if __name__ == "__main__":
    p = Project('/home/matej/prace/ferda/projects/630e88d5/cowbug-cut/')
    p.name = 'test'
    p.a = 20
    p.working_directory = '/home/flipajs/test'
    p.save()

    a = Project()
    a.load('/home/flipajs/test/test.pkl')
    print "Project name: ", a.name
