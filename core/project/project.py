__author__ = 'filip@naiser.cz'

import cPickle as pickle
import string

from PyQt4 import QtCore

from core.bg_model.model import Model
from core.graph.solver import Solver
from core.log import Log
from core.project.mser_parameters import MSERParameters
from core.project.other_parameters import OtherParameters
from core.project.solver_parameters import SolverParameters
from utils.color_manager import ColorManager


class Project:
    """
    This class encapsulates one experiment using FERDA
    """
    def __init__(self):
        self.name = ''
        self.description = ''
        self.video_paths = []
        self.video_start_t = -1
        self.video_end_t = -1
        self.working_directory = ''
        self.date_created = -1
        self.date_last_modifiaction = -1

        # REGION MANAGER
        self.rm = None
        # CHUNK MANAGER
        self.chm = None
        # GRAPH MANAGER
        self.gm = None

        self.bg_model = None
        self.arena_model = None
        self.classes = None
        self.groups = None
        self.animals = None
        self.stats = None
        self.mser_parameters = MSERParameters()
        self.other_parameters = OtherParameters()
        self.solver_parameters = SolverParameters()
        self.color_manager = None
        self.log = Log()
        self.solver = None
        self.version = "3.0.0"

        # so for new projects it is True as default but it will still works for the older ones without this support...
        self.other_parameters.store_area_info = True

    def version_is_le(self, ver):
        # returns true if self.version is lower or equal then version
        l1 = string.split(self.version, '.')
        l2 = string.split(ver, '.')

        for a, b in zip(l1, l2):
            if int(a) > int(b):
                return False

        return True

    def save_project_file_(self):
        p = Project()
        p.name = self.name
        p.description = self.description
        p.video_paths = self.video_paths
        p.working_directory = self.working_directory
        p.video_start_t = self.video_start_t
        p.video_end_t = self.video_end_t

        p.mser_parameters = self.mser_parameters
        p.other_parameters = self.other_parameters
        p.solver_parameters = self.solver_parameters
        p.version = self.version

        p.date_created = self.date_created
        p.color_manager = self.color_manager
        import time
        p.date_last_modifiaction = time.time()

        with open(self.working_directory+'/'+self.name+'.fproj', 'wb') as f:
            pickle.dump(p.__dict__, f, 2)

    def save(self):
        # BG MODEL
        if self.bg_model:
            if isinstance(self.bg_model, Model):
                if self.bg_model.is_computed():
                    self.bg_model = self.bg_model.get_model()

                    with open(self.working_directory+'/bg_model.pkl', 'wb') as f:
                        pickle.dump(self.bg_model, f)
            else:
                with open(self.working_directory+'/bg_model.pkl', 'wb') as f:
                    pickle.dump(self.bg_model, f)

        # ARENA MODEL
        if self.arena_model:
            with open(self.working_directory+'/arena_model.pkl', 'wb') as f:
                pickle.dump(self.arena_model, f)

        # CLASSES
        if self.classes:
            with open(self.working_directory+'/classes.pkl', 'wb') as f:
                pickle.dump(self.classes, f)

        # GROUPS
        if self.groups:
            with open(self.working_directory+'/groups.pkl', 'wb') as f:
                pickle.dump(self.groups, f)

        # ANIMALS
        if self.animals:
            with open(self.working_directory+'/animals.pkl', 'wb') as f:
                pickle.dump(self.animals, f)

        # STATS
        if self.stats:
            with open(self.working_directory+'/stats.pkl', 'wb') as f:
                pickle.dump(self.stats, f)

        # # Region Manager
        # if self.rm:
        #     with open(self.working_directory+'/region_manager.pkl', 'wb') as f:
        #         pickle.dump(self.rm, f, -1)

        # Chunk Manager
        if self.chm:
            for _, ch in self.chm.chunks_.iteritems():
                ch.project = None

            with open(self.working_directory+'/chunk_manager.pkl', 'wb') as f:
                pickle.dump(self.chm, f, -1)

        # Graph Manager
        if self.gm:
            self.gm.project = None
            self.gm.rm = None
            ac = self.gm.assignment_score
            self.gm.assignment_score = None

            with open(self.working_directory+'/graph_manager.pkl', 'wb') as f:
                pickle.dump(self.gm, f, -1)

            self.gm.project = self
            self.gm.rm = self.rm
            self.gm.assignment_score = ac

        self.save_qsettings()

        self.save_project_file_()

    def save_qsettings(self):
        s = QtCore.QSettings('FERDA')
        settings = {}

        for k in s.allKeys():
            try:
                settings[str(k)] = str(s.value(k, 0, str))
            except:
                pass

        with open(self.working_directory+'/settings.pkl', 'wb') as f:
            pickle.dump(settings, f)

    def load_qsettings(self):
        with open(self.working_directory+'/settings.pkl', 'rb') as f:
            settings = pickle.load(f)
            qs = QtCore.QSettings('FERDA')
            qs.clear()

            for key, it in settings.iteritems():
                try:
                    qs.setValue(key, it)
                except:
                    pass

    def load(self, path):
        with open(path, 'rb') as f:
            tmp_dict = pickle.load(f)

        self.__dict__.update(tmp_dict)

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

        # CLASSES
        try:
            with open(self.working_directory+'/classes.pkl', 'rb') as f:
                self.classes = pickle.load(f)
        except:
            pass

        # GROUPS
        try:
            with open(self.working_directory+'/groups.pkl', 'rb') as f:
                self.groups = pickle.load(f)
        except:
            pass

        # ANIMALS
        try:
            with open(self.working_directory+'/animals.pkl', 'rb') as f:
                self.animals = pickle.load(f)
        except:
            pass

        # ANIMALS
        try:
            with open(self.working_directory+'/stats.pkl', 'rb') as f:
                self.stats = pickle.load(f)
        except:
            pass

        # SETTINGS
        try:
            self.load_qsettings()
        except:
            pass

        # Region Manager
        try:
            with open(self.working_directory+'/region_manager.pkl', 'rb') as f:
                self.rm = pickle.load(f)
        except:
            pass

        # Chunk Manager
        try:
            with open(self.working_directory+'/chunk_manager.pkl', 'rb') as f:
                self.chm = pickle.load(f)
        except:
            pass

        # Graph Manager
        try:
            with open(self.working_directory+'/graph_manager.pkl', 'rb') as f:
                self.gm = pickle.load(f)
                self.gm.project = self
        except:
            pass

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

        self.rm = RegionManager(db_wd=self.working_directory)

        self.gm.project = self
        self.gm.rm = self.rm
        self.gm.update_nodes_in_t_refs()

def dummy_project():
    from core.classes_stats import dummy_classes_stats
    from core.region.region_manager import RegionManager

    p = Project()
    p.stats = dummy_classes_stats()
    p.rm = RegionManager()

    return p


if __name__ == "__main__":
    p = Project()
    p.name = 'test'
    p.a = 20
    p.working_directory = '/home/flipajs/test'
    p.save()

    a = Project()
    a.load('/home/flipajs/test/test.pkl')
    print "Project name: ", a.name