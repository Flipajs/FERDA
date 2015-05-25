__author__ = 'filip@naiser.cz'

from utils.video_manager import VideoType
import pickle
from methods.bg_model.model import Model
from PyQt4 import QtCore


class Project:
    """
    This class encapsulates one experiment using FERDA
    """
    def __init__(self):
        self.name = ''
        self.description = ''
        self.video_paths = []
        self.working_directory = ''

        self.bg_model = None
        self.arena_model = None
        self.classes = None
        self.groups = None
        self.animals = None
        self.stats = None


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


        p = Project()
        p.name = self.name
        p.description = self.description
        p.video_paths = self.video_paths
        p.working_directory = self.working_directory

        with open(self.working_directory+'/'+self.name+'.fproj', 'wb') as f:
            pickle.dump(p.__dict__, f, 2)

        self.save_qsettings()

    def save_qsettings(self):
        s = QtCore.QSettings('FERDA')
        settings = {}

        for k in s.allKeys():
            settings[k] = s.value(k, 0, str)

        with open(self.working_directory+'/settings.pkl', 'wb') as f:
            pickle.dump(settings, f)

    def load_qsettings(self):
        with open(self.working_directory+'/settings.pkl', 'rb') as f:
            settings = pickle.load(f)
            qs = QtCore.QSettings('FERDA')

            for key, it in settings.iteritems():
                qs.setValue(key, it)


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


if __name__ == "__main__":
    p = Project()
    p.name = 'test'
    p.a = 20
    p.working_directory = '/home/flipajs/test'
    p.save()

    a = Project()
    a.load('/home/flipajs/test/test.pkl')
    print "Project name: ", a.name