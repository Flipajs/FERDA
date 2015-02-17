__author__ = 'filip@naiser.cz'

from utils.video_manager import VideoType
import pickle


class Project:
    """
    This class encapsulates one experiment using FERDA
    """
    def __init__(self):
        self.name = ''
        self.description = ''
        self.video_type = VideoType.ORDINARY
        self.video_paths = []
        self.project_folder = ''

    def save(self):
        with open(self.project_folder+'/'+self.name+'.pkl', 'wb') as f:
            pickle.dump(self.__dict__, f, 2)

    def load(self, path):
        with open(path, 'rb') as f:
            tmp_dict = pickle.load(f)

        self.__dict__.update(tmp_dict)


if __name__ == "__main__":
    p = Project()
    p.name = 'test'
    p.a = 20
    p.project_folder = '/home/flipajs/test'

    p.save()


    a = Project()
    a.load('/home/flipajs/test/test.pkl')
    print "Project name: ", a.name