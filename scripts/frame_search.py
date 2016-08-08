from core.project.project import Project
from matplotlib import pyplot as plt


class FrameSearch:

    def __init__(self, project, frame):
        self.project = project
        self.frame = frame
        self.chunks = project.chm.chunks_in_frame(frame)

    def visualize_frame(self):
        for ch in self.chunks:
            contour = self._get_contour(ch)
            plt.scatter(*zip(*contour))
        plt.axis('equal')
        plt.show()

    def _get_contour(self, chunk):
        return project.gm.region(chunk[self.frame]).contour_without_holes()

    def _prepare_frame(self):
        pass


if __name__ == "__main__":
    project = Project()
    project.load("/home/simon/FERDA/projects/Cam1_/cam1.fproj")
    search = FrameSearch(project, frame=30)
    search.visualize_frame()

