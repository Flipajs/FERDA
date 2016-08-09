from scipy.spatial import ConvexHull
from core.project.project import Project
from matplotlib import pyplot as plt


class FrameSearch:

    def __init__(self, project, frame):
        """

        :param project:
        :param frame:
        """
        self.project = project
        self.frame = frame
        self.chunks_id = project.chm.chunks_in_frame(frame)
        self.contours = self._prepare_contours(self.chunks_id)
        self.hulls = self._prepare_convex_hulls(self.chunks_id)


    def find_point(self, x, y):
        """
        Finds out, whether there's a chunk in the given point
        :param x: x cooradinate
        :param y: y coordinate
        :return: id of chunk present on point, -1 if there is none
        """
        pass

    def find_range(self, xA, yA, xB, yB):
        """
        Finds out whether there are any chunks present in the given range
        :param xA: x coordinate of upper left corner
        :param yA: y coordinate of upper left corner
        :param xB: x coordinate of lower right corner
        :param yB: y coordinate of lower right corner
        :return: list od ids of chunks present, -1 if there is none
        """

    def find_closest_chunk(self, x, y):
        """
        Returns the closest chunk to the given point. If the point lies already on chunk, its id is returned.
        :param x: x cooradinate
        :param y: y coordinate
        :return: id of the closest chunk, -1 if there are no chunks in the frame
        """
        pass

    def find_count_in_range(self):
        pass

    def visualize_frame(self):
        for contour, hull in zip(self.contours.values(), self.hulls.values()):
            plt.scatter(*zip(*contour), c='r')
            plt.plot(*zip(*hull), c='b')
        plt.axis('equal')
        plt.show()

    def _prepare_contours(self, chunks):
        d = {}
        for ch in chunks:
            d[ch] = self._get_contour(ch)
        return d

    def _get_contour(self, chunk):
        return project.gm.region(chunk[self.frame]).contour_without_holes()

    def _prepare_convex_hulls(self, chunks):
        d = {}
        for ch in chunks:
            d[ch] = self._get_convex_hull(self._get_contour(ch))
        return d

    def _get_convex_hull(self, points):
        hull = ConvexHull(points)
        hull_indices = hull.vertices
        return points[hull_indices, :]

    def _point_in_area(self, x, y, points):
        last = points[-1]
        for point in points:
            pass




if __name__ == "__main__":
    project = Project()
    project.load("/home/simon/FERDA/projects/Cam1_/cam1.fproj")
    search = FrameSearch(project, frame=30)
    search.visualize_frame()

