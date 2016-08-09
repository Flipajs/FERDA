from core.project.project import Project
from matplotlib import pyplot as plt


class FrameSearch:

    def __init__(self, project, frame):
        self.project = project
        self.frame = frame
        self.chunks = project.chm.chunks_in_frame(frame)

    def find_point(self, point=None, x=None, y=None, convex_hall=1):
        """
        Finds out, whether there's a chunk in the given point
        :param point: two-dimensional iterable of x, y coordinates
        :param x: x cooradinate
        :param y: y coordinate
        :param convex_hall: convex hall approximation is used,
                if off may give more accurate results, but takes more time
        :return: id of chunk present on point, -1 if there is none
        """
        pass

    def find_range(self, pointA=None, pointB=None, xA=None, yA=None, xB=None, yB=None, convex_hall=1):
        """
        Finds out whether there are any chunks present in the given range
        :param pointA: two-dimensional iterable of upper left corner point coordinates
        :param pointB: two-dimensional iterable of upper lower right point coordinates
        :param xA: x coordinate of upper left corner
        :param yA: y coordinate of upper left corner
        :param xB: x coordinate of lower right corner
        :param yB: y coordinate of lower right corner
        :param convex_hall: convex_hall: convex hall approximation is used,
                if off may give more accurate results, but takes more time
        :return: list od ids of chunks present, -1 if there is none
        """

    def find_closest_chunk(self, point=None, x=None, y=None, convex_hall=1):
        """
        Returns the closest chunk to the given point. If the point lies already on chunk, its id is returned.
        :param point: two-dimensional iterable of x, y coordinates
        :param x: x cooradinate
        :param y: y coordinate
        :param convex_hall: convex hall approximation is used,
                if off may give more accurate results, but takes more time
        :return: id of the closest chunk, -1 if there are no chunks in the frame
        """
        pass
    
    def find_count_in_range(self):
        pass

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

