from scipy.spatial import ConvexHull
from core.project.project import Project
from matplotlib import pyplot as plt

from scripts.frame_search.line_segment_intersect import do_intersect, on_segment


def random_hex_color():
    from random import randint
    rand_num = randint(1, 3)

    hex_tokens = "0123456789ab"
    color = "#"
    for i in range(1, 4):
        if i == rand_num:
            color += "ff"
        else:
            color += (hex_tokens[randint(0, len(hex_tokens) - 1)] +
                      hex_tokens[randint(0, len(hex_tokens) - 1)])
    return color

class FrameSearch:

    def __init__(self, project, frame):
        """

        :param project:
        :param frame:
        """
        self.project = project
        self.frame = frame
        chunks = project.chm.chunks_in_frame(frame)
        self.contours = self._prepare_contours(chunks)
        self.hulls = self._prepare_convex_hulls(chunks)
        self.chunks_id = [ch.id() for ch in chunks]

    def find_point(self, x, y, convex_hull=True):
        """
        Finds out, whether there's a chunk in the given point
        :param x: x cooradinate
        :param y: y coordinate
        :param convex_hull: uses convex hull approximation, faster but less accurate
        :return: id of chunk present on point, -1 if there is none
        """
        ret = []
        for ch in self.chunks_id:
            if convex_hull:
                if self._point_in_area(x, y, self.hulls[ch]):
                    ret.append(ch)
            else:
                if self._point_in_area(x, y, self.contours[ch]):
                    ret.append(ch)
        return ret

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
        for ch in self.chunks_id:
            c = random_hex_color()
            plt.plot(*zip(*self.contours[ch]), c='r')
            # plt.scatter(*zip(*self.contours[ch]), c='r')
            plt.plot(*zip(*self.hulls[ch]), c=c, label=str(ch))
        plt.legend(loc='upper right')
        plt.axis('equal')
        # mng = plt.get_current_fig_manager()
        # mng.full_screen_toggle()
        plt.show()

    def _prepare_contours(self, chunks):
        d = {}
        for ch in chunks:
            d[ch.id()] = self._get_contour(ch)
        return d

    def _get_contour(self, chunk):
        return project.gm.region(chunk[self.frame]).contour_without_holes()

    def _prepare_convex_hulls(self, chunks):
        d = {}
        for ch in chunks:
            d[ch.id()] = self._get_convex_hull(self._get_contour(ch))
        return d

    def _get_convex_hull(self, points):
        hull = ConvexHull(points)
        hull_indices = hull.vertices
        return points[hull_indices, :]

    def _point_in_area(self, x, y, points, relative_point=(-1, -1)):
        a, b = relative_point[1] - y, x - relative_point[0]
        c = relative_point[0] * a + relative_point[1] * b
        last = points[-1]
        count = 0
        for point in points:
            if point[0] == x and point[1] == y:
                return True
            if a * point[0] + b * point[1] == c or (x - relative_point[0] == a and y - relative_point[1] == b):
                return self._point_in_area(x, y, points, relative_point=(relative_point[0] + 1, relative_point[1] - 1))
            if do_intersect(last, point, relative_point, (x, y)):
                count += 1
            last = point
        return count % 2 == 1

if __name__ == "__main__":
    project = Project()
    project.load("/home/simon/FERDA/projects/Cam1_/cam1.fproj")
    search = FrameSearch(project, frame=30)

    print search.find_point(580, 247)
    print search.find_point(600, 260)
    print search.find_point(600, 260, False)

    # search.visualize_frame()

