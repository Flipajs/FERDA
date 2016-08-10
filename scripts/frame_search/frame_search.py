from matplotlib import patches
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull
from scipy.spatial import KDTree

from core.project.project import Project
from scripts.frame_search.line_segment_intersect import do_intersect

PRIME_CONST = 51


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
    def __init__(self, project, frame=0):
        self.project = project
        self.frame = frame
        chunks = project.chm.chunks_in_frame(frame)
        self.contours = self._prepare_contours(chunks)
        self.contours_id = {}
        self.contours_tree = self._prepare_contour_tree(self.contours)
        self.hulls = self._prepare_convex_hulls(chunks)
        self.hulls_tree = self._prepare_hulls_tree(self.hulls)
        self.chunks_id = [ch.id() for ch in chunks]

    def find_point(self, x, y, convex_hull=True):
        """
        Finds out, whether there's a chunk in the given point
        :param x: x coordinate
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

    def find_range(self, x_a, y_a, x_b, y_b, convex_hull=True):
        """
        Finds out whether there are any chunks present in the given range
        :param x_a: x coordinate of lower left corner
        :param y_a: y coordinate of lower left corner
        :param x_b: x coordinate of upper right corner
        :param y_b: y coordinate of upper right corner
        :param convex_hull: uses convex hull approximation, faster but less accurate
        :return: list od ids of chunks present, -1 if there is none
        """
        if x_a > x_b:
            x_a, x_b = x_b, x_a
        if y_a > y_b:
            y_a, y_b = y_b, y_a
        ret = []
        for ch in self.chunks_id:
            if convex_hull:
                if self._range_in_area(x_a, y_a, x_b, y_b, self.hulls[ch]):
                    ret.append(ch)
                elif self._range_in_area(x_a, y_a, x_b, y_b, self.contours[ch]):
                    ret.append(ch)
        return ret

    def find_closest_chunk(self, x, y, convex_hull=True):
        """
        Returns the closest chunk to the given point. If the point lies already on chunk, its id is returned.
        :param x: x coordinate
        :param y: y coordinate
        :param convex_hull: uses convex hull approximation, faster but less accurate
        :return: tuple(id, dist, point) where id is id of the closest chunk, dist is the distance to the nearest point
        """
        if convex_hull:
            dist, closest = self.hulls_tree.query([x, y])
            closest = self.hulls_tree.data[closest]

        else:
            dist, closest = self.contours_tree.query([x, y])
            closest = self.contours_tree.data[closest]

        return self.contours_id[self._hash_point(closest)], dist, closest

    def visualize_frame(self):
        """
        Visualizes frame, only for debugging purposes. portraits every chunk with different color
        """
        for ch in self.chunks_id:
            c = random_hex_color()
            plt.plot(*zip(*self.contours[ch]), c='r')
            plt.plot(*zip(*self.hulls[ch]), c=c, label=str(ch))
        plt.legend(loc='upper right')
        plt.axis('equal')
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

    @staticmethod
    def _get_convex_hull(points):
        hull = ConvexHull(points)
        hull_indices = hull.vertices
        return points[hull_indices, :]

    def _prepare_contour_tree(self, data):
        points = []
        for ch in data:
            for point in data[ch]:
                points.append(point)
                self.contours_id[self._hash_point(point)] = ch
        return KDTree(points)

    @staticmethod
    def _prepare_hulls_tree(data):
        points = []
        for ch in data:
            print ch
            for point in data[ch]:
                points.append(point)
        print len(points)
        return KDTree(points)

    @staticmethod
    def _hash_point(point):
        return (PRIME_CONST + point[0]) * PRIME_CONST + point[1]

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

    @staticmethod
    def _range_in_area(x_a, y_a, x_b, y_b, points):
        x_min, y_min = reduce(lambda a, b: (min(a[0], b[0]), min(a[1], b[1])), points)
        x_max, y_max = reduce(lambda a, b: (max(a[0], b[0]), max(a[1], b[1])), points)

        return x_a < x_max and y_a < y_max and x_b > x_min and y_b > y_min


if __name__ == "__main__":
    project = Project()
    project.load("/home/simon/FERDA/projects/Cam1_/cam1.fproj")
    search = FrameSearch(project, frame=30)

    # examples of usage
    print search.find_point(580, 247)
    print search.find_point(600, 260)
    print search.find_point(600, 260, False)

    print search.find_range(350, 150, 400, 250)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, aspect='equal')
    ax1.add_patch(
        patches.Rectangle(
            (350, 150),
            50,
            100,
            fill=False
        )
    )

    print search.find_closest_chunk(500, 500)
    print search.find_closest_chunk(500, 500, False)

    search.visualize_frame()
