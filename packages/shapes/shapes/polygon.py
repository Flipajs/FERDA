import cv2
import numpy as np

from shapes.shape import Shape

#TODO: closed, open polygon?


class Polygon(Shape):
    line_epsilon = 0.001

    def __init__(self, points, frame=None):
        """

        :param points: polygon vertices [(x, y), (x, y), ...]
        :param frame: video frame number
        """
        super().__init__(frame=frame)
        if not isinstance(points, np.ndarray):
            points = np.array(points, dtype=np.float32)
        if len(points) == 2:
            vec = points[0] - points[1]
            vec /= np.linalg.norm(vec, axis=0)
            vec_ortho = np.array((-vec[1], vec[0]))
            points = np.vstack((points, points[1] + vec_ortho * self.line_epsilon))
        self.points = points

    def __str__(self):
        return 'Polygon {points}'.format(points=self.points)

    @property
    def xy(self):
        """ Computes the centroid (a.k.a. center of gravity) for a non-self-intersecting polygon.

        source: https://stackoverflow.com/a/56826826/322468

        Parameters
        ----------
        polygon : list of two-dimensional points (points are array-like with two elements)
            Non-self-intersecting polygon (orientation does not matter).

        Returns
        -------
        center_of_gravity : list with 2 elements
            Coordinates (or vector) to the centroid of the polygon.
        """
        offset = self.points[0]
        center_of_gravity = [0.0, 0.0]
        double_area = 0.0
        for ii in range(len(self.points)):
            p1 = self.points[ii]
            p2 = self.points[ii - 1]
            f = (p1[0] - offset[0]) * (p2[1] - offset[1]) - (p2[0] - offset[0]) * (p1[1] - offset[1])
            double_area += f
            center_of_gravity[0] += (p1[0] + p2[0] - 2 * offset[0]) * f
            center_of_gravity[1] += (p1[1] + p2[1] - 2 * offset[1]) * f
        center_of_gravity[0] = center_of_gravity[0] / (3 * double_area) + offset[0]
        center_of_gravity[1] = center_of_gravity[1] / (3 * double_area) + offset[1]
        return center_of_gravity
        # If you want to return both the CoG and the area, comment the return above
        return center_of_gravity, abs(double_area / 2)

    def to_poly(self):
        return self.points

    def is_intersecting(self, other):
        """
        Check if two polygons intersect.

        Two polygons sharing an edge are not intersecting.

        :param other: Polygon
        :return: bool
        """
        ret, _ = cv2.intersectConvexConvex(self.points.astype('float32'), other.points.astype('float32'))
        return bool(ret)

    def intersection(self, other):
        _, intersection_poly = cv2.intersectConvexConvex(self.points.astype('float32'), other.points.astype('float32'))
        return intersection_poly

    def to_array(self):
        return self.points

    def __sub__(self, other):
        return np.linalg.norm(self.xy - other.xy)

    def contains(self, xy):
        return cv2.pointPolygonTest(self.points, xy, False)

    def move(self, delta_xy):
        self.points += delta_xy
        return self

    def draw(self, ax=None, label=None, color=None):
        import matplotlib.pylab as plt
        from matplotlib.patches import Polygon
        if ax is None:
            ax = plt.gca()
        if color is None:
            color = 'r'
        patch = Polygon(self.points,
                             facecolor='none', edgecolor=color,
                             label=label, linewidth=1)
        ax.add_patch(patch)
        if label is not None:
            plt.annotate(label, self.xy)  # , xytext=(0, -self.height / 2), textcoords='offset pixels')
        return patch

    def draw_to_image(self, img, label=None, color=None):
        if color is None:
            color = (0, 0, 255)
        round_tuple = lambda x: tuple([int(round(num)) for num in x])
        cv2.polylines(img, self.points, color)
        if label is not None:
            font_size = 1
            font_thickness = 1
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            text_size, _ = cv2.getTextSize(label, font_face, font_size, font_thickness)
            cv2.putText(img, label, round_tuple((self.xy[0] - (text_size[0] / 2), self.ymin - text_size[1])),
                        font_face, font_size, (255, 255, 255), font_thickness)
