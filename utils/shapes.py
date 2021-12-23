import shapes


class Ellipse(shapes.Ellipse):
    @classmethod
    def from_region(cls, region):
        yx = region.centroid()
        tmp = cls(yx[1], yx[0], -np.rad2deg(region.theta_), 2 * region.major_axis_, 2 * region.minor_axis_,
                  region.frame())
        return tmp

    def to_region(self):
        r = Region(is_origin_interaction=True, frame=self.frame)
        r.centroid_ = self.xy[::-1]
        r.theta_ = -np.deg2rad(self.angle_deg)
        r.major_axis_ = self.major / 2
        r.minor_axis_ = self.minor / 2
        return r

class BBox(shapes.BBox):
    @classmethod
    def from_region(cls, region):
        yx = region.centroid()
        tmp = cls(yx[1], yx[0], -np.rad2deg(region.theta_), 2 * region.major_axis_, 2 * region.minor_axis_,
                  region.frame())
        return tmp

class RotatedBBox(shapes.RotatedBBox):
    @classmethod
    def from_region(cls, region):
        yx = region.centroid()
        tmp = cls(yx[1], yx[0], -np.rad2deg(region.theta_), 2 * region.major_axis_, 2 * region.minor_axis_,
                  region.frame())
        return tmp

