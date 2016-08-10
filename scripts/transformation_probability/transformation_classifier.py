import numpy as np

from core.project.project import Project


def descriptor(r1, r2):
    # centroid distance
    centr_dist = np.linalg.norm(r2.centroid() - r1.centroid())

    # margin difference
    margin_diff = r2.margin() - r1.margin()

    # intensity
    max_intensity = r2.max_intensity_ - r1.max_intensity_
    min_intensity = r2.min_intensity_ - r1.min_intensity_

    # area difference
    area_diff = r2.area() - r1.area()

    # axis difference
    axis_diff = r2.major_axis_ - r1.major_axis_

    return (centr_dist, margin_diff, max_intensity, min_intensity, area_diff, axis_diff)


if __name__ == "__main__":
    project = Project()
    project.load("/home/simon/FERDA/projects/Cam1_/cam1.fproj")

    print descriptor(project.rm[1], project.rm[2])
