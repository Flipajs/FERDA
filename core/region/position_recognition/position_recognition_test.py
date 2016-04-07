import numpy
from numpy import transpose
from core.project.project import Project
from numpy.linalg import eig
import math

NUMBER_OF_DATA = 10
PRECISION = 3

project = Project()
project.load("/home/sheemon/FERDA/projects/Cam1_/cam1.fproj")


def get_pca(chunks):
    matrix = []
    for ch in chunks:
        for vector in get_matrix(ch):
            matrix.append(vector)
    # for a in matrix:
    #     print (a)
    eigenfaces = get_eigenfaces(matrix)
    return numpy.dot(transpose(eigenfaces[:PRECISION]), matrix)


def get_eigenfaces(chunk_matrix):
    vector_matrix = numpy.array(chunk_matrix)
    transposed = transpose(chunk_matrix)
    # with open("out.txt", mode="w") as file:
    #     file.write(str(vector_matrix))
    #     file.write("\n\n\n")
    #     file.write(str(transposed))
    covariance_matrix = numpy.dot(transposed, chunk_matrix)
    eigens = eig(covariance_matrix)
    return eigens

def get_chunks_regions(ch):
    chunk = project.chm[ch]
    chunk_start = chunk.start_frame(project.gm)
    chunk_end = chunk.end_frame(project.gm)
    while chunk_start <= chunk_end:
        yield project.gm.region(chunk[chunk_start])
        chunk_start += 1


def get_matrix(chunk):
    matrix = []
    for region in get_chunks_regions(chunk):
        matrix.append(get_region_vector(region))
    return matrix


def get_region_vector(region):
    vector = []
    centroid = region.centroid()
    contours = region.contour()
    con_index = 0
    head_beam = centroid - contours[0]
    last = head_beam
    last_angle = 0
    angle = 0
    circle = math.pi * 2
    while angle < circle:
        beam = centroid - contours[con_index]
        ang = deviation(head_beam, beam)
        if ang == angle:
            vector.append(vector_norm(beam))
        elif ang > angle:
            vector.append(interpolation(angle - last_angle, ang - angle, last, beam))
        last = beam
        last_angle = ang
        con_index += 1
        angle += circle / NUMBER_OF_DATA
    # hlavicka?
    return vector


def vector_norm(u):
    return math.sqrt(sum(i ** 2 for i in u))


def dot_product(u, v):
    return sum((x * y) for x, y in zip(u, v))


def deviation(u, v):
    return math.acos(dot_product(u, v) / (vector_norm(u) * vector_norm(v)))


def interpolation(first_angle, sec_angle, u, v):
    size_u = vector_norm(u)
    size_v = vector_norm(v)
    if size_u > size_v:
        size_u, size_v = size_v, size_u
        first_angle, sec_angle = sec_angle, first_angle
    return size_u + (size_v - size_u) / 2 * (first_angle / sec_angle)


if __name__ == '__main__':
    chunks = project.gm.chunk_list()
    get_pca(chunks[:1])
