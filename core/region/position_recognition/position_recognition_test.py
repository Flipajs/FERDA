import numpy
from numpy import transpose
from core.project.project import Project
from numpy.linalg import eig
import math
import matplotlib.pyplot as plt

NUMBER_OF_DATA = 100
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
    # odmocnina ze dvou vzdalenost
    # distances
    vector = []
    centroid = region.centroid()
    contours = region.contour()
    # table of deviations and length
    results = [[] for x in range(len(contours))]
    con_index = 0
    head_beam = centroid - contours[0]
    last = head_beam
    last_angle = 0
    angle = 0
    circle = 2 * math.pi
    step = circle / NUMBER_OF_DATA

    plt.figure()
    points_aux = []
    points_aux.append(centroid)

    while con_index < len(contours):
        beam = centroid - contours[con_index]

        points_aux.append(contours[con_index])

        ang = deviation(head_beam, beam)
        print((ang * 180) / math.pi)
        distance = vector_norm(beam)
        data = (ang, distance)
        adress = int(ang / step)
        print(adress)
        if float(ang) / float(step) > 0.5:
            adress += 1;

        results[adress].append(data)


        # if ang == angle:
        #     vector.append(vector_norm(beam))
        #     angle += circle / NUMBER_OF_DATA
        # elif ang > angle:
        #     check?
        #     vector.append(interpolation(angle - last_angle, ang - angle, last, beam))
        #     angle +=

        last = beam
        last_angle = ang
        con_index += 1
        # print(angle)


    # hlavicka?
    # plt.hold(True)
    # xs = [x[1] for x in points_aux]
    # ys = [x[0] for x in points_aux]
    # plt.scatter(xs, ys)
    # # plt.scatter(*(zip(*points_aux)))
    # plt.show(True)

    return vector


def vector_norm(u):
    return math.sqrt(sum(i ** 2 for i in u))


def dot_product(u, v):
    return sum((x * y) for x, y in zip(u, v))


def deviation(u, v):
    try:
        ang = math.acos(abs(dot_product(u, v)) / ((vector_norm(u)) * vector_norm(v)))
    except:
        return 0
    if dot_product(u, v) / ((vector_norm(u)) * vector_norm(v)) < 0:
        ang += math.pi
    return ang


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
