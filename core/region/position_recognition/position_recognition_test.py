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


def get_pca(chunks, number_of_data):
    matrix = []
    for ch in chunks:
        for vector in get_matrix(ch, number_of_data):
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


def get_matrix(chunk, number_of_data):
    matrix = []
    for region in get_chunks_regions(chunk):
        matrix.append(get_region_vector(region, number_of_data))
    return matrix


def find_closest_angle_pos(beam_angle, number_of_data):
    # interpolar search

    step = math.pi * 2 / number_of_data
    left = 0
    right = number_of_data - 1
    while left < right:
        #TODO - precision!
        middle = int((left + right) / 2)
        if abs(beam_angle - (middle * step)) <= (step / 2) + 0.000001:
            return middle
        elif (middle * step) < beam_angle:
            left = middle + 1
        elif (middle * step) > beam_angle:
            right = middle - 1
        else:
            return middle
    return left


def get_region_vector(region, number_of_data):
    centroid = region.centroid()
    contours = region.contour()

    # table of the closest lower and higher deviated beams
    results = [[None, None] for x in range(number_of_data)]

    step = 2 * math.pi / number_of_data

    plt.figure()
    points_aux = []
    points_aux.append(centroid)

    head_beam = centroid - contours[0]
    con_index = 0

    while con_index < len(contours):
        beam = centroid - contours[con_index]
        points_aux.append(contours[con_index])

        beam_angle = deviation(head_beam, beam)
        if abs(head_beam[0] - abs(centroid[0]) < abs(head_beam[0] - abs(centroid[0]))):
            if abs(head_beam[1] - abs(centroid[1]) >= abs(head_beam[1] - abs(centroid[1]))):
                beam_angle += math.pi
        distance = vector_norm(beam)

        closest_angle_pos = find_closest_angle_pos(beam_angle, number_of_data)
        closest_angle = closest_angle_pos * step
        #TODO - precision
        #TODO - precision multiplier
        pr = 0.9
        if (closest_angle - beam_angle) < 0.00001:
            for i in [0, 1]:
                data_entry = results[closest_angle_pos][i]
                if data_entry is None or data_entry[1] < distance * pr:
                    results[closest_angle_pos][i] = (beam_angle, distance)
        elif closest_angle < beam_angle:
            highest_lower = results[closest_angle_pos][0]
            if highest_lower is None or highest_lower[1] < distance * pr or highest_lower[0] < beam_angle:
                results[closest_angle_pos][0] = (beam_angle, distance)
        else:
            lowest_higher = results[closest_angle_pos][1]
            if lowest_higher is None or lowest_higher[1] < distance * pr or lowest_higher[0] > beam_angle:
                results[closest_angle_pos][1] = (beam_angle, distance)

        con_index += 1

    for i in range(number_of_data):
        print (step * i * (180 / math.pi))
        print (results[i][0])
        print (results[i][1])
    results = [0 if h_l is None or l_h is None else interpolation(h_l[0], l_h[0], h_l[1], l_h[1]) for h_l, l_h in results]

    plt.hold(True)
    xs = [x[1] for x in points_aux]
    ys = [x[0] for x in points_aux]

    # res_x = [math.cos(x * step - deviation(head_beam, (0, 1))) * results[x] + centroid[0] for x in range(number_of_data)]
    # res_y = [math.sin(x * step - deviation(head_beam, (0, 1))) * results[x] + centroid[0] for x in range(number_of_data)]
    res_x = [range(number_of_data)]
    res_y = [results[x] for x in range(number_of_data)]
    plt.scatter(xs, ys)
    plt.scatter(res_x, res_y)
    plt.show(True)

    return results


def vector_norm(u):
    return math.sqrt(sum(i ** 2 for i in u))


def dot_product(u, v):
    return sum((x * y) for x, y in zip(u, v))


def deviation(u, v):
    ang = math.acos(dot_product(u, v) / ((vector_norm(u)) * vector_norm(v)))
    return ang


def interpolation(first_angle, sec_angle, first_dist, second_dist):
    if first_angle == 0:
        return first_dist
    if sec_angle == 0:
        return second_dist
    if first_dist > second_dist:
        first_dist, second_dist = second_dist, first_dist
        first_angle, sec_angle = sec_angle, first_angle
    return first_dist + ((second_dist - first_dist) / (2 * (first_angle / sec_angle)))


if __name__ == '__main__':
    print ((180/math.pi) *(deviation((1, 0), (1, 0))))
    print ((180/math.pi) *(deviation((1, 0), (1, 1)))  )
    print ((180/math.pi) *(deviation((1, 0), (0, 1)))   )
    print ((180/math.pi) *(deviation((1, 0), (-1, 1)))   )
    print ((180/math.pi) *(deviation((1, 0), (-1, 0)))    )
    print ((180/math.pi) *(deviation((1, 0), (-1, -1)))    )
    print ((180/math.pi) *(deviation((1, 0), (-1, 0)))      )
    print ((180/math.pi) *(deviation((1, 0), (1, -1)))       )
    # number_of_data = 9
    # degrees = range(3610)
    # for deg in degrees:
    #     print(deg, find_closest_angle_pos((math.pi / 180) * (deg / 10), number_of_data))
    # print(601, find_closest_angle_pos((math.pi / 180) * (601 / 10), number_of_data))


    chunks = project.gm.chunk_list()
    get_pca(chunks[:1], NUMBER_OF_DATA)
