import math

import numpy
from numpy import transpose
from numpy.linalg import eig, norm
from matplotlib import pyplot as plt
from core.project.project import Project

NUMBER_OF_DATA = 100
NUMBER_OF_PC = 100

average = 0

def get_pca(chunks, number_of_data, chm, gm):
    matrix = []
    for ch in chunks:
        for vector in get_matrix(ch, number_of_data, chm, gm):
            matrix.append(vector)
    matrix = transpose(matrix)
    eigenfaces = get_eigenfaces(matrix) #vectors = columns
    coefficient_matrix = transpose(eigenfaces).dot(matrix)
    # return eigenfaces, coefficient_matrix
    # return transpose(eigenfaces).dot(coefficient_matrix)
    pca = eigenfaces.dot(coefficient_matrix)
    # pca = map(lambda a: [i + avg for i, avg in zip(a, average)], pca)
    original = matrix
    print pca[1]
    print original[1]
    print reduce(lambda a,b: a + b, original)
    plt.scatter(range(len(original) * len(original[0])), reduce(lambda a,b: a + b, original))
    plt.scatter(range(len(original) * len(original[0])), reduce(lambda a,b: a + b, pca))


def get_eigenfaces(chunk_matrix):
    m = numpy.array(chunk_matrix)
    transposed = transpose(m)
    covariance_matrix = transposed.dot(m)
    eigenvalues, eigenvectors = eig(covariance_matrix)
    index = eigenvalues.argsort()[::-1]
    eigenfaces = eigenvectors[:, index][:NUMBER_OF_PC]
    return m.dot(transpose(eigenfaces))


def get_chunks_regions(ch, chm, gm):
    chunk = chm[ch]
    print "Chunk " + str(ch)
    chunk_start = chunk.start_frame(gm)
    chunk_end = chunk.end_frame(gm)
    while chunk_start <= chunk_end:
        yield project.gm.region(chunk[chunk_start])
        chunk_start += 1


def get_matrix(chunk, number_of_data, chm, gm):
    distance_matrix = []
    for i, region in enumerate(get_chunks_regions(chunk, chm, gm)):
        print i
        distance_matrix.append(get_region_vector(region, number_of_data))
        print distance_matrix[-1]
    global average
    average = [sum(a) / len(a) for a in zip(*distance_matrix)]
    return map(lambda a: [i - avg for i, avg in zip(a, average)], distance_matrix)

def get_region_vector(region, number_of_data):
    contour = region.contour_without_holes()
    con_length = len(contour)

    if len(contour) < number_of_data:
        print("Number of data should be smaller than contours length")

    distances = [0]
    perimeter = vector_norm(contour[0] - contour[-1])
    for i in range(1, con_length):
        perimeter += vector_norm(contour[i] - contour[i - 1])
        distances.append(perimeter)

    result = []
    step = con_length / float(number_of_data)
    i = 0
    while i <= con_length - 1:
        result.append(distances[int(i)])
        i += step

    # plt.hold(True)
    # plt.axis('equal')
    # plt.scatter([i[0] for i in contour], [i[1] for i in contour])
    # plt.scatter(list(range(number_of_data)), result)
    # plt.show(True)
    return result


def compute_contour_perimeter(contour):
    length = len(contour)
    perimeter = vector_norm(contour[0] - contour[length - 1])
    for i in range(1, length):
        perimeter += vector_norm(contour[i] - contour[i - 1])
    return perimeter


def vector_norm(u):
    return math.sqrt(sum(i ** 2 for i in u))

if __name__ == '__main__':
    from scripts import fix_project
    project = Project()
    project.load("/home/simon/FERDA/projects/Cam1_/cam1.fproj")
    chunks = project.gm.chunk_list()
    pca = get_pca(chunks[:1], NUMBER_OF_DATA, project.chm, project.gm)

# def get_region_vector_angle(region, number_of_data):
#     centroid = region.centroid()
#     contours = region.contour()
#
#     results = [[None, None] * number_of_data]
#
#     step = 2 * math.pi / number_of_data
#
#     plt.figure()
#     points_aux = []
#     points_aux.append([0, 0])
#
#     head_beam = contours[0] - centroid
#     head_ang = deviation(head_beam)
#
#     con_index = 0
#     while con_index < len(contours):
#         beam = contours[con_index] - centroid
#         points_aux.append(beam)
#         dev = deviation(beam)
#         beam_angle = dev - head_ang
#         if beam_angle < 0:
#             beam_angle += math.pi * 2
#         distance = vector_norm(beam)
#
#         closest_angle_pos = find_closest_angle_pos(beam_angle, number_of_data)
#         closest_angle = closest_angle_pos * step
#         print ("Angle: " + str(math.degrees(closest_angle)))
#         print(distance)
#         # TODO - precision
#         # TODO - precision multiplier
#         if (closest_angle - beam_angle) < 0.00001:
#             for i in [0, 1]:
#                 data_entry = results[closest_angle_pos][i]
#                 if data_entry is None or data_entry[1] < distance:
#                     results[closest_angle_pos][i] = (beam_angle, distance)
#         elif closest_angle < beam_angle:
#             highest_lower = results[closest_angle_pos][0]
#             if highest_lower is None or highest_lower[1] < distance or highest_lower[0] < beam_angle:
#                 results[closest_angle_pos][0] = (beam_angle, distance)
#         else:
#             lowest_higher = results[closest_angle_pos][1]
#             if lowest_higher is None or lowest_higher[1] < distance or lowest_higher[0] > beam_angle:
#                 results[closest_angle_pos][1] = (beam_angle, distance)
#         con_index += 1
#
#     for i in range(number_of_data):
#         print (math.degrees(step * i))
#         print (results[i][0])
#         print (results[i][1])
#
#     for i in range(number_of_data):
#         if results[i][0] is None:
#             results[i][0] = (step * i, 0)
#         if results[i][1] is None:
#             results[i][1] = (step * i, 0)
#
#     for i in range(number_of_data):
#         print (step * i * (180 / math.pi))
#         print (results[i][0])
#         print (results[i][1])
#
#     results = [interpolation(h_l[0], l_h[0], h_l[1], l_h[1]) for h_l, l_h in results]
#     plt.scatter(range(100, number_of_data + 100), [results[x] for x in range(number_of_data)], c='r')
#
#     plt.hold(True)
#     plt.axis('equal')
#     xs = [x[1] for x in points_aux]
#     ys = [x[0] for x in points_aux]
#     plt.scatter(xs, ys, c="c")
#
#     res_aux = []
#     for i in range(number_of_data):
#         y = math.sin(step * i) * results[i]
#         x = math.cos(step * i) * results[i]
#         res_aux.append((x, y))
#
#     xs = [x[1] for x in res_aux]
#     ys = [x[0] for x in res_aux]
#     plt.scatter(xs, ys)
#
#     plt.show(True)
#
#     return results
#
#
#
#
# def dot_product(u, v):
#     return sum((x * y) for x, y in zip(u, v))
#
#
# def deviation(u):
#     ang = math.atan2(u[1], u[0])
#     if ang < 0:
#         ang += 2 * math.pi
#     return ang
#
#
# def interpolation(first_angle, sec_angle, first_dist, second_dist):
#     if first_angle == 0:
#         return first_dist
#     if sec_angle == 0:
#         return second_dist
#     if first_dist > second_dist:
#         first_dist, second_dist = second_dist, first_dist
#         first_angle, sec_angle = sec_angle, first_angle
#
# def find_closest_angle_pos(beam_angle, number_of_data):
#     # interpolar search
#
#     step = math.pi * 2 / number_of_data
#     left = 0
#     right = number_of_data - 1
#     while left < right:
#         # TODO - precision!
#         middle = int((left + right) / 2)
#         if abs(beam_angle - (middle * step)) <= (step / 2) + 0.000001:
#             return middle
#         elif (middle * step) < beam_angle:
#             left = middle + 1
#         elif (middle * step) > beam_angle:
#             right = middle - 1
#         else:
#             return middle
#     return left