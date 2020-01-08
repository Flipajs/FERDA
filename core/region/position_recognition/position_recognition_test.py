import numpy
from numpy import transpose
from core.project.project import Project
from numpy.linalg import eig
import math
import matplotlib.pyplot as plt

NUMBER_OF_DATA = 100
PRECISION = 3

def get_pca(chunks, number_of_data, chm, gm):
    matrix = []
    for ch in chunks:
        for vector in get_matrix(ch, number_of_data, chm, gm):
            matrix.append(vector)
    # for a in matrix:
    #     print(a)
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


def get_chunks_regions(ch, chm, gm):
    chunk = chm[ch]
    chunk_start = chunk.start_frame()
    chunk_end = chunk.end_frame()
    while chunk_start <= chunk_end:
        yield project.gm.region(chunk[chunk_start])
        chunk_start += 1


def get_matrix(chunk, number_of_data, chm, gm):
    matrix = []
    for region in get_chunks_regions(chunk, chm, gm):
        matrix.append(get_region_vector_curve(region, number_of_data))
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


def get_region_vector_curve(region, number_of_data):
    centroid = region.centroid()
    contour = region.contour_without_holes()
    size = len(contour)
    if size < number_of_data:
        print("Number of data should be smaller than contours length")
    curve_length = compute_contour_perimeter(contour)
    step = curve_length / number_of_data - 1
    result = []
    current = 0
    length = 0
    last = contour[size - 1]
    points_aux = []
    for i in range(size):
        print(i)
        curr = contour[i]
        print(curr)
        points_aux.append(curr)
        diff = vector_norm(last - curr)
        if length + diff > current * step:
            result.append(get_interpolation_curve(vector_norm(last - centroid), vector_norm(curr - centroid),
                                                       current * step - length, (length + diff) - current * step))
            current += 1
        length += diff
        last = curr

    plt.hold(True)
    plt.axis('equal')
    xs = [x[1] for x in points_aux]
    ys = [x[0] for x in points_aux]
    xa = [step * x for x in range(number_of_data)]
    ya = [result[x] for x in range(number_of_data)]
    plt.scatter(xs, ys, c="c")
    plt.scatter(xa, ya)
    plt.show(True)
    return result


def get_interpolation_curve(distance_a, distance_b, length_a, length_b):
    return (distance_a * length_a + distance_b * length_b) / (length_a + length_b)

def compute_contour_perimeter(contour):
    length = len(contour)
    perimeter = vector_norm(contour[0] - contour[length - 1])
    for i in range(1, length):
        length += vector_norm(contour[i] - contour[i - 1])
    return perimeter


def get_region_vector_angle(region, number_of_data):
    centroid = region.centroid()
    contours = region.contour()

    # table of the closest lower and higher deviated beams
    results = [[None, None]for x in range(number_of_data)]

    step = 2 * math.pi / number_of_data

    plt.figure()
    points_aux = []
    points_aux.append([0, 0])

    head_beam = contours[0] - centroid
    head_ang = deviation(head_beam)

    con_index = 0
    while con_index < len(contours):
        beam = contours[con_index] - centroid
        points_aux.append(beam)
        dev = deviation(beam)
        beam_angle = dev - head_ang
        if beam_angle < 0:
            beam_angle += math.pi * 2
        distance = vector_norm(beam)

        closest_angle_pos = find_closest_angle_pos(beam_angle, number_of_data)
        closest_angle = closest_angle_pos * step
        print("Angle: " + str(math.degrees(closest_angle)))
        print(distance)
        #TODO - precision
        #TODO - precision multiplier
        if (closest_angle - beam_angle) < 0.00001:
            for i in [0, 1]:
                data_entry = results[closest_angle_pos][i]
                if data_entry is None or data_entry[1] < distance:
                    results[closest_angle_pos][i] = (beam_angle, distance)
        elif closest_angle < beam_angle:
            highest_lower = results[closest_angle_pos][0]
            if highest_lower is None or highest_lower[1] < distance or highest_lower[0] < beam_angle:
                results[closest_angle_pos][0] = (beam_angle, distance)
        else:
            lowest_higher = results[closest_angle_pos][1]
            if lowest_higher is None or lowest_higher[1] < distance or lowest_higher[0] > beam_angle:
                results[closest_angle_pos][1] = (beam_angle, distance)
        con_index += 1

    for i in range(number_of_data):
        print(math.degrees(step * i))
        print(results[i][0])
        print(results[i][1])

    for i in range(number_of_data):
        if results[i][0] is None:
            results[i][0] = (step * i, 0)
        if results[i][1] is None:
            results[i][1] = (step * i, 0)

    for i in range(number_of_data):
        print(step * i * (180 / math.pi))
        print(results[i][0])
        print(results[i][1])

    results = [interpolation(h_l[0], l_h[0], h_l[1], l_h[1]) for h_l, l_h in results]
    plt.scatter(list(range(100, number_of_data + 100)), [results[x] for x in range(number_of_data)], c='r')

    plt.hold(True)
    plt.axis('equal')
    xs = [x[1] for x in points_aux]
    ys = [x[0] for x in points_aux]
    plt.scatter(xs, ys, c="c")

    res_aux = []
    for i in range(number_of_data):
        y = math.sin(step * i) * results[i]
        x = math.cos(step * i) * results[i]
        res_aux.append((x, y))

    xs = [x[1] for x in res_aux]
    ys = [x[0] for x in res_aux]
    plt.scatter(xs, ys)

    plt.show(True)

    return results


def vector_norm(u):
    return math.sqrt(sum(i ** 2 for i in u))


def dot_product(u, v):
    return sum((x * y) for x, y in zip(u, v))


def deviation(u):
    ang = math.atan2(u[1], u[0])
    if ang < 0:
        ang += 2 * math.pi
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
    project = Project()
    project.load("/home/sheemon/FERDA/projects/Cam1_/cam1.fproj")
    chunks = project.chm.chunk_list()
    get_pca(chunks[:1], NUMBER_OF_DATA, project.chm, project.gm)
