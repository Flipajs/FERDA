def on_segment(p1, p2, p3):
    return max(p1[0], p3[0]) >= p2[0] >= min(p1[0], p3[0]) and \
           max(p1[1], p3[1]) >= p2[1] >= min(p1[1], p3[1])


def orientation(p1, p2, p3):
    value = (p2[1] - p1[1]) * (p3[0] - p2[0]) - \
            (p2[0] - p1[0]) * (p3[1] - p2[1])

    if value == 0:
        return 0

    return 1 if value > 0 else 2


def do_intersect(pointA, pointB, pointC, pointD):
    o1 = orientation(pointA, pointB, pointC)
    o2 = orientation(pointA, pointB, pointD)
    o3 = orientation(pointC, pointD, pointA)
    o4 = orientation(pointC, pointD, pointB)

    if o1 != o2 and o3 != o4:
        return True

    return o1 == 0 and on_segment(pointA, pointC, pointB) or o2 == 0 and on_segment(pointA, pointD,
                                                                                    pointB) or o3 == 0 and on_segment(
        pointC, pointA, pointD) or o4 == 0 and on_segment(pointC, pointB, pointD)


if __name__ == "__main__":
    points = [((0, 0), (1, 1), (1, 1), (0, 1)),
              ((1, 1), (10, 1), (1, 2), (10, 2)),
              ((10, 0), (0, 10), (0, 0), (10, 10)),
              ((-5, -5), (0, 0), (1, 1), (10, 10))]
    for pointA, pointB, pointC, pointD in points:
        print do_intersect(pointA, pointB, pointC, pointD)
