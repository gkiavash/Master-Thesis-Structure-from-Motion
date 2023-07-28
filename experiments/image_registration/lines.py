"""
Crossroads detection using plane intersections
"""
import numpy as np
import open3d as o3d
import numpy
from itertools import combinations


planes = [
    [-0.2791221, 0.1138213, 0.95348599, -1.8268928],
    [0.95609529, 0.0459348, 0.28943355, 0.66691735],
    [-0.2801648, 0.1146654, 0.95307896, -0.6680058],
    [-0.0076903, 0.9937370, -0.1114791, -0.0822946],
    [0.96592917, 0.0499808, 0.25393455, 4.02921333],
    [-0.1117107, 0.9409524, 0.31957670,  -0.660217],
    [-0.0236206, 0.9962792, -0.0828843,  0.6345402],
]


def get_plane_plane_intersection(A, B):
    def norm2(X):
        return numpy.sqrt(numpy.sum(X ** 2))

    def normalized(X):
        return X / norm2(X)

    U = normalized(numpy.cross(A[:-1], B[:-1]))
    M = numpy.array((A[:-1], B[:-1], U))
    X = numpy.array((-A[-1], -B[-1], 0.))
    return U, numpy.linalg.solve(M, X)


points_ = []
lines = []
for combo in combinations(planes, 2):
    print(combo)
    a, b = combo
    # a, b = (1, -1, 0, 2), (-1, -1, 1, 3)
    u, v = get_plane_plane_intersection(a, b)
    print(u, v)

    if u.max() < 1 and v.max() < 1:
        pt0 = -10 * numpy.array(u) + numpy.array(v)
        pt1 = 10 * numpy.array(u) + numpy.array(v)
        ind0 = len(points_)
        points_.append(pt0)
        points_.append(pt1)
        lines.append([ind0, ind0+1])

colors_ = [[1, 0, 0] for i in range(len(lines))]
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(points_)
line_set.lines = o3d.utility.Vector2iVector(lines)
line_set.colors = o3d.utility.Vector3dVector(colors_)
o3d.visualization.draw_geometries([line_set])
