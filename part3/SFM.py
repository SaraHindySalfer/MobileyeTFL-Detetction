from dataclasses import dataclass
from math import sqrt

import numpy as np


@dataclass
class Line:
    m: float
    n: float


def calc_TFL_dist(prev_container, curr_container, focal, pp):
    # calculates the distance between the traffic lights and the camera
    norm_prev_pts, norm_curr_pts, R, foe, tZ = prepare_3D_data(prev_container, curr_container, focal, pp)
    if abs(tZ) < 10e-6:
        print('tz = ', tZ)
    elif norm_prev_pts.size == 0:
        print('no prev points')
    elif norm_prev_pts.size == 0:
        print('no curr points')
    else:
        curr_container.corresponding_ind, curr_container.traffic_lights_3d_location, curr_container.valid = calc_3D_data(
            norm_prev_pts, norm_curr_pts, R, foe, tZ)

    return curr_container


def calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ):
    # calculates the distance
    norm_rot_pts = rotate(norm_prev_pts, R)
    pts_3D = []
    corresponding_ind = []
    validVec = []
    for p_curr in norm_curr_pts:
        corresponding_p_ind, corresponding_p_rot = find_corresponding_points(p_curr, norm_rot_pts, foe)
        Z = calc_dist(p_curr, corresponding_p_rot, foe, tZ)
        valid = (Z > 0)
        if not valid:
            Z = 0
        validVec.append(valid)
        P = Z * np.array([p_curr[0], p_curr[1], 1])
        pts_3D.append((P[0], P[1], P[2]))
        corresponding_ind.append(corresponding_p_ind)
    return corresponding_ind, np.array(pts_3D), validVec


def prepare_3D_data(prev_container, curr_container, focal, pp):
    # prepares the data by normalizing and decomposing the points
    norm_prev_pts = normalize(prev_container.traffic_light, focal, pp)
    norm_curr_pts = normalize(curr_container.traffic_light, focal, pp)
    R, foe, tZ = decompose(curr_container.EM)
    return norm_prev_pts, norm_curr_pts, R, foe, tZ


def normalize(pts, focal, pp):
    # transform pixels into normalized pixels using the focal length and principle point
    norm_pts = []
    for tfl in pts:
        norm_pts.append([(tfl[0] - pp[0]) / focal, (tfl[1] - pp[1]) / focal])
    return np.array(norm_pts)


def unnormalize(pts, focal, pp):
    # transform normalized pixels into original pixels using the focal length and principle point
    unnorm_pts = []
    for tfl in pts:
        unnorm_pts.append([(tfl[0] * focal) + pp[0], (tfl[1] * focal) + pp[1]])
    return np.array(unnorm_pts)


def decompose(EM):
    # extract R, foe and tZ from the Ego Motion
    R = EM[:3, :3]
    t = EM[:, 3]
    foe = (t[0] / t[2], t[1] / t[2])
    return R, foe, t[2]


def rotate(pts, R):
    # rotate the points - pts using R
    norm_rotate = []
    for tfl in pts:
        p_vec = np.array([tfl[0], tfl[1], 1])
        result = R.dot(p_vec)
        norm_rotate.append([result[0] / result[2], result[1] / result[2]])
    return np.array(norm_rotate)


def distance_point_to_line(point, line: Line):
    # calculates the distance between a line and a point
    return abs((line.m * point[0] + line.n - point[1]) / sqrt(line.m ** 2 + 1))


def find_corresponding_points(p, norm_pts_rot, foe):
    # compute the epipolar line between p and foe
    # run over all norm_pts_rot and find the one closest to the epipolar line
    # return the closest point and its index
    m = (foe[1] - p[1]) / (foe[0] - p[0])
    n = (p[1] * foe[0] - foe[1] * p[0]) / (foe[0] - p[0])
    distances = [distance_point_to_line(point, Line(m, n))
                 for point in norm_pts_rot]
    return distances.index(min(distances)), norm_pts_rot[distances.index(min(distances))]


def calc_dist(p_curr, p_rot, foe, tZ):
    # calculate the distance of p_curr using x_curr, x_rot, foe_x and tZ
    # calculate the distance of p_curr using y_curr, y_rot, foe_y and tZ
    # combine the two estimations and return estimated Z
    return calc_z(foe, p_curr, p_rot, tZ, choose_index(p_curr, p_rot))


def choose_index(p_curr, p_rot):
    if p_curr[0] - p_rot[0] > p_curr[1] - p_rot[1]:
        z_index = 0
    else:
        z_index = 1
    return z_index


def calc_z(foe, p_curr, p_rot, tZ, z_index):
    # calculate the Z distance
    return (tZ * (foe[z_index] - p_rot[z_index])) / (p_curr[z_index] - p_rot[z_index])


def get_foe_rotate(prev_container, curr_container, focal, pp):
    # return the rotated values of the foe
    norm_prev_pts, norm_curr_pts, R, norm_foe, tZ = prepare_3D_data(prev_container, curr_container, focal, pp)
    norm_rot_pts = rotate(norm_prev_pts, R)
    rot_pts = unnormalize(norm_rot_pts, focal, pp)
    foe = np.squeeze(unnormalize(np.array([norm_foe]), focal, pp))
    return foe, rot_pts
