import math

import numpy as np


def calc_TFL_dist(prev_container, curr_container, focal, pp):
    norm_prev_pts, norm_curr_pts, R, foe, tZ = prepare_3D_data(prev_container, curr_container, focal, pp)
    if abs(tZ) < 10e-6:
        print('tz = ', tZ)
    elif norm_prev_pts.size == 0:
        print('no prev points')
    elif norm_curr_pts.size == 0:
        print('no curr points')
    else:
        curr_container.corresponding_ind, curr_container.traffic_lights_3d_location, curr_container.valid = calc_3D_data(
            norm_prev_pts, norm_curr_pts, R, foe, tZ)
    return curr_container


def prepare_3D_data(prev_container, curr_container, focal, pp):
    norm_prev_pts = normalize(prev_container.traffic_light, focal, pp)
    norm_curr_pts = normalize(curr_container.traffic_light, focal, pp)
    R, foe, tZ = decompose(np.array(curr_container.EM))
    return norm_prev_pts, norm_curr_pts, R, foe, tZ


def calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ):
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


def normalize(pts, focal, pp):
    # transform pixels into normalized pixels using the focal length and principle point
    norm_pixel = []
    for pxl in pts:
        norm_pixel += [(pxl[0] - pp[0]) / focal, (pxl[1] - pp[1]) / focal]
    return np.array(norm_pixel).reshape(len(pts),2)


def unnormalize(pts, focal, pp):
    # transform normalized pixels into pixels using the focal length and principle point
    unnorm_pxl = []
    for pix in pts:
        x, y = pix[0], pix[1]
        unnorm_pxl += [x * focal + pp[0], y * focal + pp[1]]
    return np.array(unnorm_pxl).reshape(len(pts),2)


def decompose(EM):
    # extract R, foe and tZ from the Ego Motion
    EM = EM[:3, :]
    t = EM[:, [3]]
    tZ = t[2]
    R = EM[:, :3]
    foe = (t[0] / t[2], t[1] / t[2])
    return R, foe, tZ


def rotate(pts, R):
    return [np.matmul(R, np.array((pt[0], pt[1], 1))) for pt in pts]


def find_corresponding_points(p, norm_pts_rot, foe):
    # compute the epipolar line between p and foe
    # run over all norm_pts_rot and find the one closest to the epipolar line
    # return the closest point and its index
    ey, ex = foe[0], foe[1]
    x_tilda, y_tilda = p[0], p[1]
    m = (ey - y_tilda) / (ex - x_tilda)
    n = ((y_tilda * ex) - (ey * x_tilda)) / (ex - x_tilda)
    min_dist = math.inf
    idx = 0
    for i, pxl in enumerate(norm_pts_rot):
        x, y = pxl[0], pxl[1]
        d = abs((m * x + n - y) / math.sqrt(m ** 2 + 1))
        if min_dist > d:
            min_dist = d
            idx = i
    return idx, norm_pts_rot[idx]


def calc_dist(p_curr, p_rot, foe, tZ):
    # calculate the distance of p_curr using x_curr, x_rot, foe_x and tZ
    # calculate the distance of p_curr using y_curr, y_rot, foe_y and tZ
    # combine the two estimations and return estimated Z
    ZX = tZ * (foe[0] - p_rot[0]) / (p_curr[0] - p_rot[0])
    ZY = tZ * (foe[1] - p_rot[1]) / (p_curr[1] - p_rot[1])
    x_dist = abs(p_curr[0] - p_rot[0])
    y_dist = abs(p_curr[1] - p_rot[1])
    return (ZX * x_dist + ZY * y_dist) / (x_dist + y_dist)


def get_foe_rotate(prev_container, curr_container, focal, pp):
    norm_prev_pts, norm_curr_pts, R, norm_foe, tZ = prepare_3D_data(prev_container, curr_container, focal, pp)
    norm_rot_pts = rotate(norm_prev_pts, R)
    rot_pts = unnormalize(norm_rot_pts, focal, pp)
    foe = np.squeeze(unnormalize(np.array([norm_foe]), focal, pp))
    return foe, rot_pts
