from matplotlib import pyplot as plt

from part4.candidates import Candidates


def visual(candidates_lights: Candidates, candidates_tfl: Candidates, distances, rot_pts, foe):
    if distances != 0:
        green_x, green_y, red_x, red_y = separate_by_color(candidates_lights)
        im_lights = plt.imread(candidates_lights.frame_path)
        fig, (lights_sec, tfl_sec, distance_sec) = plt.subplots(1, 3, figsize=(12, 6))
        fig.suptitle(f'frame: {candidates_tfl.frame_path}')
        lights_sec.set_title('candidates')
        lights_sec.imshow(im_lights)
        lights_sec.plot(red_x, red_y, 'ro', color='r', markersize=4)
        lights_sec.plot(green_x, green_y, 'ro', color='g', markersize=4)
        green_x, green_y, red_x, red_y = separate_by_color(candidates_tfl)
        im_tfl = plt.imread(candidates_tfl.frame_path)
        tfl_sec.set_title('traffic lights')
        tfl_sec.imshow(im_tfl)
        tfl_sec.plot(red_x, red_y, 'ro', color='r', markersize=4)
        tfl_sec.plot(green_x, green_y, 'ro', color='g', markersize=4)
        distance_sec.set_title('distances')
        distance_sec.imshow(distances.img)
        curr_p = distances.traffic_light
        for i in range(len(curr_p)):
            distance_sec.plot([curr_p[i, 0], foe[0]], [curr_p[i, 1], foe[1]], 'b')
            if distances.valid[i]:
                distance_sec.text(curr_p[i, 0], curr_p[i, 1],
                                  r'{0:.1f}'.format(distances.traffic_lights_3d_location[i, 2]), color='r')
        distance_sec.plot(foe[0], foe[1], 'r+')
        distance_sec.plot(rot_pts[:, 0], rot_pts[:, 1], 'g+')
        plt.show()


def separate_by_color(candidates_lights):
    red_x = [point[0] for ind, point in enumerate(candidates_lights.points) if candidates_lights.auxiliary[ind] == 1]
    red_y = [point[1] for ind, point in enumerate(candidates_lights.points) if candidates_lights.auxiliary[ind] == 1]
    green_x = [point[0] for ind, point in enumerate(candidates_lights.points) if candidates_lights.auxiliary[ind] == 0]
    green_y = [point[1] for ind, point in enumerate(candidates_lights.points) if candidates_lights.auxiliary[ind] == 0]
    return green_x, green_y, red_x, red_y
