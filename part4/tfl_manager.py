from dataclasses import dataclass

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model

from part1.part1_api import find_tfl_lights
from part2.createDataSet import crop_image
from part3.SFM_original import calc_TFL_dist, get_foe_rotate
from part4.candidates import Candidates
from part4.visualation import visual


class FrameContainer(object):
    def __init__(self, img_path, traffic_lights):
        self.img = plt.imread(img_path)
        self.traffic_light = traffic_lights
        self.traffic_lights_3d_location = []
        self.EM = []
        self.corresponding_ind = []
        self.valid = []

@dataclass
class TflManager:
    def __init__(self, pp, focal, egomotion):
        self.principle_point = pp
        self.focal = focal
        self.em = egomotion
        # importing the model from part 2
        self.my_model = load_model("./part2/model.h5")
        self.prev_candidates = Candidates("", [], [])

    def on_frame(self, frame_path, index):
        # finding lights in frame (part1) and traffic lights from part 2
        lights_candidates, tfl_candidates = self.find_tfl(frame_path)
        if index == 0:
            distances, rot_pts, foe = 0, 0, 0
        else:
            distances, foe, rot_pts = self.calc_distance(tfl_candidates, index)
        visual(lights_candidates, tfl_candidates, distances, rot_pts, foe)
        self.prev_candidates = tfl_candidates

    def find_tfl(self, frame_path):
        # returns points of lights from part 1 and points of tfl from part 2
        # call part1
        print(frame_path,"frame")
        can, aux = self.find_lights(frame_path)
        # made the result as candidates object
        lights_candidates = Candidates(frame_path, can, aux)
        can, aux = self.recognize_tfl(lights_candidates)
        tfl_candidates = Candidates(frame_path, can, aux)
        return lights_candidates, tfl_candidates

    def find_lights(self, frame) -> (list, list):
        # calling function that finds the lights in part 1
        image = np.array(Image.open(frame))
        return find_tfl_lights(image)

    def recognize_tfl(self, candidate: Candidates) -> (list, list):
        # calling crop function from part 2
        croped_images = [crop_image(candidate.frame_path, point[0], point[1]) for point in candidate.points]
        # using model from part 2 to predict tfl
        predictions = self.my_model.predict(np.array(croped_images))
        tfl_array = []
        auxiliary = []
        for index, predict in enumerate(predictions[:, 1]):
            if predict > 0.5:
                tfl_array.append(candidate.points[index])
                auxiliary.append(candidate.auxiliary[index])
        return tfl_array, auxiliary

    def calc_distance(self, cur_frame: Candidates, index: int):
        # using part 3 to calculate distances
        prev_container = FrameContainer(self.prev_candidates.frame_path, np.array(self.prev_candidates.points))
        curr_container = FrameContainer(cur_frame.frame_path, np.array(cur_frame.points))
        curr_container.EM = self.em[index - 1]
        # using part 3 functions
        z = calc_TFL_dist(prev_container, curr_container, self.focal, self.principle_point)
        foe, rot_pts = get_foe_rotate(prev_container, curr_container, self.focal, self.principle_point)
        return z, foe, rot_pts
