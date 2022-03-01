import pickle

from part4.tfl_manager import TflManager


class Controller:
    def __init__(self, pls: str):
        self.pls_path = pls

        with open(pls, 'r', encoding="utf8") as file_reader:
            data = file_reader.read().split('\n')
            pkl_path = data[0]
        self.frame_list = [data[path] for path in range(1, len(data))][:-1]

        with open(pkl_path, 'rb') as pklfile:
            meta_data = pickle.load(pklfile, encoding='latin1')
            egomotion = [meta_data['egomotion_' + str(em + 24) + '-' + str(em + 1 + 24)] for em in
                         range(len(self.frame_list) - 1)]
        self.tfl_manage = TflManager(meta_data['principle_point'], meta_data["flx"], egomotion)

    def run(self):
        for index, frame in enumerate(self.frame_list):
            self.tfl_manage.on_frame(frame, index)

