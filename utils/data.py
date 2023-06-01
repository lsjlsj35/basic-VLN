# encoding: utf-8
from tqdm import tqdm
import json
import networkx as nx
import numpy as np


BASE = "/root/user/liushujun/VLN/selfmonitoring-agent/"
PATH_train_vocab = BASE + "tasks/R2R-pano/data/train_vocab.txt"
PATH_trainval_vocab = BASE + "tasks/R2R-pano/data/trainval_vocab.txt"
PATH_dataset_prefix = BASE + "tasks/R2R-pano/data/R2R_"
PATH_connectivity_graph_prefix = BASE + "connectivity/"


def get_vocab():
    with open(PATH_train_vocab) as f:
        vocab = [word.strip() for word in f.readlines()]
    return vocab


def load_datasets(s):
    """
    .json format:[{}, {}, ..., {}]
    dict format:{
        "distance": 11.66,
        "scan": "VLzqgDo317F",
        "path_id": 6250,
        "path": ['af3af33b...', '5be14...'],
        "heading": 3.751,
        "instructions": ['Walk down one ... on the landing.', 'Walk between ... and stop.']
        instructions为多条指令，针对同一系列动作
    }
    """
    data = []
    for name in s:
        if name == 'synthetic':
            with open(PATH_dataset_prefix + "literal_speaker_data_augmentation_paths.json") as f:
                data.extend(json.load(f))
        else:
            with open(PATH_dataset_prefix + name + ".json") as f:
                data.extend(json.load(f))
    return data


def load_nav_graphs(scans):
    def dist(p, q):
        return ((p["pose"][3] - q["pose"][3]) ** 2 +
                (p["pose"][7] - q["pose"][7]) ** 2 +
                (p["pose"][11] - q["pose"][11]) ** 2) ** 0.5

    graphs = {}
    for scan in tqdm(scans):
        with open(PATH_connectivity_graph_prefix + scan + "_connectivity.json") as f:
            G = nx.Graph()
            pos = {}
            data = json.load(f)
            # format: [{}, {}, ..., {}]
            # dict_keys ['image_id', 'pose', 'included': 是否包含在图中(?), 'visible', 'unobstructed', 'height']
            # type [str, list, bool, list, list, float]
            # ['...', 16 float numbers, T/F, 相对于此处是否可视, 相对于此处是否连通, float]
            for i, item in enumerate(data):
                if item["included"]:
                    for j, connect in enumerate(item["unobstructed"]):
                        if connect and data[j]["included"]:
                            pos[item["image_id"]] = np.array([item["pose"][3], item["pose"][7], item["pose"][11]])
                            G.add_edge(item["image_id"], data[j]["image_id"], weight=dist(item, data[j]))
            nx.set_node_attributes(G, values=pos, name="position")  # point.position <- pos[point_name]
            graphs[scan] = G
    return graphs


# load_nav_graphs(["17DRP5sb8fy"])

