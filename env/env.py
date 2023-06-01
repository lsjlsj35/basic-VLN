import csv
import base64
from tqdm import tqdm
import math
import numpy as np
import networkx as nx
import sys
sys.path.append('/root/user/liushujun/VLN/selfmonitoring-agent/build')
sys.path.append('/root/user/liushujun/VLN/selfmonitoring-agent/tasks/my-r2r')
import MatterSim as ms
from utils.data import load_datasets, load_nav_graphs
from utils.word_processor import Tokenizer


NUM_pano_features = 10567  # 跑一遍得到的
BASE = "/root/user/liushujun/VLN/selfmonitoring-agent/"
PATH_img_feature = BASE + "img_features/ResNet-152-imagenet.tsv"

csv.field_size_limit(sys.maxsize)


def get_image_feature():
    features = {}
    cols = ('scanId', 'viewpointId', 'image_w', 'image_h', 'vfov', 'features')
    # "17DRP5sb8fy", "10c252...", 640, 480, 60,
    with open(PATH_img_feature, "r") as f:
        reader = csv.DictReader(f, delimiter='\t', fieldnames=cols)
        for l in tqdm(reader, total=NUM_pano_features):
            w = l["image_w"]
            h = l["image_h"]
            vfov = l["vfov"]
            key = l["scanId"] + '_' + l["viewpointId"]
            features[key] = np.frombuffer(base64.b64decode(l["features"]), dtype=np.float32).reshape((36, 2048))
    return features, (int(w), int(h), int(vfov))


class BatchPanoEnv:
    def __init__(self, img_features, img_spec, batchsize=64):
        self.features = img_features  # dict是动态变量，传递指针，所有不影响性能
        self.w, self.h, self.vfov = img_spec
        self.sims = []
        self.bs = batchsize
        for i in range(batchsize):
            sim = ms.Simulator()
            sim.setRenderingEnabled(False)
            sim.setDiscretizedViewingAngles(True)
            sim.setCameraResolution(self.w, self.h)
            sim.setCameraVFOV(math.radians(self.vfov))
            sim.init()
            self.sims.append(sim)

    def newEpisodes(self, scanIds, viewpointIds, headings):
        for i, (sc, vp, h) in enumerate(zip(scanIds, viewpointIds, headings)):
            self.sims[i].newEpisode(sc, vp, h, 0)

    def getFeatures(self):
        feats = []
        for sim in self.sims:
            state = sim.getState()
            imgID = state.scanId + '_' + state.location.viewpointId
            feat = self.features[imgID]
            feats.append((feat, state))
        return feats


class PanoEnvLoader:
    def __init__(self, config, batch_size=64, data_type=('train',), tokenizer=None, shuffle=True):
        img_feature, img_spec = get_image_feature()
        self.env = BatchPanoEnv(img_feature, img_spec, batch_size)
        # self.env = BatchPanoEnv(None, (640, 480, 60), batch_size)
        self.data = []
        self.scans = set()  # scene
        self.config = config
        self.data_type = data_type

        # 一对多拆成一对一
        datas = load_datasets(data_type)
        for i, item in tqdm(enumerate(datas)):
            for j, ins in enumerate(item["instructions"]):
                data = dict(item)
                data["ins_id"] = str(item["path_id"]) + '_' + str(j)
                data["instructions"] = ins
                if "instr_encoding" not in item:  # synthetic data里有
                    data["instr_encoding"] = tokenizer.encode(ins)
                self.data.append(data)
                self.scans.add(data["scan"])

        self.idx_list = np.arange(len(self.data))
        if shuffle:
            np.random.shuffle(self.idx_list)
        self.bs = batch_size
        self.idx = 0
        self.num_data = len(self.data)
        self.batch = []
        self.load_graph()

    def load_graph(self):
        self.graphs = load_nav_graphs(self.scans)
        # self.graphs = load_nav_graphs(["2t7WUuJeko7"])
        self.paths = {}  # {scan: {node: {goal: path, ...}, ...}, ...}
        self.dist = {}  # {scan: {node: {goal: length, ...}, ...}, ...}
        for scan, G in self.graphs.items():
            self.paths[scan] = {}
            self.dist[scan] = {}
            dijkstra = dict(nx.all_pairs_dijkstra(G))
            # 返回(node, {path})迭代器，即node到各点的路径距离
            for node, length_path in dijkstra.items():
                self.paths[scan][node] = length_path[1]
                self.dist[scan][node] = length_path[0]

    def load_next_env_batch(self):
        """
        data ==> self.batch
        """
        self.batch = []
        for i in self.idx_list[self.idx:self.idx+self.bs]:
            self.batch.append(self.data[i])
        if self.idx + self.bs >= self.num_data:
            num = self.idx + self.bs - self.num_data
            np.random.shuffle(self.idx_list)
            for i in range(num):
                self.batch.append(self.data[i])
            self.idx = num
            return True  # new epoch
        else:
            self.idx += self.bs
        return False

    def batch_to_sim(self):
        for i, data in enumerate(self.batch):
            self.env.sims[i].newEpisode(data["scan"], data["path"][0], data["heading"], 0)

    def get_vis_feat(self):
        return self.env.getFeatures()

    def simulate(self, scan, viewpoint, heading):
        if self.config["teleport"]:
            self.env.newEpisodes(scan, viewpoint, heading)
        else:
            raise NotImplementedError

    def return_to_gt_path(self, state, path):
        """
        return: path_id
        在gt path上返回下一个节点，否则返回最近的
        """
        vp = state.location.viewpointId
        for i, node in enumerate(path):
            if vp == node:
                return path[i+1] if len(path) > i+1 else path[i]
        nearest_node = None
        nearest_dist = 99999
        for node in path:
            d = self.dist[state.scanId][vp][node]
            if d < nearest_dist:
                nearest_dist = d
                nearest_node = node
        return self.paths[state.scanId][vp][nearest_node][1]

    def get_neighbor(self, state, goal):
        adj_point = self.graphs[state.scanId].adj[state.location.viewpointId]
        # return adj_point
        teacher_path = self.paths[state.scanId][state.location.viewpointId][goal]
        if len(teacher_path) > 1:
            next_gt_vp_id = teacher_path[1]
        else:
            next_gt_vp_id = state.location.viewpointId
            gt_vp_data = (state.location.viewpointId, state.viewIndex)

        neighbor = {}
        # 停止也行，故自己也是neighbor之一
        neighbor[state.location.viewpointId] = {
            "position": state.location.point,
            "heading": state.heading,
            "rel_heading": state.location.rel_heading,
            "rel_elevation": state.location.rel_elevation,
            "index": state.viewIndex
        }

        for vp_id, w in adj_point.items():
            tmp = {}
            vp = self.graphs[state.scanId].nodes[vp_id]
            pos_dif = vp["position"] - state.location.point
            tmp["position"] = vp["position"]  # ndarray [3,]
            target_heading = math.pi / 2.0 - math.atan2(pos_dif[1], pos_dif[0])
            if target_heading < 0:
                target_heading += 2.0 * math.pi
            tmp["rel_heading"] = target_heading - state.heading
            tmp["heading"] = target_heading

            dist = np.linalg.norm(pos_dif)
            rel_elevation = np.arcsin(pos_dif[2] / dist)
            tmp["rel_elevation"] = rel_elevation

            vp_horizontal_idx = int(round(target_heading / (math.pi / 6.0)))
            vp_horizontal_idx = 0 if vp_horizontal_idx == 12 else vp_horizontal_idx
            elevation_range = round(rel_elevation / (math.pi / 6.0)) + 1
            elevation_range = max(min(2, elevation_range), 0)
            vp_idx = int(vp_horizontal_idx + 12 * elevation_range)
            tmp["index"] = vp_idx

            if vp_id == next_gt_vp_id:
                gt_vp_data = (vp_id, vp_idx)
            neighbor[vp_id] = tmp
        return neighbor, gt_vp_data

    @staticmethod
    def get_angle_feature(state, tile=32):
        angle = 30  # degree

        hori_sin = []
        hori_cos = []
        for i in range(12):
            heading = i * angle * math.pi / 180 - state.heading
            heading = heading + 2 * math.pi if heading < 0 else heading
            hori_sin.append(math.sin(heading))
            hori_cos.append(math.cos(heading))
        sin_phi = np.array(hori_sin * 3)[:, None]
        cos_phi = np.array(hori_cos * 3)[:, None]

        verti_sin = []
        verti_cos = []
        for i in [-1, 0, 1]:
            elevation = i * angle * math.pi / 180 - state.elevation
            verti_sin.append(math.sin(elevation))
            verti_cos.append(math.cos(elevation))
        sin_theta = np.array(verti_sin * 12)[:, None]
        cos_theta = np.array(verti_cos * 12)[:, None]
        feat = np.concatenate([sin_phi, cos_phi, sin_theta, cos_theta], axis=1)
        return np.repeat(feat, tile, axis=1)

    def get_state(self):
        obs = []
        for i, (feat, state) in enumerate(self.env.getFeatures()):
            data = self.batch[i]
            goal = self.return_to_gt_path(state, data["path"]) if self.config["student_forcing"] \
                else data["path"][-1]
            neighbor, gt_vp_data = self.get_neighbor(state, goal)
            # gt_vp_data: (viewpointId, vp_index)

            # angle feature!
            ang_feat = self.get_angle_feature(state)  # [32, 128]
            feat = np.concatenate((feat, ang_feat), axis=1)

            # synthetic缺少ins_id
            if "synthetic" in self.data_type:
                data["ins_id"] = str(data["path_id"])
            obs.append({
                "ins_id": data["ins_id"],
                "scan": state.scanId,
                "viewpoint": state.location.viewpointId,
                "viewIndex": state.viewIndex,
                "heading": state.heading,
                "elevation": state.elevation,
                "feature": feat,
                "step": state.step,
                "neighbor": neighbor,
                "instruction": data["instructions"],
                "gt_path_all": data["path"],
                "nearest_path_to_goal": self.paths[state.scanId][state.location.viewpointId][data["path"][-1]],
                "next_gt_node_data": gt_vp_data,
                "instr_encoding": data["instr_encoding"]
            })

        return obs


if __name__ == "__main__":
    class tmp_loc:
        def __init__(self):
            self.viewpointId = "976c954bc5944db493d6149ee4bc7a2a"

    class tmp_mini_state:
        def __init__(self):
            self.scanId = "2t7WUuJeko7"
            self.location = tmp_loc()
    p = PanoEnvLoader(None, batch_size=1)
    # print(p.get_neighbor(tmp_mini_state(), None))
    # {
    # '073656b4030d4389ba7f304298be1d73': {'weight': 1.7736835481562092},
    # '06addff1d8274747b7a1957b2f03b736': {'weight': 3.15244947071321},
    # '8d51f41d9ce04dfdaae944da9c6d3847': {'weight': 1.7529038669020047}
    # }
    p.env.newEpisodes(["2t7WUuJeko7"], ["976c954bc5944db493d6149ee4bc7a2a"], [0])
    s = p.env.sims[0].getState()
    # print(p.graphs["2t7WUuJeko7"].nodes["976c954bc5944db493d6149ee4bc7a2a"])
    # {'position': array([3.252, -6.4332, 1.7182])}
    # print(type(s))  # ms.SimState
    # print(ms.SimState.__dict__)
    # {scanId, step, rgb, location, heading, elevation, viewIndex,
    # navigableLocations, viewIndex }
    # rgb不知道怎么使用
    #
    # print(s.scanId, s.step)
    # 2t7WUuJeko7 0
    #
    # print(s.location.viewpointId, s.location.point, s.location.rel_heading,
    #       s.location.rel_elevation, s.location.rel_distance)
    # 976c954bc5944db493d6149ee4bc7a2a
    # [3.2603299617767334, -6.421249866485596, 1.718269944190979]
    # 0.0     0.0     0.0
    #
    # print(s.heading, s.elevation)
    # 0.0     0.0
    #
    # print(s.viewIndex, s.navigableLocations)
    # 12, [ms.ViewPoint, ...]
    #
    # print(ms.ViewPoint.__dict__)
    # {viewpointId, point, rel_heading, rel_elevation, rel_distance}


