import torch.nn.functional as F
import torch.distributions as D
from torch import nn
import torch
import sys
import numpy as np
sys.path.append('/root/user/liushujun/VLN/selfmonitoring-agent/tasks/my-r2r')
from env.env import PanoEnvLoader
from utils.data import load_datasets


class BaseAgent:
    def __init__(self, env, log_path):
        self.env = env
        self.log_path = log_path
        self.results = {}

    def write_results(self):
        output = []
        for k, v in self.results.items():
            output.append(
                {
                    'instr_id': k,
                    'trajectory': v['path'],
                    'distance': v['distance'],
                    'img_attn': v['img_attn'],
                    'ctx_attn': v['ctx_attn'],
                    'value': v['value'],
                    'viewpoint_idx': v['viewpoint_idx'],
                    'navigable_idx': v['navigable_idx']
                }
            )
        with open(self.log_path, 'w') as f:
            json.dump(output, f)


class PanoAgent:
    def __init__(self, config, env, criterion, model, method="sample", epi_len=20, img_feat=2176, CE_weight=0.9,
                 device="cuda:0"):
        super(PanoAgent, self).__init__()
        self.config = config
        self.criterion = criterion
        self.mse = nn.MSELoss(reduction="sum")
        self.model = model
        self.m = method  # sample / max
        self.epi_len = epi_len
        self.env = env
        self.bs = self.env.bs
        self.img_feat = img_feat
        self.gt = {}
        self.device = device
        self.CE_weight = CE_weight
        for item in load_datasets(env.data_type):
            self.gt[int(item["path_id"])] = item

    def _get_distance(self, ob):
        try:
            gt = self.gt[int(ob['ins_id'].split('_')[0])]
        except:  # synthetic data only has 1 instruction per path
            gt = self.gt[int(ob['ins_id'])]
        distance = self.env.dist[ob['scan']][ob['viewpoint']][gt['path'][-1]]
        return distance

    def _next_viewpoint(self, obs, vp_id, neighbor_index, action, ended):
        vp = []
        heading = []
        vp_idx = []

        for i, ob in enumerate(obs):
            if action[i] > 0:
                vp_idx.append(neighbor_index[i][action[i] - 1])
            else:
                vp_idx.append("STAY")
                ended[i] = True
            next_point = vp_id[i][action[i]]
            vp.append(next_point)  # 第一个是自己，所以不用减一
            heading.append(ob["neighbor"][next_point]["heading"])
        return vp, heading, vp_idx, ended

    def init_traj(self, obs):
        """initialize the trajectory"""
        batch_size = len(obs)

        traj, scan_id = [], []
        for ob in obs:
            traj.append({
                'instr_id': ob['ins_id'],
                'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])],
                'length': 0,
                'feature': [ob['feature']],
                'img_attn': [],
                'ctx_attn': [],
                'value': [],
                'progress_monitor': [],
                'action_confidence': [],
                'regret': [],
                'viewpoint_idx': [],
                'navigable_idx': [],
                'distance': [self._get_distance(ob)]
            })
            scan_id.append(ob['scan'])

        self.longest_dist = [traj_tmp['distance'][0] for traj_tmp in traj]
        self.traj_length = [1] * batch_size
        self.value_loss = torch.tensor(0).float().to(self.device)

        ended = np.array([False] * batch_size)

        return traj, scan_id, ended

    def _update_traj(self, obs, traj, pm, next_vp_idx, neighbor_idx, ended):
        for i, ob in enumerate(obs):
            if not ended[i]:
                traj[i]["path"].append((ob['viewpoint'], ob['heading'], ob['elevation']))
                traj[i]["distance"].append(self._get_distance(ob))
                traj[i]["value"].append(pm[i].detach().cpu().item())
                traj[i]["viewpoint_idx"].append(next_vp_idx[i])
                traj[i]["navigable_idx"].append(neighbor_idx[i])
                traj[i]["length"] += 1
        return traj

    def pano_neighbor_feat(self, obs, ended):
        """
        obs: [{
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
        }]
        """
        num_feature, feature_size = obs[0]["feature"].shape
        pano_f = torch.zeros(len(obs), num_feature, feature_size)
        neighbor_f = torch.zeros(len(obs), 16, feature_size)
        neighbor_idx = []
        goal_idx = []
        vp = []
        for i, ob in enumerate(obs):
            pano_f[i, :] = torch.tensor(ob["feature"])
            gt_vp_id, gt_vp_idx = ob["next_gt_node_data"]
            neighbor_idx_one = []
            neighbor_id_one = []
            for j, vp_id in enumerate(ob["neighbor"].keys()):
                if vp_id == gt_vp_id:
                    if ended[i]:
                        goal_idx.append(16)  # ignore_index
                    else:
                        goal_idx.append(j)
                if vp_id == ob["viewpoint"]:
                    tag = j
                neighbor_id_one.append(vp_id)
                neighbor_idx_one.append(int(ob["neighbor"][vp_id]["index"]))

            neighbor_idx.append(neighbor_idx_one)
            vp.append(neighbor_id_one)
            neighbor_f[i, :len(neighbor_id_one)] = pano_f[i, neighbor_idx_one]
            neighbor_f[i, tag] = 0
        return pano_f, neighbor_f, (vp, neighbor_idx, goal_idx)

    def _select_action(self, pro, ended):
        pro = pro.to("cpu")
        # print(pro, pro.shape)
        if self.m == "sample":
            d = D.Categorical(pro)
            action = d.sample()
        elif self.m == "max":
            _, action = pro.max(-1)
            action = action.detach()
        else:
            raise ValueError
        for i, end in enumerate(ended):
            if end:
                action[i] = 0
        return action

    def monitor_loss(self, traj, pre, ended):
        gt = []
        for i, item in enumerate(traj):
            d0 = item["distance"][0]
            d = item["distance"][-1]
            loss = (d - d0) / d0
            gt.append(loss)
            if d <= 3.0:
                gt[-1] = 1
            if ended[i]:
                gt[-1] = pre[i].detach()
        gt = torch.tensor(gt).to("cuda:0")
        return self.mse(gt, pre.squeeze())

    def rollout_from_batch_one_epoch(self):
        flag = self.env.load_next_env_batch()
        self.env.batch_to_sim()
        obs = self.env.get_state()
        bs = len(obs)

        h, c = self.model.init_hidden_feat(bs)
        seq = torch.tensor([item["instr_encoding"] for item in obs]).to("cuda:0")
        seq_len = torch.tensor([len(s) for s in seq]).to("cuda:0")
        last_act_img = torch.zeros(bs, self.img_feat).to("cuda:0")
        traj, scan, ended = self.init_traj(obs)
        # pano_img_feat, neighbor_img_feat, neighbor_info = self.pano_neighbor_feat(ob, ended)
        # neighbor_id, neighbor_idx, goal_idx = neighbor_info
        # pano_img_feat = pano_img_feat.to("cuda:0")
        # neighbor_img_feat = neighbor_img_feat.to("cuda:0")
        # target = torch.tensor(goal_idx).to("cuda:0")

        loss = 0
        it = 0

        for _ in range(self.epi_len):
            it += 1
            pano_img_feat, neighbor_img_feat, neighbor_info = self.pano_neighbor_feat(obs, ended)
            neighbor_id, neighbor_idx, goal_idx = neighbor_info
            num_neighbor = [len(i)-1 for i in neighbor_idx]
            pano_img_feat = pano_img_feat.to("cuda:0")
            neighbor_img_feat = neighbor_img_feat.to("cuda:0")
            target = torch.tensor(goal_idx).to("cuda:0")

            h, c, logit, pm, mask = self.model(neighbor_img_feat, last_act_img, num_neighbor, seq, seq_len, h, c)
            logit = logit.masked_fill((mask == 0).data, float("-inf"))

            pro = F.softmax(logit, dim=-1)
            loss_pre = self.criterion(logit, target)
            action = self._select_action(pro, ended)
            loss_monitor = self.monitor_loss(traj, pm, ended)
            loss += self.CE_weight * loss_pre + (1 - self.CE_weight) * loss_monitor

            last_act_img = pano_img_feat[torch.arange(bs), action]
            next_vp, next_heading, next_vp_idx, ended = self._next_viewpoint(obs, neighbor_id,
                                                                             neighbor_idx, action, ended)
            if ended.all():
                break
            self.env.simulate(scan, next_vp, next_heading)
            obs = self.env.get_state()
            # traj = self._update_traj(obs, traj, pm, next_vp_idx, )

        return loss, traj, flag


class State:
    def __init__(self, score, last_act, h, c, pm, end, use_ppl=False):
        self.seq_len = len(self.idx_list)
        self.score = score  # log Prob
        self.last_act = last_act
        self.h = h  # [1, hidden_size]
        self.c = c
        self.pm = pm
        self.end = end
        self.use_ppl = use_ppl

    def __lt__(self, other):
        if self.use_ppl:
            return self.score/self.seq_len < other.score/other.seq_len
        else:
            return self.pm < other.pm

    def get_data_dict(self):
        """
        返回模型输入
        """
        pass


class BeamSearch:
    def __init__(self, model, seq_len, beam_size=10):
        self.model = model
        self.seq_len = seq_len
        self.beam_size = beam_size
        self.topk = []
        self.sm = nn.Softmax(dim=-1)

    def initial(self):
        pass

    def _create_state(self, state_dict, pm, p):
        """
        根据模型输出创建State
        """
        pass

    def _get_env(self):
        pass

    def iterate(self):
        cand = []
        for item in self.topk:
            if item.end:
                cand.append(item)
                continue
            data = item.get_data_dict()
            h, c, logit, pm, mask = self.model(**data)
            _, indices = self.sm(logit).topk(self.beam_size, axis=-1)
            state_dict = self.get_env()
            for i in range(self.beam_size):
                cand.append(self._create_state(state_dict, pm, logit[0][indices[i]]))
        if len(cand) == self.beam_size:
            self.topk = cand
            return True
        else:
            cand.sort()
            self.topk = cand[:self.beam_size]
            return False

    def calculate(self):
        self.initial()
        for _ in range(self.seq_len - 1):  # initial里有一次
            if self.iterate():
                break
        return [(state, state.score) for state in self.topk]

