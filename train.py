import torch
from torch import nn
import numpy as np
import os
import random
from tqdm import tqdm

from utils.data import get_vocab
from utils.word_processor import Tokenizer
from env.agent import PanoEnvLoader
from model.backbone import MonitorNet
from env.agent import PanoAgent


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True


def main():
    set_seed()
    device = "cuda:0"
    config = {"teleport": True, "student_forcing": True}

    vocab = get_vocab()
    tokenizer = Tokenizer(vocab, seq_len=80)
    padding_idx = 0  # index of <PAD>
    # encoder: (self, vocab_size, embedding_size=256, hidden_size=512, padding_idx=None,
    #                  dropout_ratio=0.5, bidirectional=False, num_layers=1)
    # PanoEnvLoader: (self, config, batch_size=64, data_type=('train',), tokenizer=None, shuffle=True)

    model = MonitorNet(vocab_size=len(vocab), padding_idx=padding_idx).to(device)
    env = PanoEnvLoader(config, tokenizer=tokenizer)
    criterion = nn.CrossEntropyLoss(ignore_index=16)
    agent = PanoAgent(config, env, criterion, model)
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-4)

    res = []
    pbar = tqdm(range(1, 101))
    for epoch in pbar:
        flag = False
        while not flag:
            model.train()
            optimizer.zero_grad()
            loss, traj, flag = agent.rollout_from_batch_one_epoch()
            loss.backward()
            optimizer.step()
            res.append(loss.item())
            pbar.set_description(f"loss:{loss.item()}")
    with open("./result1.txt", "w") as f:
        for i, lo in enumerate(res):
            f.write(f"{i}    {lo}\n")
    torch.save(model.state_dict(), "./tasks/result/2.pth")
    return


if __name__ == "__main__":
    main()






