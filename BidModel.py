# -*- coding: utf-8 -*-
# Created by: Vincentzyx
#
# 叫牌/加倍用的“手牌 -> 得分”回归网络。
#
# 历史问题（已修复）：
#   本文件曾同时定义 Net(全连接 fc1..fc6) 与 Net2(卷积 conv1 + dense1..5)，
#   而 weights/bid_weights.pkl 里保存的键名是 conv1.* / dense1..5.*，即 Net2 的权重；
#   旧代码把权重复制进 net2，却用从未加载过权重的 net 去做 predict()，
#   等于用随机初始化的网络出分。现在只保留与权重匹配的 Net2，并删除那个失效的 predict()。
import os

import torch
from torch import nn
import torch.nn.functional as F

from app_paths import WEIGHTS_DIR

# 权重文件（绝对路径，不依赖当前工作目录）
WEIGHTS_PATH = str(WEIGHTS_DIR / "bid_weights.pkl")


def EnvToOnehot(cards):
    Env2IdxMap = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 11: 8, 12: 9, 13: 10, 14: 11, 17: 12, 20: 13, 30: 14}
    cards = [Env2IdxMap[i] for i in cards]
    Onehot = torch.zeros((4, 15))
    for i in range(0, 15):
        Onehot[:cards.count(i), i] = 1
    return Onehot


def RealToOnehot(cards):
    RealCard2EnvCard = {'3': 0, '4': 1, '5': 2, '6': 3, '7': 4,
                        '8': 5, '9': 6, 'T': 7, 'J': 8, 'Q': 9,
                        'K': 10, 'A': 11, '2': 12, 'X': 13, 'D': 14}
    cards = [RealCard2EnvCard[c] for c in cards]
    Onehot = torch.zeros((4, 15))
    for i in range(0, 15):
        Onehot[:cards.count(i), i] = 1
    return Onehot


class Net2(nn.Module):
    """输入 1*60（4*15 的持牌 one-hot 展平），输出一个标量得分。"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=(3,), padding=1)
        self.dense1 = nn.Linear(1020, 1024)
        self.dense2 = nn.Linear(1024, 512)
        self.dense3 = nn.Linear(512, 256)
        self.dense4 = nn.Linear(256, 128)
        self.dense5 = nn.Linear(128, 1)

    def forward(self, xi):
        x = xi.unsqueeze(1)
        x = F.leaky_relu(self.conv1(x))
        x = x.flatten(1, 2)
        x = torch.cat((x, xi), 1)
        x = F.leaky_relu(self.dense1(x))
        x = F.leaky_relu(self.dense2(x))
        x = F.leaky_relu(self.dense3(x))
        x = F.leaky_relu(self.dense4(x))
        x = self.dense5(x)
        return x


UseGPU = torch.cuda.is_available()
device = torch.device('cuda:0') if UseGPU else torch.device('cpu')

net2 = Net2()


def _load_weights(model, path):
    """加载权重，并把“键名不匹配”变成显式报错，避免再出现静默使用随机权重的情况。"""
    if not os.path.exists(path):
        print("[BidModel] 警告: 未找到 {}，叫牌得分将是随机值！".format(path))
        return False
    state = torch.load(path, map_location=device)
    own = model.state_dict()
    matched = {k: v for k, v in state.items() if k in own and tuple(v.shape) == tuple(own[k].shape)}
    if len(matched) != len(own):
        missing = [k for k in own if k not in matched]
        print("[BidModel] 警告: {} 与网络结构不匹配，缺少 {} 个参数: {}".format(path, len(missing), missing[:5]))
    model.load_state_dict(matched, strict=False)
    return len(matched) == len(own)


_loaded = _load_weights(net2, WEIGHTS_PATH)
net2.eval()
if UseGPU:
    net2 = net2.to(device)


def predict_score(cards):
    """返回叫牌/加倍用的原始得分（界面上的“叫牌得分”），阈值见 data.json 的 bid1~bid3。"""
    input = RealToOnehot(cards)
    if UseGPU:
        input = input.to(device)
    input = torch.flatten(input)
    input = input.unsqueeze(0)
    with torch.no_grad():
        result = net2(input)
    return result[0].item()


if __name__ == "__main__":
    print("weights loaded:", _loaded)
    print(predict_score("333444569TTJJQKK2"))
