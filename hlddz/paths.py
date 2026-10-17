# -*- coding: utf-8 -*-
"""项目内资源路径集中管理。

历史问题：早期所有资源都用相对路径（'data.json'、'./pics'、'./weights'、'music/1.wav'、
'baselines/resnet/*.ckpt'），必须先把当前工作目录切到项目根目录才能运行。
从别处启动（双击 bat、IDE 工作目录配错、`pip install` 后从任意目录 import）时会
静默失败：图片找不到、甚至叫牌网络退回随机权重。

现在统一以**项目根目录**（本文件所在包目录的上一级）解析，任何工作目录下启动都能定位资源。
"""
from pathlib import Path

# hlddz/paths.py -> hlddz/ -> 项目根
ROOT = Path(__file__).resolve().parent.parent

DATA_JSON = ROOT / "data.json"
PICS_DIR = ROOT / "pics"
MUSIC_DIR = ROOT / "music"
WEIGHTS_DIR = ROOT / "weights"
BASELINES_DIR = ROOT / "baselines"
STYLE_QSS = ROOT / "style.qss"
LOGS_DIR = ROOT / "logs"


def resource(*parts) -> str:
    """返回项目内资源的绝对路径字符串（open / cv2.imread / torch.load / QFile 都接受）。"""
    return str(ROOT.joinpath(*parts))
