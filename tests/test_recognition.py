# -*- coding: utf-8 -*-
"""画面识别与坐标映射的回归测试（不依赖游戏窗口）。

覆盖：
  - 不同窗口尺寸的画面都能归一化到 1440x810
  - 规范画布坐标 -> 客户区坐标的映射
  - 牌面识别在 960/1152/1920 宽度下结果一致
  - 识别顺序为从左到右（click_cards 靠这个顺序与位置对齐）
"""
import os
import sys

import cv2
import numpy as np
import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app_paths import PICS_DIR  # noqa: E402


def _worker_without_init():
    """构造一个不做 __init__ 的 Worker：识别相关方法只用到类属性，避免加载模型。"""
    import mian
    return mian.Worker.__new__(mian.Worker)


def _make_frame(cards, spacing=46, x0=345, y0=594):
    """在 1440x810 画布上按从左到右的顺序贴出指定牌面。"""
    import mian  # noqa: F401  确保模块级 helper/模板已加载
    canvas = np.full((810, 1440, 3), (30, 110, 60), np.uint8)
    for i, c in enumerate(cards):
        tpl = cv2.imread(str(PICS_DIR / ("m%s.png" % c)))
        assert tpl is not None, c
        y, x = y0, x0 + i * spacing
        canvas[y:y + tpl.shape[0], x:x + tpl.shape[1]] = tpl
    return canvas


def _resize_roundtrip(frame_bgr, size):
    """模拟窗口被缩放：缩放到 size，再经过 normalize_frame 回到规范画布。"""
    import GameHelper

    small = cv2.resize(frame_bgr, size, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    back = GameHelper.GameHelper.normalize_frame(Image.fromarray(rgb))
    return cv2.cvtColor(np.asarray(back), cv2.COLOR_RGB2BGR)


def test_normalize_frame_always_returns_canvas():
    import GameHelper

    for size in [(960, 540), (1152, 648), (1280, 720), (1920, 1080), (1440, 810)]:
        im = Image.new("RGB", size, (10, 10, 10))
        out = GameHelper.GameHelper.normalize_frame(im)
        assert out.size == (1440, 810), size


def test_to_client_mapping():
    import GameHelper

    gh = GameHelper.GameHelper.__new__(GameHelper.GameHelper)
    gh.ClientSize = (1440, 810)
    assert gh.to_client((720, 405)) == (720, 405)

    gh.ClientSize = (960, 540)
    assert gh.to_client((1440, 810)) == (960, 540)
    assert gh.to_client((720, 405)) == (480, 270)

    gh.ClientSize = (1920, 1080)
    assert gh.to_client((720, 405)) == (960, 540)


def test_card_recognition_scale_invariance():
    w = _worker_without_init()
    cards = ["Q", "K", "A", "3", "4"]
    base = _make_frame(cards)

    pos = (180, 560, 1050, 90)
    assert w.find_cards(base, pos, mark="m") == "QKA34"

    for size in [(960, 540), (1152, 648), (1920, 1080)]:
        frame = _resize_roundtrip(base, size)
        got = w.find_cards(frame, pos, mark="m")
        assert got == "QKA34", "%s 下识别结果不同: %r" % (size, got)


def test_card_order_is_left_to_right():
    w = _worker_without_init()
    # AllCards 的类型顺序是 D X 2 A K Q J T 9 8 7 6 5 4 3，
    # 这里故意按大小乱序摆放，识别结果必须跟随“位置”而不是“类型顺序”。
    cards = ["3", "A", "5", "K", "4"]
    frame = _make_frame(cards)
    assert w.find_cards(frame, (180, 560, 1050, 90), mark="m") == "3A5K4"


def test_detail_positions_are_in_canvas_coordinates():
    w = _worker_without_init()
    frame = _make_frame(["3", "4"])
    detail = w.find_cards_detail(frame, (180, 560, 1050, 90), mark="m")
    assert len(detail) == 2
    assert [d[0] for d in detail] == ["3", "4"]
    # x 应为整幅画布坐标（345 / 391 附近），而不是区域内的 0/46
    assert 340 <= detail[0][1] <= 350
    assert 385 <= detail[1][1] <= 397


def test_multiscale_finds_small_bottom_cards():
    """底牌比手牌小，用 z 的多尺度匹配应能找到缩放到 0.5 的牌。"""
    w = _worker_without_init()
    cards = ["2", "5", "4"]
    canvas = np.full((810, 1440, 3), (30, 110, 60), np.uint8)
    x0, y0 = 679, 33
    for i, c in enumerate(cards):
        tpl = cv2.imread(str(PICS_DIR / ("m%s.png" % c)))
        tpl = cv2.resize(tpl, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        y, x = y0, x0 + i * 21
        canvas[y:y + tpl.shape[0], x:x + tpl.shape[1]] = tpl

    got = w.find_cards(canvas, (600, 33, 220, 103), mark="z", confidence=0.72)
    assert got == "254", got
