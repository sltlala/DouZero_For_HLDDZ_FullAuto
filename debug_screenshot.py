"""坐标调试脚本：截图并在手牌识别区域画框，用来核对 1440x810 的截图区域常量。

用法：先让欢乐斗地主以窗口模式运行，再执行 `python debug_screenshot.py`。
"""
import GameHelper as gh
from GameHelper import GameHelper
import cv2
import numpy as np

GameHelper = GameHelper()

# img = cv2.imread("3.png")
# img = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2RGB)
img, origin = GameHelper.Screenshot()
if img is None:
    # 没找到 UnityWndClass 窗口时 Screenshot 会返回 (None, (0, 0))，旧代码会在这里崩
    raise SystemExit("没找到游戏窗口：请确认欢乐斗地主已以窗口模式运行（类名 UnityWndClass）")
img = gh.DrawRectWithText(img, (180, 560, 1050, 90), "test")
gh.ShowImg(img)
