# -*- coding: utf-8 -*-
"""环境自检：窗口 / 权限 / 截图 / 模板识别。

调试“识别不到牌 / 截图全黑 / 点不动”时先跑这个，它会指出卡在哪一步：
    python tests/env_probe.py
注意：若游戏以管理员运行，本脚本也必须以管理员身份运行，否则第 3 步就会全黑。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import win32gui
import win32process


def main():
    print("=" * 60)
    # 1) 权限
    import ctypes
    admin = bool(ctypes.windll.shell32.IsUserAnAdmin())
    print("1) 管理员权限      :", admin)

    # 2) 窗口
    hwnd = win32gui.FindWindow("UnityWndClass", None)
    print("2) 游戏窗口        :", hwnd, "| 标题:", repr(win32gui.GetWindowText(hwnd)) if hwnd else "未找到")
    if not hwnd:
        print("   → 请先以窗口模式启动欢乐斗地主")
        return 1
    _, pid = win32process.GetWindowThreadProcessId(hwnd)
    print("   进程/尺寸       : pid=%s rect=%s minimized=%s"
          % (pid, win32gui.GetWindowRect(hwnd), bool(win32gui.IsIconic(hwnd))))
    if win32gui.IsIconic(hwnd):
        print("   → 窗口已最小化，请从任务栏恢复（允许被遮挡，但不能最小化）")

    # 3) 截图
    from PyQt5 import QtWidgets
    QtWidgets.QApplication([])
    import mian
    helper = mian.helper
    img, origin = helper.Screenshot()
    if img is None:
        print("3) 截图            : 失败（见上面的报错）")
        if not admin:
            print("   → 多半是权限：游戏以管理员运行时本脚本也必须提权")
        return 2
    arr = np.asarray(img)
    print("3) 截图            : OK %s | 原点 %s | 均值/标准差 %.1f / %.1f"
          % (img.size, origin, arr.mean(), arr.std()))
    if arr.std() < 1.0:
        print("   → 抓到的是黑图：窗口最小化，或权限不足（UIPI 拦截 PrintWindow）")
        return 3

    out = os.path.join(os.environ.get("TEMP", "."), "env_probe.png")
    img.save(out)
    print("   已保存          :", out)

    # 4) 关键模板
    full = (0, 0, 1440, 810)
    names = ["quick_start", "start_game", "continue", "reward", "over", "win", "lose",
             "chat", "tuoguan", "sure", "tuijian", "good", "zhidao", "chacha",
             "jiaodizhu_btn", "bujiao_btn", "qiangdizhu_btn", "buqiang_btn",
             "jiabei_btn", "bujiabei_btn", "chaojijiabei_btn", "mingpai_btn",
             "play_card", "pass_btn", "buchu", "laotou", "landlord"]
    print("4) 界面模板命中    :")
    hits = 0
    for name in names:
        if name not in helper.PicsCV:
            continue
        loc = helper.LocateOnScreen(name, region=full, confidence=0.85, img=img)
        if loc is not None:
            hits += 1
            print("     %-18s @ %s" % (name, loc))
    if not hits:
        print("     （无命中：可能仍在加载页/大厅）")

    # 5) 牌面识别
    w = mian.Worker()
    print("5) 手牌            :", w.find_my_cards() or "（空，尚未发牌）")
    print("   底牌            :", w.find_three_cards() or "（空）")
    print("   我的出牌区      :", w.find_played_cards(w.MPlayedCardsPos) or "（空）")
    print("   身份            :", w.find_landlord())
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
