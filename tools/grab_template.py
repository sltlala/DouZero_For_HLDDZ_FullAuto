# -*- coding: utf-8 -*-
"""按钮模板采集工具：把鼠标移到按钮上，自动裁一块存成 pics/<名字>.png。

为什么需要它：新版客户端（含单机模式）的按钮外观与仓库里旧的 pics/*.png 不一致，
模板匹配只有 0.4~0.5 分，程序认不出按钮就点不了。用这个工具对着当前客户端
重新采一次模板即可，采集完立刻用同屏自检，分数应接近 1.0。

用法（必须用**管理员权限**的终端，因为游戏是提权运行的）：

    python tools/grab_template.py jiaodizhu_btn
    python tools/grab_template.py bujiao_btn
    ...

可选参数：宽度 高度（默认 120x44，按按钮实际大小调整）

    python tools/grab_template.py play_card 150 52

采集时的操作：运行后会有 3 秒倒计时，把鼠标**停在按钮正中间**不要动。
"""
import ctypes
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PICS = ROOT / "pics"


def enable_dpi_awareness():
    for call in (lambda: ctypes.windll.user32.SetProcessDpiAwarenessContext(-4),
                 lambda: ctypes.windll.shcore.SetProcessDpiAwareness(2),
                 lambda: ctypes.windll.user32.SetProcessDPIAware()):
        try:
            call()
            return True
        except Exception:
            continue
    return False


def grab_screen_bgr():
    from PIL import ImageGrab

    img = np.array(ImageGrab.grab(all_screens=True))
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    name = sys.argv[1].replace(".png", "")
    width = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    height = int(sys.argv[3]) if len(sys.argv) > 3 else 44
    delay = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0

    enable_dpi_awareness()
    import win32api

    print("目标模板: %s  尺寸: %dx%d" % (name, width, height))
    print("把鼠标移到按钮正中间，%.0f 秒后自动截图..." % delay)
    for i in range(int(delay), 0, -1):
        print("  %d" % i, flush=True)
        time.sleep(1)

    x, y = win32api.GetCursorPos()
    print("鼠标位置:", (x, y))
    screen = grab_screen_bgr()
    sh, sw = screen.shape[:2]
    print("屏幕尺寸:", (sw, sh))

    x0 = max(0, min(sw - width, x - width // 2))
    y0 = max(0, min(sh - height, y - height // 2))
    crop = screen[y0:y0 + height, x0:x0 + width]
    if crop.size == 0:
        print("裁剪失败")
        return 2

    PICS.mkdir(parents=True, exist_ok=True)
    out = PICS / (name + ".png")
    cv2.imwrite(str(out), crop)
    print("已保存:", out)

    # 同屏自检：新模板立刻回匹配一次，分数接近 1.0 才算采对
    res = cv2.matchTemplate(screen, crop, cv2.TM_CCOEFF_NORMED)
    mv, _, ml, _ = cv2.minMaxLoc(res)
    print("自检匹配分: %.4f @ %s （应接近 1.0）" % (mv, ml))
    if mv < 0.95:
        print("警告: 自检分数偏低，可能截到了动态内容（动画/闪烁），建议重采")
    print("完成。全部按钮都采完后重启程序即可识别。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
