# -*- coding: utf-8 -*-
# @Time : 2024/2/14 17:06
# @Author : MaYun
# @File : GameHelper.py
# @Software: PyCharm
import ctypes
import json
import win32gui
import win32ui
import win32api
import win32con
from ctypes import windll
from PIL import Image
import cv2
import pyautogui
import pygame
import matplotlib.pyplot as plt
import numpy as np
import os
import time

from ..paths import DATA_JSON, PICS_DIR, resource


def enable_dpi_awareness():
    """让本进程按物理像素工作。

    必须在创建 QApplication 之前调用。否则在高 DPI（如 125%）下 Windows 会把窗口
    坐标按缩放比「虚拟化」返回：GetWindowRect 得到 1152x648 而不是真实的 1440x810，
    ClientToScreen / SetCursorPos 也都在虚拟坐标系里，点击就会偏。
    """
    try:
        # -4 = DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        return "PerMonitorV2"
    except Exception:
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return "PerMonitor"
    except Exception:
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
        return "System"
    except Exception:
        return "None"


DPI_MODE = enable_dpi_awareness()

from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtCore import QTime, QEventLoop
from skimage.metrics import structural_similarity as ssim

Pics = {}

# 规范画布：所有识别区域与点击坐标都以它为准，实际窗口多大都会先缩放/映射到这个尺寸
CANVAS_W, CANVAS_H = 1440, 810


def read_json():
    with open(DATA_JSON, 'r', encoding='utf-8') as f:
        content = f.read()
        data = json.loads(content)
    return data


def write_json(data):
    with open(DATA_JSON, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)


def compare_images(image1, image2):
    img2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    # 转换为灰度图
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 使用结构相似性指数（SSIM）比较相似度
    ssim_index, _ = ssim(gray1, gray2, full=True)
    return ssim_index


def ShowImg(image):
    plt.imshow(image)
    plt.show()


def DrawRectWithText(image, rect, text):
    img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    x, y, w, h = rect
    img2 = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    img2 = cv2.putText(img2, text, (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))


def CompareCard(card):
    order = {"3": 0, "4": 1, "5": 2, "6": 3, "7": 4, "8": 5, "9": 6, "T": 7, "J": 8, "Q": 9, "K": 10, "A": 11, "2": 12,
             "X": 13, "D": 14}
    return order[card]


def CompareCardInfo(card):
    order = {"3": 0, "4": 1, "5": 2, "6": 3, "7": 4, "8": 5, "9": 6, "T": 7, "J": 8, "Q": 9, "K": 10, "A": 11, "2": 12,
             "X": 13, "D": 14}
    return order[card[0]]


def CompareCards(cards1, cards2):
    if len(cards1) != len(cards2):
        return False
    cards1.sort(key=CompareCard)
    cards2.sort(key=CompareCard)
    for i in range(0, len(cards1)):
        if cards1[i] != cards2[i]:
            return False
    return True


def GetListDifference(l1, l2):
    temp1 = []
    temp1.extend(l1)
    temp2 = []
    temp2.extend(l2)
    for i in l2:
        if i in temp1:
            temp1.remove(i)
    for i in l1:
        if i in temp2:
            temp2.remove(i)
    return temp1, temp2


def FindImage(fromImage, template, threshold=0.9):
    w, h, _ = template.shape
    fromImage = cv2.cvtColor(np.asarray(fromImage), cv2.COLOR_RGB2BGR)
    res = cv2.matchTemplate(fromImage, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    points = []
    for pt in zip(*loc[::-1]):
        points.append(pt)
    return points


def LocateOnImage(image, template, region=None, confidence=0.8):
    """在 image 的 region 内找 template，返回规范画布坐标；找不到返回 None。"""
    ox, oy = 0, 0
    if region is not None:
        ox, oy, w, h = region
        image = image[oy:oy + h, ox:ox + w, :]
    if image.size == 0 or template is None:
        return None
    if template.shape[0] > image.shape[0] or template.shape[1] > image.shape[1]:
        return None
    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if max_val >= confidence:
        return ox + max_loc[0], oy + max_loc[1]
    return None


def LocateAllOnImage(image, template, region=None, confidence=0.8):
    if region is not None:
        x, y, w, h = region
        image = image[y:y + h, x:x + w]
    w, h = image.shape[1], image.shape[0]

    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= confidence)
    points = []
    for pt in zip(*loc[::-1]):
        points.append((pt[0], pt[1], w, h))
    return points


def play_sound(sound_file):
    pygame.mixer.init()
    pygame.mixer.music.load(sound_file)
    pygame.mixer.music.play()


class GameHelper:
    def __init__(self):
        self.ScreenZoomRate = None
        self.counter = QTime()
        self.Pics = {}
        self.PicsCV = {}
        st = time.time()
        self.Handle = win32gui.FindWindow("UnityWndClass", None)
        self.Interrupt = False
        self.RealRate = (CANVAS_W, CANVAS_H)
        # 最近一次截图时客户区的真实像素尺寸/屏幕位置；用于把规范画布坐标映射回真实点击点
        self.ClientSize = (CANVAS_W, CANVAS_H)
        self.ClientOrigin = (0, 0)
        # 一次性告警标记，避免每次截图都刷屏
        self._minimized_warned = False
        self._active_warned = False
        self._size_warned = False
        self.GetZoomRate()
        for file in os.listdir(PICS_DIR):
            info = file.split(".")
            if len(info) > 1 and info[1] == "png":
                pic_path = PICS_DIR / file
                tmpImage = Image.open(pic_path)
                imgCv = cv2.imread(str(pic_path))
                self.Pics.update({info[0]: tmpImage})
                self.PicsCV.update({info[0]: imgCv})

    def sleep(self, ms):
        self.counter.restart()
        while self.counter.elapsed() < ms:
            QtWidgets.QApplication.processEvents(QEventLoop.AllEvents, 50)

    # ------------------------------------------------------------------ 截图
    def _client_geometry(self, hwnd):
        """返回客户区在屏幕上的左上角与像素尺寸。"""
        left, top = win32gui.ClientToScreen(hwnd, (0, 0))
        l, t, r, b = win32gui.GetClientRect(hwnd)
        return int(left), int(top), int(r - l), int(b - t)

    @staticmethod
    def _is_blank(im):
        try:
            return float(np.asarray(im.convert("RGB")).std()) < 1.0
        except Exception:
            return True

    def _grab_client(self, hwnd, left, top, width, height):
        """抓客户区画面：先 PrintWindow(PW_CLIENTONLY[|PW_RENDERFULLCONTENT])，
        失败再退回屏幕 BitBlt。返回 PIL.Image 或 None。"""
        hwndDC = win32gui.GetWindowDC(hwnd)
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        im = None
        try:
            for flag in (3, 1):  # 3=仅客户区+完整渲染内容, 1=仅客户区
                saveDC = mfcDC.CreateCompatibleDC()
                bmp = win32ui.CreateBitmap()
                bmp.CreateCompatibleBitmap(mfcDC, width, height)
                saveDC.SelectObject(bmp)
                try:
                    res = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), flag)
                except Exception:
                    res = 0
                if res:
                    info = bmp.GetInfo()
                    im = Image.frombuffer("RGB", (info["bmWidth"], info["bmHeight"]),
                                          bmp.GetBitmapBits(True), "raw", "BGRX", 0, 1)
                saveDC.DeleteDC()
                win32gui.DeleteObject(bmp.GetHandle())
                if im is not None and not self._is_blank(im):
                    break
                im = None
        finally:
            mfcDC.DeleteDC()
            win32gui.ReleaseDC(hwnd, hwndDC)

        if im is None:
            try:
                screen = win32gui.GetDC(0)
                src = win32ui.CreateDCFromHandle(screen)
                dst = src.CreateCompatibleDC()
                bmp = win32ui.CreateBitmap()
                bmp.CreateCompatibleBitmap(src, width, height)
                dst.SelectObject(bmp)
                dst.BitBlt((0, 0), (width, height), src, (left, top), win32con.SRCCOPY)
                info = bmp.GetInfo()
                im = Image.frombuffer("RGB", (info["bmWidth"], info["bmHeight"]),
                                      bmp.GetBitmapBits(True), "raw", "BGRX", 0, 1)
                dst.DeleteDC()
                win32gui.DeleteObject(bmp.GetHandle())
                win32gui.ReleaseDC(0, screen)
                if self._is_blank(im):
                    im = None
            except Exception:
                im = None
        return im

    @staticmethod
    def normalize_frame(im):
        """把任意尺寸的客户区画面等比缩放到规范画布 1440x810。

        游戏 UI 是 16:9 等比布局，缩放后所有识别区域与点击坐标都能沿用同一套常量，
        因此窗口被拉大、缩小，或系统处于 125% 缩放，都不影响识别。
        """
        if im.size == (CANVAS_W, CANVAS_H):
            return im
        return im.resize((CANVAS_W, CANVAS_H), Image.LANCZOS)

    def Screenshot(self, region=None):  # -> (im, (left, top))
        try_count = 3
        while try_count > 0:
            try_count -= 1
            try:
                self.Handle = win32gui.FindWindow("UnityWndClass", None)
                hwnd = self.Handle
                if not hwnd:
                    print("未找到游戏窗口（UnityWndClass），请先以窗口模式启动欢乐斗地主")
                    self.sleep(200)
                    continue

                # 最小化时抓不到内容。提权环境下可自己恢复，失败再提示用户。
                if win32gui.IsIconic(hwnd):
                    try:
                        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                        self.sleep(300)
                    except Exception:
                        pass
                if win32gui.IsIconic(hwnd):
                    if not self._minimized_warned:
                        print("游戏窗口已最小化，无法识别：请从任务栏恢复窗口（允许被遮挡，但不能最小化）")
                        self._minimized_warned = True
                    self.sleep(200)
                    continue

                left, top, cw, ch = self._client_geometry(hwnd)
                if cw < 320 or ch < 180:
                    if not self._size_warned:
                        print("游戏客户区太小（%dx%d），请恢复窗口" % (cw, ch))
                        self._size_warned = True
                    self.sleep(200)
                    continue

                im = self._grab_client(hwnd, left, top, cw, ch)
                if im is None:
                    if not self._active_warned:
                        print("截图失败：窗口最小化、被遮挡，或权限不足（游戏以管理员运行时本程序也需提权）")
                        self._active_warned = True
                    self.sleep(200)
                    continue

                # 记录真实客户区尺寸，供点击坐标反向映射使用
                self.ClientSize = (cw, ch)
                self.ClientOrigin = (left, top)
                self.RealRate = (cw, ch)
                im = self.normalize_frame(im)
                if region is not None:
                    im = im.crop((region[0], region[1],
                                  region[0] + region[2], region[1] + region[3]))
                return im, (left, top)
            except Exception as e:
                print("截图时出现错误:", repr(e))
                self.sleep(200)
        return None, (0, 0)

    def GetZoomRate(self):
        try:
            self.ScreenZoomRate = ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100
        except Exception:
            self.ScreenZoomRate = 1.0

    # ------------------------------------------------------- 坐标映射与点击
    def to_client(self, pos):
        """规范画布坐标(1440x810) -> 客户区像素坐标。"""
        cw, ch = self.ClientSize
        return (int(round(pos[0] * cw / float(CANVAS_W))),
                int(round(pos[1] * ch / float(CANVAS_H))))

    def _click(self, pos, restore_cursor=True):
        hwnd = self.Handle or win32gui.FindWindow("UnityWndClass", None)
        if not hwnd:
            return
        cx, cy = self.to_client(pos)
        try:
            sx, sy = win32gui.ClientToScreen(hwnd, (cx, cy))
            win32api.SetCursorPos((sx, sy))
        except Exception:
            pass

        lparam = win32api.MAKELONG(cx, cy)
        win32gui.PostMessage(hwnd, win32con.WM_ACTIVATE, win32con.WA_ACTIVE, 0)
        # 先发 MOUSEMOVE：Unity 需要先更新悬停状态，否则部分控件收不到点击
        win32gui.SendMessage(hwnd, win32con.WM_MOUSEMOVE, 0, lparam)
        win32gui.SendMessage(hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lparam)
        win32gui.SendMessage(hwnd, win32con.WM_LBUTTONUP, 0, lparam)
        self.sleep(120)

        if restore_cursor:
            # 光标挪出手牌区，避免悬停浮层挡住下次识别
            try:
                bx, by = self.to_client((1000, 550))
                sx, sy = win32gui.ClientToScreen(hwnd, (bx, by))
                win32api.SetCursorPos((sx, sy))
            except Exception:
                pass

    def LocateOnScreen(self, templateName, region, confidence=0.8, img=None):
        if img is not None:
            image = img
        else:
            image, _ = self.Screenshot()
        if image is None:
            return None
        imgcv = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        return LocateOnImage(imgcv, self.PicsCV[templateName], region=region, confidence=confidence)

    def ClickOnImage(self, templateName, region=None, confidence=0.8, img=None):
        if img is not None:
            image = img
        else:
            image, _ = self.Screenshot()
        if image is None:
            return
        imgcv = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        result = LocateOnImage(imgcv, self.PicsCV[templateName], region=region, confidence=confidence)
        if result is not None:
            self.LeftClick(result)

    def LeftClick(self, pos):
        self._click(pos)

    def LeftClick2(self, pos):
        # 连点选牌时不挪回光标
        self._click(pos, restore_cursor=False)

    def MoveTo(self, pos):
        hwnd = self.Handle or win32gui.FindWindow("UnityWndClass", None)
        if not hwnd:
            return
        cx, cy = self.to_client(pos)
        try:
            sx, sy = win32gui.ClientToScreen(hwnd, (cx, cy))
            pyautogui.moveTo(sx, sy)
        except Exception:
            pass

    @staticmethod
    def MouseScroll(amount):
        # 发送鼠标滚轮事件
        win32api.mouse_event(win32con.MOUSEEVENTF_WHEEL, 0, 0, amount, 0)
