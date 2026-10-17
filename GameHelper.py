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

from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtCore import QTime, QEventLoop
from skimage.metrics import structural_similarity as ssim

from app_paths import DATA_JSON, PICS_DIR, resource

Pics = {}


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
    if region is not None:
        x, y, w, h = region
        imgShape = image.shape
        image = image[y:y + h, x:x + w, :]
    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    _, _, _, maxLoc = cv2.minMaxLoc(res)
    if (res >= confidence).any():
        return region[0] + maxLoc[0], region[1] + maxLoc[1]
    else:
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
        self.RealRate = (1440, 810)
        # 一次性告警标记，避免每次截图都刷屏
        self._minimized_warned = False
        self._active_warned = False
        self._move_warned = False
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

    def Screenshot(self, region=None):  # -> (im, (left, top))
        try_count = 3
        success = False
        while try_count > 0 and not success:
            try:
                try_count -= 1
                self.Handle = win32gui.FindWindow("UnityWndClass", None)
                hwnd = self.Handle
                if not hwnd:
                    print("未找到游戏窗口（UnityWndClass），请先以窗口模式启动欢乐斗地主")
                    self.sleep(200)
                    continue

                # 窗口已最小化时 PrintWindow 只能抓到标题栏大小的黑图，直接提示并重试
                if win32gui.IsIconic(hwnd):
                    if not self._minimized_warned:
                        print("游戏窗口已最小化，无法识别：请从任务栏恢复窗口（允许被遮挡，但不能最小化）")
                        self._minimized_warned = True
                    self.sleep(200)
                    continue

                # SetActiveWindow / MoveWindow 只是为了让窗口前置并固定为 1440x810。
                # 当游戏以管理员运行而本程序未提权时会被 UIPI 拒绝（error 5），
                # 但后台截图 PrintWindow 本身仍可用，所以失败不应中断整个截图流程。
                try:
                    win32gui.SetActiveWindow(hwnd)
                except Exception as e:
                    if not self._active_warned:
                        print("SetActiveWindow 被拒绝（不影响后台截图）:", repr(e))
                        self._active_warned = True

                left, top, right, bot = win32gui.GetWindowRect(hwnd)
                try:
                    win32gui.MoveWindow(hwnd, left, top, 1440, 810, True)
                except Exception as e:
                    if not self._move_warned:
                        print("MoveWindow 被拒绝（窗口若不是 1440x810，识别可能不准）:", repr(e))
                        self._move_warned = True
                width = 1440
                height = 810
                self.RealRate = (width, height)
                width = int(width)
                height = int(height)
                hwndDC = win32gui.GetWindowDC(hwnd)
                mfcDC = win32ui.CreateDCFromHandle(hwndDC)
                saveDC = mfcDC.CreateCompatibleDC()
                saveBitMap = win32ui.CreateBitmap()
                saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
                saveDC.SelectObject(saveBitMap)
                result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 3)
                bmpinfo = saveBitMap.GetInfo()
                bmpstr = saveBitMap.GetBitmapBits(True)
                im = Image.frombuffer(
                    "RGB",
                    (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
                    bmpstr, 'raw', 'BGRX', 0, 1)
                win32gui.DeleteObject(saveBitMap.GetHandle())
                saveDC.DeleteDC()
                mfcDC.DeleteDC()
                win32gui.ReleaseDC(hwnd, hwndDC)
                im = im.resize((1440, 810))
                if region is not None:
                    im = im.crop((region[0], region[1], region[0] + region[2], region[1] + region[3]))
                if result:
                    success = True
                    return im, (left, top)
            except Exception as e:
                print("截图时出现错误:", repr(e))
                self.sleep(200)
        return None, (0, 0)

    def GetZoomRate(self):
        self.ScreenZoomRate = ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100

    def LocateOnScreen(self, templateName, region, confidence=0.8, img=None):
        if img is not None:
            image = img
        else:
            image, _ = self.Screenshot()
        imgcv = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        return LocateOnImage(imgcv, self.PicsCV[templateName], region=region, confidence=confidence)

    def ClickOnImage(self, templateName, region=None, confidence=0.8, img=None):
        if img is not None:
            image = img
        else:
            image, _ = self.Screenshot()
        imgcv = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        result = LocateOnImage(imgcv, self.PicsCV[templateName], region=region, confidence=confidence)

        if result is not None:
            self.LeftClick(result)
            # print(result)

    def LeftClick(self, pos):
        x = int((pos[0] / 1440) * self.RealRate[0])
        y = int((pos[1] / 810) * self.RealRate[1])

        left, top, _, _ = win32gui.GetWindowRect(self.Handle)
        m, n = int(left + x), int(top + y)
        client_pos = (x, y)
        win32api.SetCursorPos((m, n))
        tmp = win32api.MAKELONG(client_pos[0], client_pos[1])

        win32gui.PostMessage(self.Handle, win32con.WM_ACTIVATE, win32con.WA_ACTIVE, 0)
        win32gui.SendMessage(self.Handle, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, tmp)
        win32gui.SendMessage(self.Handle, win32con.WM_LBUTTONUP, win32con.MK_LBUTTON, tmp)
        self.sleep(200)
        win32api.SetCursorPos((int(left + 1000), int(top + 550)))

    def LeftClick2(self, pos):
        x = int((pos[0] / 1440) * self.RealRate[0])
        y = int((pos[1] / 810) * self.RealRate[1])

        left, top, _, _ = win32gui.GetWindowRect(self.Handle)
        m, n = int(left + x), int(top + y)
        client_pos = (x, y)

        win32api.SetCursorPos((m, n))
        tmp = win32api.MAKELONG(client_pos[0], client_pos[1])
        win32gui.PostMessage(self.Handle, win32con.WM_ACTIVATE, win32con.WA_ACTIVE, 0)
        win32gui.SendMessage(self.Handle, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, tmp)
        win32gui.SendMessage(self.Handle, win32con.WM_LBUTTONUP, win32con.MK_LBUTTON, tmp)

    def MoveTo(self, pos):
        x = int((pos[0] / 1440) * self.RealRate[0])
        y = int((pos[1] / 810) * self.RealRate[1])
        left, top, _, _ = win32gui.GetWindowRect(self.Handle)
        x, y = int(left + x), int(top + y)
        pyautogui.moveTo(x, y)

    @staticmethod
    def MouseScroll(amount):
        # 发送鼠标滚轮事件
        win32api.mouse_event(win32con.MOUSEEVENTF_WHEEL, 0, 0, amount, 0)
