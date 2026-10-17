# -*- coding: utf-8 -*-
"""自动化编排：Worker 线程（状态机 / 识别 / 出牌）。

由原 mian.py 拆分而来：GUI 外壳见 hlddz/app.py，程序入口见 hlddz/app.py:main。
"""
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from skimage.metrics import structural_similarity as ssim

from douzero.env.game import AllEnvCard, EnvCard2RealCard, GameEnv, RealCard2EnvCard, bombs
from douzero.env.move_detector import get_move_type
from douzero.evaluation.deep_agent import DeepAgent

from .models import bid as BidModel, farmer as FarmerModel, landlord as LandlordModel
from .paths import BASELINES_DIR, MUSIC_DIR, ROOT
from .perception import color as DC
from .perception.screen import play_sound, read_json
from .runtime import helper


# 牌面整数 <-> 真实字符、整副牌、炸弹表统一来自 douzero.env.game（原本这里各复制了一份，
# 一旦某处改了顺序/映射就会与引擎不一致）。这里只保留界面记牌器需要的显示顺序。
AllCards = ['D', 'X', '2', 'A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3']


def cards_filter(location, distance):  # 牌检测结果滤波
    if len(location) == 0:
        return 0
    locList = [location[0][0]]
    poslist = [location[0]]
    count = 1
    for e in location:
        flag = 1
        for have in locList:
            if abs(e[0] - have) <= distance:
                flag = 0
                break
        if flag:
            count += 1
            locList.append(e[0])
            poslist.append(e)
    return count, poslist


def send_GG_and_MM():
    helper.ClickOnImage("chat", region=(1302, 744, 117, 56))
    time.sleep(0.5)
    helper.MouseScroll(-1000)
    time.sleep(0.5)
    helper.LeftClick((1110, 543))


def give_thumb():
    helper.ClickOnImage("chat", region=(1302, 744, 117, 56))
    time.sleep(0.5)
    helper.LeftClick((1375, 596))
    time.sleep(0.5)
    helper.LeftClick((1075, 293))


def give_coffee(side):
    if side == "right":
        print("Giving coffee to right")
        helper.MoveTo((1167, 293))
        time.sleep(1)
        helper.LeftClick((920, 237))
        time.sleep(0.1)
    elif side == "left":
        print("Giving coffee to left")
        helper.MoveTo((260, 300))
        time.sleep(1)
        helper.LeftClick((660, 237))
        time.sleep(0.1)
    else:
        print("咖啡side错误")


def compareImage(img1, img2):
    # 转换为灰度图
    gray1 = cv2.cvtColor(np.asarray(img1), cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(np.asarray(img2), cv2.COLOR_BGR2GRAY)

    # 使用结构相似性指数（SSIM）比较相似度
    ssim_index, _ = ssim(gray1, gray2, full=True)
    if ssim_index < 0.99:
        return True

    return False


def real_to_env(cards):
    env_card = [RealCard2EnvCard[c] for c in cards]
    env_card.sort()
    return env_card


def subtract_strings(str1, str2):
    result = ""
    for char1, char2 in zip(str1, str2):
        # 如果字符2在字符1中出现，则从字符1中删除该字符
        if char2 in str1:
            str1 = str1.replace(char2, '', 1)
    result = str1
    return result


class Worker(QThread):
    auto_game = pyqtSignal(int)  # Automatic start
    hand_game = pyqtSignal(int)  # Human start
    init_interface = pyqtSignal(int)  # Initialize interface
    player_display = pyqtSignal(int)  # Player Display
    state_display = pyqtSignal(str)  # 游戏状态显示
    stop = pyqtSignal(int)
    bid_display = pyqtSignal(str)  # 叫牌score
    pre_display = pyqtSignal(str)  # 局前score
    three_cards_display = pyqtSignal(str)  # Three cards display
    winrate_display = pyqtSignal(str)  # Winrate Display
    pre_cards_display = pyqtSignal(str)  # 显示AI推荐出牌
    left_cards_display = pyqtSignal(str)  # 显示上家的牌
    right_cards_display = pyqtSignal(str)  # 显示下家的牌
    recorder_display = pyqtSignal(str)  # 记牌器

    def __init__(self):
        super(Worker, self).__init__()
        self.MingpaiThreshold = None
        self.FarmerJiabeiThreshold = None
        self.JiabeiThreshold = None
        self.BidThresholds = None
        self.my_pass_sign = None
        self.my_played_cards_env = None
        self.my_played_cards_real = None
        self.auto_sign = None
        self.winrate = None
        self.in_game_flag = None
        self.initial_mingpai = None
        self.initial_multiply = None
        self.initial_cards = None
        self.RunGame = False
        self.initial_bid_rate = None
        self.initial_model_rate = None
        self.other_hands_cards_str = None
        self.stop_sign = False
        self.loop_sign = True
        self.env = None
        self.three_cards_env = None
        self.three_cards_real = None
        self.user_hand_cards_env = None
        self.user_hand_cards_real = None
        self.play_order = None
        self.card_play_data_list = None
        self.other_hand_cards = None
        self.other_played_cards_env = None
        self.other_played_cards_real = None
        self.user_position = None
        self.user_position_code = None
        self.distance = 30

        self.MyHandCardsPos = (180, 560, 1050, 90)  # 我的截图区域
        # 出牌区放高一些：长顺子/连对在一排放不下时游戏会排成两排，
        # 区域太矮会直接切掉第二排（表现为对面牌多时识别不对）
        self.LPlayedCardsPos = (320, 265, 400, 155)  # 左边出牌截图区域
        self.RPlayedCardsPos = (720, 265, 400, 155)  # 右边出牌截图区域
        self.MPlayedCardsPos = (300, 335, 800, 130)  # 我的出牌截图区域

        self.LandlordCardsPos = (600, 33, 220, 103)  # 地主底牌截图区域
        self.LPassPos = (360, 360, 120, 80)  # 左边不出截图区域
        self.RPassPos = (940, 360, 120, 80)  # 右边不出截图区域
        self.MPassPos = (636, 469, 152, 87)  # 我的不出截图区域

        self.PassBtnPos = (200, 450, 1000, 120)  # 要不起截图区域
        self.GeneralBtnPos = (200, 450, 1000, 120)  # 叫地主、抢地主、加倍按钮截图区域
        self.LandlordFlagPos = [(1247, 245, 48, 52), (12, 661, 51, 53), (123, 243, 52, 54)]  # 地主标志截图区域(右-我-左)
        self.blue_cards_num = [(273, 388, 33, 42), (1117, 387, 33, 44)]  # 加倍阶段上家和下家的牌数显示区域

        self.model_path_dict = {
            'landlord': str(BASELINES_DIR / "resnet" / "resnet_landlord.ckpt"),
            'landlord_up': str(BASELINES_DIR / "resnet" / "resnet_landlord_up.ckpt"),
            'landlord_down': str(BASELINES_DIR / "resnet" / "resnet_landlord_down.ckpt")
        }
        LandlordModel.init_model(self.model_path_dict['landlord'])

    def run(self):
        if self.auto_sign:
            print("现在是自动模式")
        else:
            print("现在是手动模式")
        self.loop_sign = True
        self.stop_sign = False
        try:
            self.run_threshold()
            while not self.stop_sign:
                try:
                    self.detect_start_btn()
                    self.init_interface.emit(1)
                    self.before_start()
                    self.init_cards()
                except Exception:
                    # 单局内的识别/解析异常不应该让整个 QThread（乃至进程）崩溃
                    traceback.print_exc()
                    print("本局处理异常，5 秒后重试")
                    self.RunGame = False
                    self.init_interface.emit(1)
                    time.sleep(5)
                time.sleep(2)
        except Exception:
            traceback.print_exc()
        finally:
            print("自动化线程结束")

    def run_threshold(self):
        # 阈值已由主线程在点击「手动/自动」时写盘（MyPyQT_Form.write_threshold），
        # 这里直接读文件即可。旧实现是 emit 信号 + sleep(0.1) 的跨线程握手，
        # 主线程恰好繁忙时 worker 会读到上一次的旧阈值。
        data = read_json()

        self.BidThresholds = [float(data['bid1']), float(data['bid2']), float(data['bid3'])]
        self.JiabeiThreshold = (
            (float(data['jiabei1']), float(data['jiabei2'])), (float(data['jiabei3']), float(data['jiabei4'])))
        self.FarmerJiabeiThreshold = (float(data['jiabei7']), float(data['jiabei8']))
        self.MingpaiThreshold = float(data['mingpai'])

    def detect_start_btn(self):
        beans = [(308, 204, 254, 60), (295, 474, 264, 60), (882, 203, 230, 60)]
        for i in beans:
            result = helper.LocateOnScreen("over", region=i, confidence=0.9)
            if result is not None:
                print("\n豆子出现，对局结束")
                self.RunGame = False
                try:
                    if self.env is not None:
                        self.env.game_over = True
                        self.env.reset()
                    self.init_interface.emit(1)
                except AttributeError as e:
                    traceback.print_exc()
                time.sleep(3)
                break

        if self.auto_sign:
            result = helper.LocateOnScreen("continue", region=(1100, 617, 200, 74))
            if result is not None:
                if not self.loop_sign:
                    print("游戏已结束")
                    self.state_display.emit("游戏已结束")
                    self.stop.emit(1)
                else:
                    self.RunGame = False
                    try:
                        if self.env is not None:
                            self.env.game_over = True
                            self.env.reset()
                        self.init_interface.emit(1)
                    except AttributeError as e:
                        traceback.print_exc()
                        time.sleep(1)
                    helper.ClickOnImage("continue", region=(1100, 617, 200, 74))

            win = helper.LocateOnScreen("win", region=(756, 191, 240, 107))
            lose = helper.LocateOnScreen("lose", region=(756, 191, 240, 107))
            if win is not None or lose is not None:
                if not self.loop_sign:
                    print("游戏已结束")
                    self.state_display.emit("游戏已结束")
                    self.stop.emit(1)
                else:
                    self.RunGame = False
                    try:
                        if self.env is not None:
                            self.env.game_over = True
                            self.env.reset()
                        self.init_interface.emit(1)
                    except AttributeError as e:
                        traceback.print_exc()
                        time.sleep(1)
                    time.sleep(1)

            result = helper.LocateOnScreen("quick_start", region=(1107, 679, 252, 118))
            if result is not None:
                helper.ClickOnImage("quick_start", region=(1107, 679, 252, 118))
                time.sleep(1)

            result = helper.LocateOnScreen("reward", region=(587, 570, 224, 86))
            if result is not None:
                helper.ClickOnImage("reward", region=(587, 570, 224, 86))
                time.sleep(1)

            result = helper.LocateOnScreen("start_game", region=(720, 466, 261, 117))
            if result is not None:
                helper.ClickOnImage("start_game", region=(720, 466, 261, 117))
                time.sleep(1)

            result = helper.LocateOnScreen("tuoguan", region=(577, 613, 271, 128))
            if result is not None:
                helper.ClickOnImage("tuoguan", region=(577, 613, 271, 128))
                time.sleep(1)

            result = helper.LocateOnScreen("sure", region=(630, 456, 289, 143))
            if result is not None:
                helper.ClickOnImage("sure", region=(630, 456, 289, 143))
                time.sleep(1)

            result = helper.LocateOnScreen("tuijian", region=(729, 465, 219, 100))
            if result is not None:
                helper.ClickOnImage("tuijian", region=(729, 465, 219, 100))
                time.sleep(1)

            result = helper.LocateOnScreen("good", region=(434, 599, 219, 88))
            if result is not None:
                helper.ClickOnImage("good", region=(434, 599, 219, 88))
                time.sleep(1)

            result = helper.LocateOnScreen("zhidao", region=(593, 543, 224, 94))
            if result is not None:
                helper.ClickOnImage("zhidao", region=(593, 543, 224, 94))
                time.sleep(1)

            result = helper.LocateOnScreen("chacha", region=(1036, 65, 300, 230))
            if result is not None:
                #helper.ClickOnImage("chacha", region=(1036, 65, 300, 230))
                #有些情况下弹窗关闭失败，这种情况下使用LeftClick2可以关闭弹窗
                helper.LeftClick2((result[0] + 20, result[1] + 20))
                time.sleep(1)
        else:
            pass

    def _hand_card_count(self):
        try:
            return len(self.find_my_cards())
        except Exception:
            return 0

    def _in_game_signal(self):
        """是否已经在游戏场内。

        旧实现只认聊天图标（chat）：新版客户端在发牌/对局过程中聊天入口可能被隐藏，
        实测全图最高只有 0.46，导致明明已经开局却永远停在「未进入游戏」。
        这里改用更可靠的信号：手牌数量 / 底牌数量 / 聊天入口 / 快速开始。
        """
        if self._hand_card_count() in (17, 20):
            return True
        if len(self.find_three_cards()) == 3:
            return True
        if helper.LocateOnScreen("chat", region=(1302, 744, 117, 56), confidence=0.7) is not None:
            return True
        if helper.LocateOnScreen("quick_start", region=self.GeneralBtnPos, confidence=0.7) is not None:
            return True
        return False

    def _game_started_signal(self):
        """是否已经发牌、可以对局。"""
        if self._hand_card_count() in (17, 20):
            return True
        if len(self.find_three_cards()) == 3:
            return True
        return helper.LocateOnScreen("laotou", region=self.LandlordCardsPos, confidence=0.7) is not None

    def before_start(self):
        self.in_game_flag = False
        print("未进入游戏", end='')
        while not self._in_game_signal():
            time.sleep(0.5)
            print(".", end="")
            self.state_display.emit("未进入游戏")
            self.detect_start_btn()
        self.in_game_flag = True
        time.sleep(0.3)
        if self.in_game_flag:
            print("\n在游戏场内")
            print("游戏未开始", end="")
            while not self._game_started_signal():
                time.sleep(0.5)
                print(".", end="")
                self.state_display.emit("游戏未开始")
                self.detect_start_btn()

            print("\n开始游戏")
            self.state_display.emit("开始游戏")
        try:
            self.RunGame = True
            self.choose_multiples_stage()
        except Exception as e:
            # 打印堆栈，否则“加倍阶段有问题”完全没法定位
            print("加倍阶段有问题: %s" % e)
            traceback.print_exc()

    def choose_multiples_stage(self):
        # 旧代码这里还声明了 never-assigned 的 initialBeishu，一旦被读取就是 NameError
        global win_rate, cards_str
        self.initial_multiply = 0
        self.initial_mingpai = 0
        self.initial_bid_rate = 0
        have_bid = False
        is_stolen = 0
        cards = self.find_three_cards()
        while len(cards) == 0:
            if not self.RunGame:
                break
            if self.auto_sign:
                self.state_display.emit("自动模式")
                self.winrate_display.emit("     自动模式：   叫地主  抢地主  加倍")
                cards = self.find_my_cards()
                while len(cards) < 5:
                    if not self.RunGame or not self.auto_sign:
                        break
                    time.sleep(0.01)
                    cards = self.find_my_cards()
                    self.detect_start_btn()
                print("前6张牌：", cards)

                if "DX222" in cards:
                    ming_btn = helper.LocateOnScreen("ming_btn", region=self.GeneralBtnPos)
                    if ming_btn is not None:
                        helper.ClickOnImage("ming_btn", region=self.GeneralBtnPos)
                        print("明牌")

                bid_wait_start = time.time()
                last_phase_check = bid_wait_start
                bid_diag_done = False
                # 每个阶段只处理一次：按钮没消失时会反复检出同一阶段，
                # 旧逻辑会无限重复点「抢地主」（实测刷屏几十次）。
                bid_acted = False
                double_acted = False
                while self.RunGame and self.auto_sign:
                    outterBreak = False
                    jiaodizhu_btn = helper.LocateOnScreen("jiaodizhu_btn", region=self.GeneralBtnPos)
                    qiangdizhu_btn = helper.LocateOnScreen("qiangdizhu_btn", region=self.GeneralBtnPos)
                    jiabei_btn = helper.LocateOnScreen("jiabei_btn", region=self.GeneralBtnPos)
                    print("自动加倍或叫地主", end="")
                    while jiaodizhu_btn is None and qiangdizhu_btn is None and jiabei_btn is None:
                        if not self.RunGame or not self.auto_sign:
                            break
                        print(".", end="")
                        time.sleep(0.1)
                        jiaodizhu_btn = helper.LocateOnScreen("jiaodizhu_btn", region=self.GeneralBtnPos)
                        qiangdizhu_btn = helper.LocateOnScreen("qiangdizhu_btn", region=self.GeneralBtnPos)
                        jiabei_btn = helper.LocateOnScreen("jiabei_btn", region=self.GeneralBtnPos)
                        self.detect_start_btn()
                        # 模板全不相符（本机按钮外观与 pics/ 旧图不一致）时，
                        # 用“按钮排布 + 底牌是否亮出”判断当前阶段，再走原有分支：
                        #   底牌已亮 3 张 → 加倍阶段（未处理过才处理）
                        #   还没叫过       → 叫地主阶段
                        #   叫过还未抢     → 抢地主阶段
                        #   都处理过了     → 等按钮消失，超时则跳过
                        if jiaodizhu_btn is None and qiangdizhu_btn is None and jiabei_btn is None:
                            boxes = self.find_action_buttons()
                            if boxes:
                                if len(self.find_three_cards()) == 3:
                                    if not double_acted:
                                        jiabei_btn = boxes[0]
                                        print("\n（按按钮位置判断：加倍阶段）")
                                elif not bid_acted:
                                    if have_bid:
                                        qiangdizhu_btn = boxes[0]
                                        print("\n（按按钮位置判断：抢地主阶段）")
                                    else:
                                        jiaodizhu_btn = boxes[0]
                                        print("\n（按按钮位置判断：叫地主阶段）")
                                if jiaodizhu_btn is not None or qiangdizhu_btn is not None or jiabei_btn is not None:
                                    break
                                # 该阶段已经点过，按钮却还在：先等它消失，等不到就跳过
                                if time.time() - bid_wait_start > 8:
                                    print("\n叫牌/加倍已处理完但按钮仍在，跳过该阶段")
                                    outterBreak = True
                                    break
                        # 等这么久还没有按钮：打印按钮区域的实际匹配分并存图，便于定位
                        if not bid_diag_done and time.time() - bid_wait_start > 10:
                            bid_diag_done = True
                            self._diag_buttons()
                        # 中途接管时叫牌/加倍阶段可能已经过去：按钮永远不会出现，
                        # 若此时底牌已经亮出（3 张），直接进入出牌阶段，避免死等。
                        if time.time() - bid_wait_start > 20 and time.time() - last_phase_check > 2:
                            last_phase_check = time.time()
                            if len(self.find_three_cards()) == 3:
                                print("\n叫牌/加倍阶段已过（底牌已亮出、无按钮），直接进入出牌")
                                outterBreak = True
                                break
                    if outterBreak:
                        break
                    if jiabei_btn is None:
                        cards = self.find_my_cards()
                        while len(cards) != 17 and len(cards) != 20:
                            if not self.RunGame or not self.auto_sign:
                                break
                            time.sleep(0.2)
                            cards = self.find_my_cards()
                            self.detect_start_btn()
                        cards_str = "".join([card[0] for card in cards])
                        win_rate = BidModel.predict_score(cards_str)
                        farmer_score = FarmerModel.predict(cards_str, "farmer")
                        self.bid_display.emit("叫牌得分: " + str(round(win_rate, 3)))
                        self.pre_display.emit("不叫得分: " + str(round(farmer_score, 3)))
                        self.initial_bid_rate = round(win_rate, 3)
                        is_stolen = 0

                        if jiaodizhu_btn is not None:
                            have_bid = True
                            bid_acted = True
                            if win_rate > self.BidThresholds[0]:
                                self.click_action("jiaodizhu_btn", True, self.GeneralBtnPos)
                                print("\n叫地主")

                            else:
                                self.click_action("bujiao_btn", False, self.GeneralBtnPos)
                                print("\n不叫地主")
                            self._wait_buttons_gone()
                        elif qiangdizhu_btn is not None:
                            is_stolen = 1
                            bid_acted = True
                            if have_bid:
                                threshold_index = 1
                            else:
                                threshold_index = 2
                            if win_rate > self.BidThresholds[threshold_index]:
                                self.click_action("qiangdizhu_btn", True, self.GeneralBtnPos)
                                print("\n抢地主")
                            else:
                                self.click_action("buqiang_btn", False, self.GeneralBtnPos)
                                print("\n不抢地主")
                            have_bid = True
                            self._wait_buttons_gone()

                    else:
                        self.state_display.emit("加倍阶段")

                        self.three_cards_real = self.find_three_cards()
                        print("正在识别底牌", end="")
                        three_deadline = time.time() + 15
                        while len(self.three_cards_real) != 3:
                            print(".", end="")
                            if not self.RunGame or not self.auto_sign:
                                break
                            if time.time() > three_deadline:
                                print("\n底牌识别超时，退出本局")
                                self.RunGame = False
                                return
                            time.sleep(0.2)
                            self.three_cards_real = self.find_three_cards()
                            self.detect_start_btn()
                        print("\n底牌:", self.three_cards_real)
                        cards = self.find_my_cards()
                        hand_deadline = time.time() + 15
                        while len(cards) != 17 and len(cards) != 20:
                            if not self.RunGame or not self.auto_sign:
                                break
                            if time.time() > hand_deadline:
                                print("\n手牌识别超时，退出本局")
                                self.RunGame = False
                                return
                            time.sleep(0.2)
                            cards = self.find_my_cards()
                            self.detect_start_btn()
                        cards_str = "".join([card[0] for card in cards])
                        self.initial_cards = cards_str
                        if len(cards_str) == 20:
                            print("手牌：", cards_str)
                            self.user_position_code = 1
                            win_rate = LandlordModel.predict_by_model(cards_str, self.three_cards_real)
                            self.pre_display.emit("局前得分: " + str(round(win_rate, 3)))
                            print("预估地主得分:", round(win_rate, 3))
                        else:
                            win_rate = FarmerModel.predict(cards_str, "up")
                            print("预估农民得分:", round(win_rate, 3))
                            self.pre_display.emit("局前得分: " + str(round(win_rate, 3)))
                        if len(cards_str) == 20:
                            JiabeiThreshold = self.JiabeiThreshold[is_stolen]
                        else:
                            # 农民只有一套加倍阈值（jiabei7/jiabei8）。
                            # 旧代码先赋 jiabei5/6 再用 jiabei7/8 覆盖，前者永不生效，已删除。
                            JiabeiThreshold = self.FarmerJiabeiThreshold

                        if len(cards_str) == 17:
                            time.sleep(3)
                            print("休息 3 秒, 观察他人加倍情况")
                        else:
                            time.sleep(2)
                            print("休息 2 秒, 观察他人加倍情况")

                        if win_rate > JiabeiThreshold[0]:
                            chaojijiabei_btn = helper.LocateOnScreen("chaojijiabei_btn", region=self.GeneralBtnPos)
                            if (chaojijiabei_btn is not None and
                                    ('DX' in cards_str or 'D2' in cards_str or 'X22' in cards_str or
                                     '222A' in cards_str or '22AAA' in cards_str)):
                                helper.ClickOnImage("chaojijiabei_btn", region=self.GeneralBtnPos)
                                print("超级加倍")
                                self.initial_multiply = 4
                            else:
                                self.click_action("jiabei_btn", True, self.GeneralBtnPos, confidence=0.9)
                                print("加倍")
                                self.initial_multiply = 2
                        elif win_rate > JiabeiThreshold[1]:
                            print("加倍")
                            self.click_action("jiabei_btn", True, self.GeneralBtnPos, confidence=0.9)
                            self.initial_multiply = 2
                        else:
                            print("不加倍")
                            self.click_action("bujiabei_btn", False, self.GeneralBtnPos)
                            self.initial_multiply = 0
                        double_acted = True
                        break
                    self.detect_start_btn()
                    if outterBreak:
                        break
                    time.sleep(1)

                if win_rate > self.MingpaiThreshold and len(cards_str) == 20:
                    try:
                        print("Going to Mingpai, Good Luck!")
                        time.sleep(5)
                        helper.ClickOnImage("mingpai_btn", region=self.GeneralBtnPos)
                        print("明牌")
                        self.initial_mingpai = 1
                    except Exception as e:
                        print(e)

                self.winrate = win_rate
                print("自动叫地主结束")
                self.state_display.emit("加倍结束")

            else:
                self.state_display.emit("手动模式")
                self.winrate_display.emit("     手动模式：   叫地主  抢地主  加倍")
                print("手动加倍或叫地主")
                while self.RunGame and not self.auto_sign:
                    jiaodizhu_btn = helper.LocateOnScreen("jiaodizhu_btn", region=self.GeneralBtnPos)
                    qiangdizhu_btn = helper.LocateOnScreen("qiangdizhu_btn", region=self.GeneralBtnPos)
                    jiabei_btn = helper.LocateOnScreen("jiabei_btn", region=self.GeneralBtnPos)

                    while jiaodizhu_btn is None and qiangdizhu_btn is None and jiabei_btn is None:
                        if not self.RunGame or self.auto_sign:
                            break
                        time.sleep(0.1)
                        jiaodizhu_btn = helper.LocateOnScreen("jiaodizhu_btn", region=self.GeneralBtnPos)
                        qiangdizhu_btn = helper.LocateOnScreen("qiangdizhu_btn", region=self.GeneralBtnPos)
                        jiabei_btn = helper.LocateOnScreen("jiabei_btn", region=self.GeneralBtnPos)
                        self.detect_start_btn()
                    if jiabei_btn is None:
                        cards = self.find_my_cards()
                        while len(cards) != 17 and len(cards) != 20:
                            if not self.RunGame or self.auto_sign:
                                break
                            time.sleep(0.2)
                            cards = self.find_my_cards()
                            self.detect_start_btn()
                        cards_str = "".join([card[0] for card in cards])
                        win_rate = BidModel.predict_score(cards_str)
                        farmer_score = FarmerModel.predict(cards_str, "farmer")
                        self.bid_display.emit("叫牌得分: " + str(round(win_rate, 3)))
                        self.pre_display.emit("不叫得分: " + str(round(farmer_score, 3)))

                        if jiaodizhu_btn is not None:
                            print("\n 正在叫地主")
                            self.state_display.emit("正在叫地主")
                            time.sleep(2)

                        elif qiangdizhu_btn is not None:
                            print("\n正在抢地主")
                            self.state_display.emit("正在抢地主")
                            time.sleep(2)

                    else:
                        print("\n正在加倍")
                        self.state_display.emit("正在加倍")
                        self.three_cards_real = self.find_three_cards()
                        print("正在识别底牌", end="")
                        three_deadline = time.time() + 15
                        while len(self.three_cards_real) != 3:
                            print(".", end="")
                            if not self.RunGame or not self.auto_sign:
                                break
                            if time.time() > three_deadline:
                                print("\n底牌识别超时，退出本局")
                                self.RunGame = False
                                return
                            time.sleep(0.2)
                            self.three_cards_real = self.find_three_cards()
                            self.detect_start_btn()
                        print("\n底牌:", self.three_cards_real)
                        cards = self.find_my_cards()
                        hand_deadline = time.time() + 15
                        while len(cards) != 17 and len(cards) != 20:
                            if not self.RunGame or not self.auto_sign:
                                break
                            if time.time() > hand_deadline:
                                print("\n手牌识别超时，退出本局")
                                self.RunGame = False
                                return
                            time.sleep(0.2)
                            cards = self.find_my_cards()
                            self.detect_start_btn()
                        cards_str = "".join([card[0] for card in cards])
                        self.initial_cards = cards_str
                        if len(cards_str) == 20:
                            print("手牌：", cards_str)
                            self.user_position_code = 1
                            win_rate = LandlordModel.predict_by_model(cards_str, self.three_cards_real)
                            self.pre_display.emit("局前得分: " + str(round(win_rate, 3)))
                            print("预估地主得分:", round(win_rate, 3))
                        else:
                            win_rate = FarmerModel.predict(cards_str, "up")
                            print("预估农民得分:", round(win_rate, 3))
                            self.pre_display.emit("局前得分: " + str(round(win_rate, 3)))
                            time.sleep(3)
                        break
                    self.detect_start_btn()

                self.winrate = win_rate
                print("手动叫地主结束")
                self.state_display.emit("加倍结束")
                cards = self.find_three_cards()
                self.detect_start_btn()

    def init_cards(self):
        print("进入出牌前的阶段")
        # 玩家角色代码：0-地主上家, 1-地主, 2-地主下家
        self.RunGame = True
        self.user_position_code = None
        self.initial_model_rate = 0
        self.user_hand_cards_env = []
        self.my_played_cards_env = []
        self.other_played_cards_env = []
        # 其他玩家手牌（整副牌减去玩家手牌，后续再减掉历史出牌）
        self.other_hand_cards = []
        self.three_cards_env = []
        # 其余手牌在被执行牌前先清空，避免沿用上一局的残留
        self.three_cards_real = ""
        self.user_hand_cards_real = ""
        # 开局时三个玩家的手牌
        self.card_play_data_list = {}

        # 识别三张底牌
        print("正在识别底牌", end="")
        three_tries = 0
        three_deadline = time.time() + 20
        while len(self.three_cards_real) != 3:
            print(".", end="")
            if not self.RunGame:
                break
            if time.time() > three_deadline:
                print("\n底牌识别超时（20 秒），退出本局")
                self.RunGame = False
                return
            time.sleep(0.2)
            self.three_cards_real = self.find_three_cards()
            three_tries += 1
            if three_tries % 10 == 0:
                print("\n底牌识别异常（识别到 %d 张: %r）"
                      % (len(self.three_cards_real), self.three_cards_real))
                self.save_debug_shot("three_cards_%d" % len(self.three_cards_real))
            self.detect_start_btn()
        print("\n底牌： ", self.three_cards_real)
        self.three_cards_display.emit("底牌：" + self.three_cards_real)
        self.three_cards_env = [RealCard2EnvCard[c] for c in list(self.three_cards_real)]

        # 识别玩家的角色
        time.sleep(0.5)
        print("正在识别玩家角色", end="")
        role_tries = 0
        while self.user_position_code is None:
            if not self.RunGame:
                break
            print(".", end="")
            self.user_position_code = self.find_landlord()
            role_tries += 1
            if role_tries % 10 == 0 and self.user_position_code is None:
                hand = self.find_my_cards()
                print("\n玩家角色识别中：未找到地主标志，手牌 %d 张（%r）"
                      % (len(hand), hand))
                self.save_debug_shot("role_%d" % len(hand))
                # 手牌已经空了 ⇒ 本局早就结束（地主标志也随界面消失），别再空转
                if len(hand) == 0:
                    print("手牌为空（本局已结束），退出本局等待下一局")
                    self.RunGame = False
                    return
            # 硬超时：30 秒还认不出角色就放弃本局，避免界面卡死
            if role_tries > 150:
                print("\n识别角色超时，退出本局")
                self.RunGame = False
                return
            time.sleep(0.2)
            self.detect_start_btn()

        if self.user_position_code is None:
            # 停止/跳出循环时可能仍为 None，直接索引会 TypeError
            print("\n未能识别玩家角色，跳过本局")
            self.RunGame = False
            return

        self.user_position = ['landlord_up', 'landlord', 'landlord_down'][self.user_position_code]
        print("我的角色：", self.user_position)
        self.player_display.emit(self.user_position_code)

        # 识别玩家手牌
        self.user_hand_cards_real = self.find_my_cards()
        self.recorder_preview()
        hand_tries = 0
        expect_hand = 20 if self.user_position_code == 1 else 17
        hand_deadline = time.time() + 10
        while len(self.user_hand_cards_real) != expect_hand:
            if not self.RunGame:
                break
            time.sleep(0.2)
            self.user_hand_cards_real = self.find_my_cards()
            hand_tries += 1
            got = len(self.user_hand_cards_real)
            if time.time() > hand_deadline:
                # 10 秒都凑不齐开局张数 ⇒ 本局已经在打了（中途接管），等下一局。
                # 用时间判定而不是“张数不变”，因为识别张数会在 14/15 之间抖动。
                print("\n当前对局已进行中（手牌 %d 张，期望 %d），无法中途接管；等待本局结束"
                      % (got, expect_hand))
                self.state_display.emit("等待下一局")
                while self.RunGame and self.find_my_cards():
                    time.sleep(2)
                    self.detect_start_btn()
                self.RunGame = False
                return
            if hand_tries % 10 == 0:
                print("\n手牌识别异常（期望 %d 张，实际 %d 张: %s）"
                      % (expect_hand, got, self.user_hand_cards_real))
                self.save_debug_shot("my_cards_%d" % got)
            self.detect_start_btn()
        self.user_hand_cards_env = [RealCard2EnvCard[c] for c in list(self.user_hand_cards_real)]

        # 整副牌减去玩家手上的牌，就是其他人的手牌,再分配给另外两个角色（如何分配对AI判断没有影响）
        for i in set(AllEnvCard):
            self.other_hand_cards.extend([i] * (AllEnvCard.count(i) - self.user_hand_cards_env.count(i)))
        self.emit_recorder()
        self.card_play_data_list.update({
            'three_landlord_cards': self.three_cards_env,
            ['landlord_up', 'landlord', 'landlord_down'][(self.user_position_code + 0) % 3]:
                self.user_hand_cards_env,
            ['landlord_up', 'landlord', 'landlord_down'][(self.user_position_code + 1) % 3]:
                self.other_hand_cards[0:17] if (self.user_position_code + 1) % 3 != 1 else self.other_hand_cards[17:],
            ['landlord_up', 'landlord', 'landlord_down'][(self.user_position_code + 2) % 3]:
                self.other_hand_cards[0:17] if (self.user_position_code + 1) % 3 == 1 else self.other_hand_cards[17:]
        })
        print("开始对局")
        print("手牌:", self.user_hand_cards_real)
        print("底牌:", self.three_cards_real)

        # 出牌顺序：0-玩家出牌, 1-玩家下家出牌, 2-玩家上家出牌
        self.play_order = 0 if self.user_position == "landlord" else 1 if self.user_position == "landlord_up" else 2

        # 创建一个代表玩家的AI
        AI = [0, 0]
        AI[0] = self.user_position
        AI[1] = DeepAgent(self.user_position, self.model_path_dict[self.user_position])
        self.env = GameEnv(AI)

        try:
            self.game_start()
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            traceback.print_tb(exc_tb)
            self.RunGame = False
            if self.env is not None:
                self.env.game_over = True
                self.env.reset()
            self.init_interface.emit(1)

    def game_start(self):
        self.my_pass_sign = False
        self.env.card_play_init(self.card_play_data_list)
        print("开始对局")
        self.state_display.emit("游戏开始")
        first_run = True

        while not self.env.game_over:
            if not self.RunGame:
                break
            while self.play_order == 0:
                if not self.RunGame:
                    break
                cards = self.find_played_cards(self.MPlayedCardsPos)
                if len(cards) > 0:
                    time.sleep(0.5)
                    cards = self.find_played_cards(self.MPlayedCardsPos)
                if len(cards) == 0:

                    action_message, action_list = self.env.step(self.user_position, update=False)
                    score = float(action_message['win_rate'])
                    if "resnet" in self.model_path_dict[self.user_position]:
                        score *= 8
                    hand_cards_str = ''.join(
                        [EnvCard2RealCard[c] for c in self.env.info_sets[self.user_position].player_hand_cards])
                    action_list = action_list[:3]
                    if not self.winrate:
                        self.winrate = 0
                    self.bid_display.emit("当前得分：" + str(round(float(action_list[0][1]), 3)))
                    self.pre_cards_display.emit(action_message["action"] if action_message["action"] else "pass")
                    action_list_str = " | ".join([ainfo[0] + " = " + ainfo[1] for ainfo in action_list])
                    print(action_list_str)
                    self.winrate_display.emit(action_list_str)

                    if self.auto_sign:
                        if first_run:
                            self.initial_model_rate = round(float(action_message["win_rate"]), 3)  # win_rate at start
                        if action_message["action"] == "":
                            self.env.step(self.user_position, [])
                            hand_cards_str = ''.join(
                                [EnvCard2RealCard[c] for c in self.env.info_sets[self.user_position].player_hand_cards])
                            print("出牌:", "Pass", "| 得分:", round(action_message["win_rate"], 4), "| 剩余手牌:",
                                  hand_cards_str)

                            pass_btn = helper.LocateOnScreen("pass_btn", region=self.PassBtnPos, confidence=0.7)
                            yaobuqi = helper.LocateOnScreen("yaobuqi", region=self.GeneralBtnPos, confidence=0.7)
                            buchu = helper.LocateOnScreen("buchu", region=self.MPassPos)
                            pass_wait_start = time.time()
                            pass_diag_done = False
                            while pass_btn is None and yaobuqi is None and buchu is None:
                                if not self.RunGame or not self.auto_sign:
                                    break
                                pass_btn = helper.LocateOnScreen("pass_btn", region=self.PassBtnPos, confidence=0.7)
                                yaobuqi = helper.LocateOnScreen("yaobuqi", region=self.GeneralBtnPos, confidence=0.7)
                                buchu = helper.LocateOnScreen("buchu", region=self.MPassPos)
                                self.detect_start_btn()
                                # 模板一直认不出按钮时，改用“按钮排布”点「不出」
                                # （从左到右：出牌 / 不出 / 提示，只有一个按钮时它就是要不起）
                                if time.time() - pass_wait_start > 3:
                                    boxes = self.find_action_buttons()
                                    if boxes:
                                        self.click_button_by_layout(1 if len(boxes) >= 2 else 0)
                                        print("\n按按钮位置点击「不出」（共 %d 个按钮）" % len(boxes))
                                        break
                                if not pass_diag_done and time.time() - pass_wait_start > 10:
                                    pass_diag_done = True
                                    print("\n[提示] 找不到「不出/要不起」按钮：本机按钮外观与 pics/ 模板不符，"
                                          "请用 `python tools/grab_template.py pass_btn` 重新采集模板")
                                    self._diag_buttons()
                                    self.save_debug_shot("pass_wait")

                            if not self.my_pass_sign:
                                if pass_btn is not None:
                                    self.click_action("pass_btn", False, self.PassBtnPos)
                                if yaobuqi is not None:
                                    self.click_action("yaobuqi", False, self.GeneralBtnPos)
                                if buchu is not None:
                                    print("你们太牛X！ 我要不起")
                                    time.sleep(0.5)
                                time.sleep(0.2)
                            else:
                                self.my_pass_sign = False
                            time.sleep(0.1)
                        else:
                            # 先选牌，再点「出牌」。click_action 会先试模板，认不出就按
                            # 按钮排布点最左边那个（出牌/叫地主），避免死等模板。
                            before_hand = len(self.find_my_cards())
                            selected = self.click_cards(action_message["action"])
                            if not selected:
                                # 游戏“智能选牌”可能多选/打乱，选不准则重选一次
                                print("   选中的牌与目标不一致，重选一次")
                                time.sleep(0.3)
                                self.resync_hand()
                                selected = self.click_cards(action_message["action"])
                            if not selected:
                                # 宁可不出，也不要把错误的牌打出去
                                self._select_fails = getattr(self, "_select_fails", 0) + 1
                                print("   ⚠ 选牌仍不一致（连续 %d 次），本次不点出牌"
                                      % self._select_fails)
                                self.save_debug_shot("select_mismatch")
                                if self._select_fails >= 3:
                                    print("   连续 3 次选牌失败，本局停止自动出牌")
                                    self._select_fails = 0
                                    self.RunGame = False
                                    return
                                time.sleep(1.0)
                                self.detect_start_btn()
                                continue
                            self._select_fails = 0
                            if first_run:
                                time.sleep(0.5)
                            play_wait_start = time.time()
                            while True:
                                if not self.RunGame or not self.auto_sign:
                                    break
                                if self.click_action("play_card", True, self.PassBtnPos):
                                    print("点击出牌按钮")
                                    break
                                if time.time() - play_wait_start > 15:
                                    print("\n[提示] 15 秒内没点到「出牌」按钮，已存图排查")
                                    self._diag_buttons()
                                    self.save_debug_shot("play_wait")
                                    break
                                time.sleep(0.2)
                                self.detect_start_btn()
                            # 校验：出牌成功后手牌必须减少，否则说明选牌/点击没生效，重试
                            for attempt in range(3):
                                time.sleep(0.8)
                                now_hand = len(self.find_my_cards())
                                if now_hand < before_hand:
                                    break
                                print("\n出牌未生效（手牌仍 %d 张），重试选牌并点出牌" % now_hand)
                                if attempt == 2:
                                    self.save_debug_shot("play_stuck")
                                self.click_cards(action_message["action"])
                                time.sleep(0.4)
                                self.click_action("play_card", True, self.PassBtnPos)

                            self.my_played_cards_env = [RealCard2EnvCard[c] for c in list(action_message["action"])]
                            self.my_played_cards_env.sort()
                            self.env.step(self.user_position, self.my_played_cards_env)

                            hand_cards_str = ''.join(
                                [EnvCard2RealCard[c] for c in self.env.info_sets[self.user_position].player_hand_cards])
                            print("出牌:", action_message["action"] if action_message["action"] else "Pass", "| 得分:",
                                  round(action_message["win_rate"], 4), "| 剩余手牌:", hand_cards_str)

                            if action_message["action"] == "DX":
                                time.sleep(0.1)
                            ani = self.animation(action_message["action"])
                            if ani:
                                time.sleep(0.5)

                            if len(hand_cards_str) == 0:
                                self.RunGame = False
                                try:
                                    if self.env is not None:
                                        self.env.game_over = True
                                        self.env.reset()
                                    self.init_interface.emit(1)

                                except AttributeError as e:
                                    traceback.print_exc()
                                    print("程序走到这里")
                                    time.sleep(1)
                                break
                        hand_cards_str = ''.join(
                            [EnvCard2RealCard[c] for c in self.env.info_sets[self.user_position].player_hand_cards])
                        # print("炸弹数量：", self.env.info_sets[self.user_position].player_hand_cards)
                        if len(hand_cards_str) == 4 and self.env.info_sets[
                            self.user_position].player_hand_cards in bombs:
                            if self.user_position == "landlord_up":
                                give_coffee("left")
                            elif self.user_position == "landlord_down":
                                give_coffee("right")
                            time.sleep(1)
                        if len(hand_cards_str) == 2 and self.user_position != "landlord":
                            if hand_cards_str[0] == hand_cards_str[1]:
                                send_GG_and_MM()
                                time.sleep(0.5)
                            else:
                                give_thumb()
                                time.sleep(0.5)

                    if not self.auto_sign:
                        print("现在是手动模式，请手动出牌")
                        play_sound(str(MUSIC_DIR / "1.wav"))
                        if action_message["action"] == "":
                            self.pre_cards_display.emit("推荐：pass")
                        else:
                            self.pre_cards_display.emit(f"推荐：" + action_message["action"])
                        pass_flag = helper.LocateOnScreen('buchu', region=self.MPassPos)
                        centralCards = self.find_played_cards(self.MPlayedCardsPos)
                        print("等待自己出牌", end="")
                        while len(centralCards) == 0 and pass_flag is None:
                            if not self.RunGame or self.auto_sign or self.env.game_over:
                                break
                            print(".", end="")
                            time.sleep(0.2)
                            have_ani = self.waitUntilNoAnimation()
                            if have_ani:
                                self.pre_cards_display.emit("等待动画")
                                time.sleep(0.2)
                            pass_flag = helper.LocateOnScreen('buchu', region=self.MPassPos)
                            centralCards = self.find_played_cards(self.MPlayedCardsPos)
                            self.detect_start_btn()

                        if pass_flag is None:
                            while True:
                                if not self.RunGame or self.auto_sign:
                                    break
                                centralOne = self.find_played_cards(self.MPlayedCardsPos)
                                time.sleep(0.3)
                                centralTwo = self.find_played_cards(self.MPlayedCardsPos)
                                if centralOne == centralTwo and len(centralOne) > 0:
                                    self.my_played_cards_real = centralOne
                                    if ("X" in centralOne or "D" in centralOne) and not ("DX" in centralOne):
                                        time.sleep(0.5)
                                        self.my_played_cards_real = self.find_played_cards(self.MPlayedCardsPos)
                                        if len(self.my_played_cards_real) == 0:
                                            self.my_played_cards_real = centralOne
                                    break
                                self.detect_start_btn()
                        else:
                            self.my_played_cards_real = ""
                        print("\n自己出牌：", self.my_played_cards_real if self.my_played_cards_real else "pass")
                        self.my_played_cards_env = [RealCard2EnvCard[c] for c in list(self.my_played_cards_real)]
                        self.my_played_cards_env.sort()
                        action_message, _ = self.env.step(self.user_position, self.my_played_cards_env)
                        hand_cards_str = ''.join(
                            [EnvCard2RealCard[c] for c in self.env.info_sets[self.user_position].player_hand_cards])

                        print("出牌:", action_message["action"] if action_message["action"] else "Pass", "| 得分:",
                              round(action_message["win_rate"], 4), "| 剩余手牌:", hand_cards_str)

                        self.pre_cards_display.emit(self.my_played_cards_real if self.my_played_cards_real else "pass")
                    time.sleep(0.5)
                    self.play_order = 1
                else:
                    print("已经出过牌")
                    time.sleep(0.5)
                    self.play_order = 1
                first_run = False
                self.detect_start_btn()
            self.detect_start_btn()

            if self.play_order == 1:
                self.right_cards_display.emit("等待下家出牌")

                pass_flag = helper.LocateOnScreen('buchu', region=self.RPassPos)
                rightCards = self.find_other_cards(self.RPlayedCardsPos)
                print("等待下家出牌", end="")

                while len(rightCards) == 0 and pass_flag is None:
                    if not self.RunGame:
                        break
                    print(".", end="")
                    time.sleep(0.2)
                    have_ani = self.waitUntilNoAnimation()
                    if have_ani:
                        self.right_cards_display.emit("等待动画")
                        time.sleep(0.2)
                    pass_flag = helper.LocateOnScreen('buchu', region=self.RPassPos)
                    rightCards = self.find_other_cards(self.RPlayedCardsPos)
                    # 本机“不出”标记模板匹配不上：改用“是否已轮到我”反推下家不出
                    if pass_flag is None and len(rightCards) == 0 and self._my_turn_by_buttons():
                        print("\n下家不出（已轮到我出牌）")
                        pass_flag = "layout"
                    self.detect_start_btn()

                if pass_flag is None:
                    while True:
                        if not self.RunGame:
                            break
                        rightOne = self.find_other_cards(self.RPlayedCardsPos)
                        time.sleep(0.3)
                        rightTwo = self.find_other_cards(self.RPlayedCardsPos)

                        if rightOne == rightTwo and len(rightOne) > 0:
                            self.other_played_cards_real = rightOne
                            if "X" in rightOne or "D" in rightOne and not ("DX" in rightOne):
                                time.sleep(0.5)
                                self.other_played_cards_real = self.find_other_cards(self.RPlayedCardsPos)
                                if len(self.other_played_cards_real) == 0:
                                    self.other_played_cards_real = rightOne
                            break
                        else:
                            pass_flag = helper.LocateOnScreen('buchu', region=self.RPassPos)
                            if pass_flag is not None:
                                self.other_played_cards_real = ""
                                break
                        self.detect_start_btn()
                else:
                    self.other_played_cards_real = ""
                print("\n下家出牌：", self.other_played_cards_real if self.other_played_cards_real else "pass")
                self.other_played_cards_env = [RealCard2EnvCard[c] for c in list(self.other_played_cards_real)]
                self.other_played_cards_env.sort()
                self.env.step(self.user_position, self.other_played_cards_env)

                self.right_cards_display.emit(self.other_played_cards_real if self.other_played_cards_real else "pass")
                self.other_hands_cards_str = subtract_strings(self.other_hands_cards_str, self.other_played_cards_real)
                self.recorder_display.emit(self.other_hands_cards_str)
                time.sleep(0.2)
                self.play_order = 2

            if self.play_order == 2:
                self.left_cards_display.emit("等待上家出牌")

                pass_flag = helper.LocateOnScreen('buchu', region=self.LPassPos)
                leftCards = self.find_other_cards(self.LPlayedCardsPos)
                print("等待上家出牌", end="")
                while len(leftCards) == 0 and pass_flag is None:
                    if not self.RunGame:
                        break
                    print(".", end="")
                    time.sleep(0.2)
                    have_ani = self.waitUntilNoAnimation()
                    if have_ani:
                        self.left_cards_display.emit("等待动画")
                        time.sleep(0.2)
                    pass_flag = helper.LocateOnScreen('buchu', region=self.LPassPos)
                    leftCards = self.find_other_cards(self.LPlayedCardsPos)
                    # 本机“不出”标记模板匹配不上：改用“是否已轮到我”反推上家不出
                    if pass_flag is None and len(leftCards) == 0 and self._my_turn_by_buttons():
                        print("\n上家不出（已轮到我出牌）")
                        pass_flag = "layout"
                    self.detect_start_btn()

                if pass_flag is None:
                    while True:
                        if not self.RunGame:
                            break
                        leftOne = self.find_other_cards(self.LPlayedCardsPos)
                        time.sleep(0.3)
                        leftTwo = self.find_other_cards(self.LPlayedCardsPos)
                        if leftOne == leftTwo and len(leftOne) > 0:
                            self.other_played_cards_real = leftOne
                            if ("X" in leftOne or "D" in leftOne) and not ("DX" in leftOne):
                                time.sleep(0.5)
                                self.other_played_cards_real = self.find_other_cards(self.LPlayedCardsPos)
                                if len(self.other_played_cards_real) == 0:
                                    self.other_played_cards_real = leftOne
                            break
                        else:
                            pass_flag = helper.LocateOnScreen('buchu', region=self.LPassPos)
                            if pass_flag is not None:
                                self.other_played_cards_real = ""
                                break
                        self.detect_start_btn()
                else:
                    self.other_played_cards_real = ""
                print("\n上家出牌：", self.other_played_cards_real if self.other_played_cards_real else "pass")
                self.other_played_cards_env = [RealCard2EnvCard[c] for c in list(self.other_played_cards_real)]
                self.other_played_cards_env.sort()
                self.env.step(self.user_position, self.other_played_cards_env)

                self.left_cards_display.emit(self.other_played_cards_real if self.other_played_cards_real else "pass")
                self.other_hands_cards_str = subtract_strings(self.other_hands_cards_str,
                                                              self.other_played_cards_real)
                self.recorder_display.emit(self.other_hands_cards_str)
                time.sleep(0.05)
                self.play_order = 0

        if not self.loop_sign:
            self.stop.emit(1)
            print("游戏结束")
        self.state_display.emit("游戏结束")
        self.init_interface.emit(1)
        print("本局已结束")
        print()
        time.sleep(5)

    # 各区域牌面模板的缩放范围。底牌比手牌明显小（实测约 0.5 倍），
    # 其它区域给一点余量以兼容窗口缩放导致的尺寸差异。
    CARD_SCALES = {
        "m": (0.90, 1.00, 1.10),
        "o": (0.80, 0.90, 1.00, 1.10),
        "c": (0.80, 0.90, 1.00, 1.10),
        "z": (0.45, 0.50, 0.55, 0.60, 0.65, 0.70),
    }

    # 底牌区域改用「缩小的手牌模板」匹配：pics/z*.png 是旧版客户端的尺寸，
    # 新版底牌只有手牌模板的约 0.5 倍，套旧模板最高只有 0.65 分。
    CARD_TEMPLATE_MARK = {"z": "m"}

    def find_cards_detail(self, img, pos, mark="", confidence=0.8, gap=12):
        """识别区域内所有牌面，返回 [(牌面字符, x, y, w, h), ...]，按从左到右排序。

        x/y 是**整幅规范画布**坐标（已加上区域偏移），可直接用于点击定位。
        相比旧实现：多尺度匹配、跨牌型按 x 聚类（一张牌只取最高分牌型）、
        大小王用被匹配到的那张牌自身的裁剪块判红黑。
        """
        if img is None:
            return []
        x0, y0, w0, h0 = pos
        crop = img[y0:y0 + h0, x0:x0 + w0]
        if crop.size == 0:
            return []

        scales = self.CARD_SCALES.get(mark, (1.0,))
        tpl_mark = self.CARD_TEMPLATE_MARK.get(mark, mark)
        hits = []
        for card in AllCards:
            tpl0 = helper.PicsCV.get(tpl_mark + card)
            if tpl0 is None:
                continue
            for s in scales:
                tpl = cv2.resize(tpl0, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                if tpl.shape[0] >= crop.shape[0] or tpl.shape[1] >= crop.shape[1]:
                    continue
                res = cv2.matchTemplate(crop, tpl, cv2.TM_CCOEFF_NORMED)
                ys, xs = np.where(res >= confidence)
                for yy, xx in zip(ys, xs):
                    hits.append((int(xx), int(yy), float(res[yy, xx]), card,
                                 int(tpl.shape[1]), int(tpl.shape[0])))
        if not hits:
            return []

        # 先按 y 分行（长顺子/连对在一行放不下时游戏会排成两排），
        # 再在每一行内按 x 聚类。只按 x 聚类会把上下两排同位置的牌并成一张。
        hits.sort(key=lambda h: (h[1], h[0]))
        rows = []
        for h in hits:
            if rows and abs(h[1] - rows[-1][-1][1]) <= max(10, int(h[5] * 0.5)):
                rows[-1].append(h)
            else:
                rows.append([h])

        clusters = []
        for row in rows:
            row.sort(key=lambda h: h[0])
            row_clusters = []
            last_x = None
            for h in row:
                # 与「上一个命中点」比较而不是与聚类代表比较，
                # 避免命中点沿 x 链式漂移把相邻两张牌并成一张
                if row_clusters and last_x is not None and h[0] - last_x < gap:
                    if h[2] > row_clusters[-1][2]:
                        row_clusters[-1] = h
                else:
                    row_clusters.append(h)
                last_x = h[0]
            clusters.extend(row_clusters)

        detail = []
        for xx, yy, score, card, tw, th in clusters:
            ch = self._king_label(crop, xx, yy, tw, th) if card in ("X", "D") else card[0]
            detail.append((ch, int(xx) + x0, int(yy) + y0, tw, th, float(score)))
        return detail

    @staticmethod
    def _king_label(crop, x, y, w, h):
        """用大小王模板命中处的裁剪块判红黑：红=大王(D)，黑=小王(X)。"""
        try:
            patch = crop[max(0, y):y + h, max(0, x):x + w]
            if patch.size:
                for name, weight in DC.ColorClassify(debug=False).classify(patch):
                    if name == "Red":
                        return "D" if weight > 0.7 else "X"
        except Exception:
            pass
        return "X"

    def find_cards(self, img, pos, mark="", confidence=0.8, gap=12):
        """识别区域内所有牌面，返回**从左到右**排列的牌面字符串。

        顺序与位置一致（都按 x），click_cards 用 zip(cards, pos_list) 建映射不会错位。
        """
        return "".join(d[0] for d in self.find_cards_detail(img, pos, mark, confidence, gap))

    def find_my_cards(self):
        img, _ = helper.Screenshot()
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        my_cards_real = self.find_cards(img, self.MyHandCardsPos, mark="m")
        return my_cards_real

    def find_other_cards(self, pos):
        img, _ = helper.Screenshot()
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        other_cards_real = self.find_cards(img, pos, mark="o")
        return other_cards_real

    def find_played_cards(self, pos):
        img, _ = helper.Screenshot()
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        other_cards_real = self.find_cards(img, pos, mark="c")
        return other_cards_real

    def _diag_buttons(self):
        """等待叫牌/加倍按钮超时时打印现场信息（区域统计 + 各按钮模板分 + 存图）。"""
        try:
            img, _ = helper.Screenshot()
            if img is None:
                print("\n[诊断] 截图失败")
                return
            bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            x0, y0, w0, h0 = self.GeneralBtnPos
            crop = bgr[y0:y0 + h0, x0:x0 + w0]
            print("\n[诊断] 按钮区域 %s mean=%.1f std=%.1f"
                  % (self.GeneralBtnPos, crop.mean(), crop.std()))
            for name in ("jiaodizhu_btn", "bujiao_btn", "qiangdizhu_btn",
                         "buqiang_btn", "jiabei_btn", "bujiabei_btn"):
                tpl = helper.PicsCV.get(name)
                if tpl is None or tpl.shape[0] >= crop.shape[0] or tpl.shape[1] >= crop.shape[1]:
                    continue
                res = cv2.matchTemplate(crop, tpl, cv2.TM_CCOEFF_NORMED)
                print("   %-16s %.3f @ %s" % (name, float(res.max()), cv2.minMaxLoc(res)[3]))
            self.save_debug_shot("bid_wait")
        except Exception as e:
            print("\n[诊断] 失败: %s" % e)

    def find_action_buttons(self):
        """在当前按钮区找出一排按钮，返回按 x 排序的 [(x, y, w, h), ...]（画布坐标）。

        不依赖模板外观：本机客户端（单机模式）的按钮与 pics/ 里的旧图对不上
        （匹配分只有 0.25~0.59），但按钮本身是“一排亮色圆角矩形”，
        用高饱和高亮连通域 + 尺寸/长宽比过滤即可稳定定位。
        实测布局（从左到右）：出牌/叫地主、不出/不叫、提示。
        """
        try:
            img, _ = helper.Screenshot()
            if img is None:
                return []
            bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            x0, y0, w0, h0 = self.PassBtnPos
            crop = bgr[y0:y0 + h0, x0:x0 + w0]
            if crop.size == 0:
                return []
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array((0, 120, 150)), np.array((180, 255, 255)))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            boxes = []
            for i in range(1, n):
                x, y, w, h, area = stats[i]
                # 按钮实测尺寸：约 137x60 ~ 205x65，长宽比 2~3.2。
                # 这里收紧过滤，避免把“剩余牌数/当前这一手”等亮色元素也当成按钮
                if area < 2000 or h < 45 or h > 90 or w < 90 or w < 0 or w > 260:
                    continue
                if not (1.5 <= w / float(h) <= 4.0):
                    continue
                boxes.append((int(x) + x0, int(y) + y0, int(w), int(h)))
            if len(boxes) < 2:
                return boxes
            # 先按 y 中心聚成“一行”，取元素最多的那一行，再按 x 排序；
            # 这样即使别处有亮色块，也不会把真正的按钮挤掉。
            boxes.sort(key=lambda b: b[1] + b[3] / 2.0)
            rows = []
            for b in boxes:
                cy = b[1] + b[3] / 2.0
                if rows and abs(cy - rows[-1][0]) <= 12:
                    rows[-1][1].append(b)
                    rows[-1][0] = sum(x[1] + x[3] / 2.0 for x in rows[-1][1]) / len(rows[-1][1])
                else:
                    rows.append([cy, [b]])
            best_row = max(rows, key=lambda r: len(r[1]))[1]
            best_row.sort()
            return best_row
        except Exception:
            return []

    def click_button_by_layout(self, index, template=None):
        """按按钮排布点击：0=出牌/叫地主，1=不出/不叫，2=提示。

        若给了 template，则在检出的按钮里挑**与该模板相关性最高**的那个
        （本机模板绝对分只有 0.3~0.5，但排序往往仍然有效），
        相关性都太低时退回按位置的 index。
        """
        boxes = self.find_action_buttons()
        if not boxes:
            return False
        print("   检出按钮: %s" % [(b[0], b[1], b[2], b[3]) for b in boxes])
        target = None
        if template:
            try:
                img, _ = helper.Screenshot()
                bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
                tpl = helper.PicsCV.get(template)
                best = (-9.0, None)
                if tpl is not None:
                    for b in boxes:
                        x, y, w, h = b
                        patch = bgr[y:y + h, x:x + w]
                        if patch.size == 0 or patch.shape[0] < tpl.shape[0] or patch.shape[1] < tpl.shape[1]:
                            continue
                        res = cv2.matchTemplate(patch, tpl, cv2.TM_CCOEFF_NORMED)
                        score = float(res.max())
                        if score > best[0]:
                            best = (score, b)
                if best[1] is not None and best[0] >= 0.30:
                    target = best[1]
                    print("（按钮相关性 %.3f → 选 x=%d）" % (best[0], target[0]))
            except Exception:
                target = None
        if target is None:
            idx = min(max(index, 0), len(boxes) - 1)
            target = boxes[idx]
        x, y, w, h = target
        print("   点击按钮: x=%d y=%d %dx%d → 点 (%d,%d)"
              % (x, y, w, h, x + w // 2, y + h // 2))
        helper.LeftClick2((x + w // 2, y + h // 2))
        return True

    def click_action(self, template, positive=True, region=None, confidence=0.7):
        """点某个动作按钮：先按模板匹配，模板认不出时按“按钮排布”兜底。

        positive=True 取左边那个（出牌/叫地主/抢地主/加倍），False 取中间那个
        （不出/不叫/不抢/不加倍）。
        """
        region = region or self.PassBtnPos
        try:
            if helper.LocateOnScreen(template, region=region, confidence=confidence) is not None:
                helper.ClickOnImage(template, region=region, confidence=confidence)
                return True
        except Exception:
            pass
        if self.click_button_by_layout(0 if positive else 1, template=template):
            print("  （模板 %s 未匹配，按按钮位置点击）" % template)
            return True
        return False

    def _wait_buttons_gone(self, timeout=4.0):
        """点击动作按钮后等它消失，避免同一阶段被反复检出、反复点击。"""
        start = time.time()
        while time.time() - start < timeout:
            if not self.find_action_buttons():
                return True
            time.sleep(0.2)
        print("（按钮仍然存在，可能这次点击没生效）")
        return False

    def _my_turn_by_buttons(self, min_interval=1.0):
        """用“出牌/不出/提示 按钮是否出现”判断是否已经轮到我（不依赖模板）。

        对手“不出”的标记模板（buchu）在本机匹配不上，于是改用反推：
        对手区没出牌、而我的动作按钮已经出现 ⇒ 对手一定是不出。
        截图+HSV 处理有开销，按 min_interval 节流。
        """
        now = time.time()
        if now - getattr(self, "_last_btn_check", 0.0) < min_interval:
            return getattr(self, "_last_btn_state", False)
        self._last_btn_check = now
        self._last_btn_state = bool(self.find_action_buttons())
        return self._last_btn_state

    def find_three_cards(self):
        """识别三张底牌。

        底牌比手牌小很多（约 0.5 倍），用缩小的手牌模板多尺度匹配
        （见 CARD_TEMPLATE_MARK）。另外逐级降低阈值重试：大小王牌面窄，
        缩小后匹配分天然偏低（实测约 0.63），固定 0.72 会漏成 2 张；
        若某轮命中超过 3 张，按分数取最高的 3 张再把顺序还原成从左到右。
        """
        img, _ = helper.Screenshot()
        if img is None:
            return ""
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

        detail = []
        for conf in (0.72, 0.64, 0.58):
            detail = self.find_cards_detail(img, self.LandlordCardsPos, mark="z", confidence=conf)
            if len(detail) == 3:
                break
            if len(detail) > 3:
                detail = sorted(detail, key=lambda d: d[5], reverse=True)[:3]
                detail.sort(key=lambda d: d[1])
                break
        return "".join(d[0] for d in detail)

    def save_debug_shot(self, tag):
        """识别异常时保存现场截图到 logs/，便于事后排查（最多保留 30 张）。"""
        try:
            img, _ = helper.Screenshot()
            if img is None:
                return
            logs_dir = Path(ROOT) / "logs"
            logs_dir.mkdir(exist_ok=True)
            filename = "debug_%s_%s.png" % (tag, time.strftime("%Y%m%d_%H%M%S"))
            path = logs_dir / filename
            img.save(path)
            print("已保存诊断截图:", path)
            for old in sorted(logs_dir.glob("debug_*.png"))[:-30]:
                try:
                    old.unlink()
                except Exception:
                    pass
        except Exception as e:
            print("保存诊断截图失败:", repr(e))

    def emit_recorder(self):
        """刷新界面记牌器（整副牌 − 我的手牌 = 其余玩家的牌）。

        之前只在 init_cards 里算一次，若开局流程卡在叫牌阶段就一直显示 0。
        现在识别出手牌后立即刷新一次，并把各牌数量打进日志便于核对。
        """
        try:
            self.other_hands_cards_str = str(
                ''.join([EnvCard2RealCard[c] for c in self.other_hand_cards]))[::-1]
            print("记牌器: %s" % " ".join(
                "%s=%d" % (c, self.other_hands_cards_str.count(c)) for c in AllCards))
            self.recorder_display.emit(self.other_hands_cards_str)
        except Exception as e:
            print("记牌器刷新失败: %s" % e)

    def recorder_preview(self):
        """按当前已识别的手牌先刷新一次记牌器（不修改 other_hand_cards）。"""
        try:
            if len(self.user_hand_cards_real) not in (17, 20):
                return
            mine = [RealCard2EnvCard[c] for c in self.user_hand_cards_real]
            others = []
            for c in set(AllEnvCard):
                others.extend([c] * (AllEnvCard.count(c) - mine.count(c)))
            s = str(''.join([EnvCard2RealCard[c] for c in others]))[::-1]
            print("记牌器(预览): %s" % " ".join("%s=%d" % (c, s.count(c)) for c in AllCards))
            self.recorder_display.emit(s)
        except Exception as e:
            print("记牌器预览失败: %s" % e)

    def _hand_detail(self):
        """当前手牌的识别明细（每张牌含位置与分数）。"""
        try:
            img, _ = helper.Screenshot()
            if img is None:
                return []
            bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            return self.find_cards_detail(bgr, self.MyHandCardsPos, mark="m")
        except Exception:
            return []

    def _raised_entries(self, detail=None):
        """返回“已选中（被抬起）”的牌。

        选中后手牌会向上抬约 15~25px，所以按 y 分成两簇即可判断。
        只有明确是「少数牌被抬起」时才返回，判断不可靠一律返回空，
        避免把未选中的牌照当成已选中去点（那会把牌选错）。
        """
        detail = detail if detail is not None else self._hand_detail()
        if len(detail) < 2:
            return []
        base = max(d[2] for d in detail)
        raised = [d for d in detail if d[2] < base - 8]
        if not raised or len(raised) >= len(detail) - len(raised):
            return []
        return raised

    def resync_hand(self):
        """把手牌状态按当前识别结果重设，缓解识别偏差导致的状态漂移（出牌失败时用）。"""
        try:
            real = self.find_my_cards()
            if not real:
                return False
            self.user_hand_cards_real = real
            self.user_hand_cards_env = [RealCard2EnvCard[c] for c in real]
            self.env.info_sets[self.user_position].player_hand_cards = \
                list(self.user_hand_cards_env)
            print("   （手牌状态已按当前识别结果重设：%s）" % real)
            return True
        except Exception as e:
            print("   重设手牌失败: %s" % e)
            return False

    def _is_raised(self, entry, detail=None):
        """这张牌当前是否已经处于选中（抬起）状态。"""
        detail = detail if detail is not None else self._hand_detail()
        if not detail:
            return False
        x = entry[0]
        best = min(detail, key=lambda d: abs(d[1] - x))
        if abs(best[1] - x) > 14:
            return False
        base = max(d[2] for d in detail)
        return best[2] < base - 8

    def _fix_selection(self, out_cards, cards_dict, point_fn, rounds=2):
        """核对选中的牌与目标组合是否一致：漏的补点、多选的取消。

        游戏有“智能选牌”（点一张可能把同点数的一起选中），
        只靠“点一下就完事”很容易选错，所以每轮选完都核对整个组合。
        判断不出选中状态时不做任何纠正，避免越点越乱。
        """
        from collections import Counter

        target = Counter(out_cards)
        for _ in range(rounds):
            raised = self._raised_entries(self._hand_detail())
            if not raised:
                return True
            actual = Counter(d[0] for d in raised)
            if actual == target:
                return True
            extra = actual - target
            missing = target - actual
            print("   选牌核对：已选中 %s，目标 %s%s%s"
                  % ("".join(sorted(actual.elements())), out_cards,
                     ("；多选 " + "".join(sorted(extra.elements()))) if extra else "",
                     ("；漏选 " + "".join(sorted(missing.elements()))) if missing else ""))
            for value, cnt in extra.items():
                for d in [d for d in raised if d[0] == value][:cnt]:
                    helper.LeftClick2((d[1] + d[3] // 2, d[2] + d[4] // 2))
                    time.sleep(0.12)
            for value, cnt in missing.items():
                for e in (cards_dict.get(value) or [])[-cnt:]:
                    helper.LeftClick2(point_fn(e))
                    time.sleep(0.12)
            time.sleep(0.15)

        actual = Counter(d[0] for d in self._raised_entries(self._hand_detail()))
        ok = actual == target
        print("   选牌最终：%s（目标 %s）%s"
              % ("".join(sorted(actual.elements())), out_cards, "" if ok else " ← 不一致"))
        return ok

    def click_cards(self, out_cards):
        try:
            cards = self.find_my_cards()
            num = len(cards)
            space = 45.6
            res1 = helper.LocateOnScreen("up_left", region=self.MyHandCardsPos, confidence=0.65)
            while res1 is None:
                if not self.RunGame:
                    break
                print("未找到手牌区域")
                time.sleep(0.5)
                res1 = helper.LocateOnScreen("up_left", region=self.MyHandCardsPos, confidence=0.65)
                self.detect_start_btn()
            pos = res1[0] + 6, res1[1] + 7

            res2 = helper.LocateOnScreen("up_left", region=(180, 580, 1050, 90), confidence=0.65)
            if res2 is not None:
                pos = res1[0] + 6, res1[1] + 7

            # 每张牌的可点击点：用识别到的牌框中心。
            # 旧实现用固定 "牌的 y + 100"，而本机手牌高度只有 ~40px，
            # y+100 会落到牌下面，点击等于没点（表现为“出牌失败”）。
            img0, _ = helper.Screenshot()
            detail = []
            if img0 is not None:
                img0 = cv2.cvtColor(np.asarray(img0), cv2.COLOR_RGB2BGR)
                detail = self.find_cards_detail(img0, self.MyHandCardsPos, mark="m")
            if len(detail) == num:
                pos_list = [(d[1], d[2], d[3], d[4]) for d in detail]
            else:
                cy = self.MyHandCardsPos[1] + self.MyHandCardsPos[3] // 2
                pos_list = [(int(pos[0] + i * space), cy - 20, 40, 40) for i in range(num)]

            def _point(entry):
                return (entry[0] + entry[2] // 2, entry[1] + entry[3] // 2)

            # 将两个列表合并转为字典
            cards_dict = defaultdict(list)
            for key, value in zip(cards, pos_list):
                cards_dict[key].append(value)
            # 转换为普通的字典
            cards_dict = dict(cards_dict)
            remove_dict = {key: [] for key in cards_dict.keys()}

            # 先把上一次残留的“已选中”牌取消掉（选中后手牌会抬起），
            # 避免叠加选择导致“选不到对应的牌”
            for d in self._raised_entries(self._hand_detail()):
                helper.LeftClick2((d[1] + d[3] // 2, d[2] + d[4] // 2))
                time.sleep(0.08)

            # print(cards_dict)
            selected_ok = True
            if out_cards == "DX" and "X" in cards_dict:
                helper.LeftClick2(_point(cards_dict["X"][0]))
                time.sleep(0.5)

            else:
                for i in out_cards:
                    if i not in cards_dict or not cards_dict[i]:
                        # AI 要出的牌在当前识别到的手牌里没有（识别偏差导致的状态漂移）。
                        # 直接跳过，不要让 KeyError 把整次出牌打断。
                        print("   手牌里没有 %s（识别偏差），跳过这一张" % i)
                        continue
                    cars_pos = cards_dict[i][-1]
                    point = _point(cars_pos)
                    # 关键：游戏有“智能选牌”，点一张 7 可能把另一张 7 也一并选中。
                    # 所以点之前先看这张牌是不是已经抬起（已选中），已选中就不再点，
                    # 否则会把它点掉、后面就越选越乱。
                    if self._is_raised(cars_pos):
                        print("   %s 已被游戏自动选中，跳过点击" % i)
                    else:
                        helper.LeftClick2(point)
                        time.sleep(0.12)
                    remove_dict[i].append(cards_dict[i][-1])
                    cards_dict[i].remove(cards_dict[i][-1])

                # 选完牌后核对整个组合：既补漏，也取消多选的
                selected_ok = self._fix_selection(out_cards, cards_dict, _point)

                img, _ = helper.Screenshot()
                img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
                check_cards = self.find_cards(img, self.MyHandCardsPos, mark="m")
                for i in out_cards:
                    cards = cards.replace(i, "", 1)
                # print("检查剩的牌： ", check_cards, "应该剩的牌： ", cards)
                if len(check_cards) < len(cards):
                    for m in check_cards:
                        cards = cards.replace(m, "", 1)
                    # print("系统多点的牌： ", cards)
                    for n in cards:
                        # print("字典里还剩的牌： ", cards_dict)
                        cars_pos2 = cards_dict[n][-1]
                        # print("准备点回来的牌：", cars_pos2)
                        point2 = _point(cars_pos2)
                        helper.LeftClick2(point2)
                        time.sleep(0.1)
                        remove_dict[n].append(cards_dict[n][-1])
                        cards_dict[n].remove(cards_dict[n][-1])
                elif len(check_cards) > len(cards):
                    for m in cards:
                        check_cards = check_cards.replace(m, "", 1)
                    # print("系统少点的牌： ", check_cards)
                    for n in check_cards:
                        # print("删除的字典： ", remove_dict)
                        cars_pos3 = remove_dict[n][0]
                        # print("准备再点出去的牌：", cars_pos3)
                        point3 = _point(cars_pos3)
                        helper.LeftClick2(point3)
                        time.sleep(0.3)
                        remove_dict[n].remove(remove_dict[n][0])
                        # print(remove_dict)
        except Exception as e:
            print("检测到出牌错误")
            traceback.print_exc()
            selected_ok = False
        time.sleep(0.2)
        return selected_ok

    def find_landlord(self):
        """判断自己在牌桌上的身份：0=landlord_up，1=landlord，2=landlord_down。

        优先用「地主」标志（LandlordFlagPos 依次是 右 / 我 / 左 三个座位），
        比按手牌张数推断可靠得多——开局时还没人出牌，旧逻辑会一直返回 None
        卡在「正在识别玩家角色」。
        """
        for code, region in enumerate(self.LandlordFlagPos):
            try:
                if helper.LocateOnScreen("landlord", region=region, confidence=0.8) is not None:
                    return code
            except Exception:
                pass

        # 退化：按手牌数量 + 出牌区推断
        my_cards = self.find_my_cards()
        if len(my_cards) == 20:
            return 1
        if len(my_cards) == 17:
            right_cards = self.find_other_cards(self.RPlayedCardsPos)
            left_cards = self.find_other_cards(self.LPlayedCardsPos)
            # 右边先出牌说明右边是地主，自己在右边下家；否则看左边
            if len(right_cards) > 0:
                return 0
            if len(left_cards) > 0:
                return 2
        return None

    def animation(self, cards):
        move_type = get_move_type(real_to_env(cards))
        animation_types = {4, 5, 13, 14, 8, 9, 10, 11, 12}
        if move_type["type"] in animation_types or len(cards) >= 6:
            return True

    def waitUntilNoAnimation(self, ms=0.1):
        ani = self.haveAnimation(ms)
        if ani:
            print("\n检测到炸弹、顺子、飞机 Biu~~ Biu~~  Bomb!!! Bomb!!!")
            return True
        return False

    def haveAnimation(self, waitTime=0.1):
        regions = [
            (960, 160, 960 + 20, 160 + 20),  # 下家动画位置
            (485, 160, 485 + 20, 160 + 20),  # 上家动画位置
            (700, 400, 700 + 20, 400 + 20),  # 自己上方动画位置
        ]
        img, _ = helper.Screenshot()
        lastImg = img
        for i in range(2):
            time.sleep(waitTime)
            img, _ = helper.Screenshot()
            for region in regions:
                if compareImage(img.crop(region), lastImg.crop(region)):
                    return True
            lastImg = img

        return False
