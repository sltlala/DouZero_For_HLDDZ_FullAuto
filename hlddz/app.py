# -*- coding: utf-8 -*-
"""GUI 外壳：主窗口（MyPyQT_Form）与程序入口 main()。"""
import os
import sys

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QFile, QTextStream, Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QTableWidgetItem

from . import __version__
from .paths import LOGS_DIR, MUSIC_DIR, ROOT, STYLE_QSS
from .perception.screen import play_sound, read_json, write_json
from .ui.main_window import Ui_Form
from .worker import AllCards, Worker


class MyPyQT_Form(QtWidgets.QWidget, Ui_Form):
    def __init__(self):
        super(MyPyQT_Form, self).__init__()
        self.setupUi(self)
        self.setWindowFlags(QtCore.Qt.WindowMinimizeButtonHint |  # 使能最小化按钮
                            QtCore.Qt.WindowStaysOnTopHint |  # 窗体总在最前端
                            QtCore.Qt.WindowCloseButtonHint)
        self.setWindowIcon(QIcon(':/pics/favicon.ico'))
        self.setWindowTitle("DouZero欢乐斗地主  v5.7")
        play_sound(str(MUSIC_DIR / "2.wav"))
        self.setFixedSize(self.width(), self.height())  # 固定窗体大小
        self.move(20, 20)
        window_pale = QtGui.QPalette()

        self.setPalette(window_pale)
        self.setWindowOpacity(1)
        self.HandButton.clicked.connect(self.hand_game)
        self.AutoButton.clicked.connect(self.auto_game)
        self.StopButton.clicked.connect(self.stop)
        self.ResetButton.clicked.connect(self.init_threshold)
        self.ClassicsButton.setStyleSheet('background-color: rgba(255, 170, 0, 0.5);')

        self.read_threshold()
        self.Players = [self.RPlayedCard, self.PredictedCard, self.LPlayedCard]
        self.thread = Worker()
        self.ClassicsButton.setStyleSheet('background-color: rgba(255, 170, 0, 0.5);')
        self.thread.auto_game.connect(self.auto_game)
        self.thread.hand_game.connect(self.hand_game)
        self.thread.init_interface.connect(self.init_interface)
        self.thread.player_display.connect(self.player_display)
        self.thread.state_display.connect(self.state_display)
        self.thread.three_cards_display.connect(self.three_cards_display)
        self.thread.bid_display.connect(self.bid_display)
        self.thread.pre_display.connect(self.pre_display)
        self.thread.stop.connect(self.stop)
        self.thread.winrate_display.connect(self.winrate_display)
        self.thread.pre_cards_display.connect(self.pre_cards_display)
        self.thread.left_cards_display.connect(self.left_cards_display)
        self.thread.right_cards_display.connect(self.right_cards_display)
        self.thread.recorder_display.connect(self.cards_recorder)

    def hand_game(self, result):
        self.thread.auto_sign = False
        self.write_threshold(None)  # 先把界面阈值落盘，再启动 worker（消除旧版的跨线程握手竞态）
        self.thread.start()
        self.AutoButton.setStyleSheet('background-color: none;')
        self.HandButton.setStyleSheet('background-color: rgba(255, 85, 255, 0.5);')

    def auto_start(self):
        """程序启动后默认进入自动模式（等价于点一下「自动」）。"""
        try:
            if self.thread.isRunning():
                return
            print("默认自动模式：正在启动自动化线程...", flush=True)
            self.auto_game(False)
        except Exception as e:
            print("自动启动失败: %s" % e)

    def auto_game(self, result):
        self.thread.auto_sign = True
        self.write_threshold(None)
        self.thread.start()

        self.AutoButton.setStyleSheet('background-color: rgba(255, 85, 255, 0.5);')
        self.HandButton.setStyleSheet('background-color: none;')

    def stop(self, result):
        print()
        print("停止线程")
        # 先请求协作式退出（worker 的循环都在检查 stop_sign / RunGame），
        # 只有在 1 秒内还没退出时才强杀，避免 QThread.terminate() 把
        # 正在跑的推理/截图硬切断。
        self.thread.stop_sign = True
        self.thread.RunGame = False
        if not self.thread.wait(1000):
            print("线程未响应，强制终止")
            self.thread.terminate()
            self.thread.wait(1000)
        self.init_interface(1)
        self.AutoButton.setStyleSheet('background-color: none;')
        self.HandButton.setStyleSheet('background-color: none;')

    def player_display(self, result):
        for player in self.Players:
            player.setStyleSheet('background-color: rgba(0, 255, 0, 0);')
        self.Players[result].setStyleSheet('background-color: rgba(0, 255, 0, 0.5);')

    def init_threshold(self):
        self.bid_lineEdit_1.setText("0.2")
        self.bid_lineEdit_2.setText("0.4")
        self.bid_lineEdit_3.setText("0.5")
        self.jiabei_lineEdit_1.setText("0.5")
        self.jiabei_lineEdit_2.setText("0.4")
        self.jiabei_lineEdit_3.setText("0.6")
        self.jiabei_lineEdit_4.setText("0.5")
        self.jiabei_lineEdit_7.setText("1.2")
        self.jiabei_lineEdit_8.setText("1.0")
        self.mingpai_lineEdit.setText("1.5")

        data = {'bid1': self.bid_lineEdit_1.text(), 'bid2': self.bid_lineEdit_2.text(),
                'bid3': self.bid_lineEdit_3.text(), 'jiabei1': self.jiabei_lineEdit_1.text(),
                'jiabei2': self.jiabei_lineEdit_2.text(), 'jiabei3': self.jiabei_lineEdit_3.text(),
                'jiabei4': self.jiabei_lineEdit_4.text(),
                'jiabei7': self.jiabei_lineEdit_7.text(),
                'jiabei8': self.jiabei_lineEdit_8.text(), 'mingpai': self.mingpai_lineEdit.text()}
        write_json(data)

    def write_threshold(self, result):
        data = {'bid1': self.bid_lineEdit_1.text(), 'bid2': self.bid_lineEdit_2.text(),
                'bid3': self.bid_lineEdit_3.text(), 'jiabei1': self.jiabei_lineEdit_1.text(),
                'jiabei2': self.jiabei_lineEdit_2.text(), 'jiabei3': self.jiabei_lineEdit_3.text(),
                'jiabei4': self.jiabei_lineEdit_4.text(),
                'jiabei7': self.jiabei_lineEdit_7.text(),
                'jiabei8': self.jiabei_lineEdit_8.text(), 'mingpai': self.mingpai_lineEdit.text()}
        write_json(data)

    def read_threshold(self):
        data = read_json()
        self.bid_lineEdit_1.setText(data['bid1'])
        self.bid_lineEdit_2.setText(data['bid2'])
        self.bid_lineEdit_3.setText(data['bid3'])
        self.jiabei_lineEdit_1.setText(data['jiabei1'])
        self.jiabei_lineEdit_2.setText(data['jiabei2'])
        self.jiabei_lineEdit_3.setText(data['jiabei3'])
        self.jiabei_lineEdit_4.setText(data['jiabei4'])
        self.jiabei_lineEdit_7.setText(data['jiabei7'])
        self.jiabei_lineEdit_8.setText(data['jiabei8'])
        self.mingpai_lineEdit.setText(data['mingpai'])

    def init_interface(self, result):
        self.WinRate.setText("评分")
        self.WinRate.setStyleSheet('background-color: none;')
        self.label.setText("游戏状态")
        self.BidWinrate.setText("叫牌得分")
        self.PreWinrate.setText("局前得分")
        self.label.setStyleSheet('background-color: none;')
        self.LPlayedCard.setText("上家出牌区域")
        self.RPlayedCard.setText("下家出牌区域")
        self.PredictedCard.setText("AI出牌区域")
        self.ThreeLandlordCards.setText("底牌")
        self.recorder2zero()
        for player in self.Players:
            player.setStyleSheet('background-color: none;')

    def state_display(self, result):
        self.label.setText(result)
        self.label.setStyleSheet('background-color: rgba(255, 0, 0, 0.5);')

    def three_cards_display(self, result):
        self.ThreeLandlordCards.setText(result)

    def bid_display(self, result):
        self.BidWinrate.setText(result)

    def pre_display(self, result):
        self.PreWinrate.setText(result)

    def pre_cards_display(self, result):
        self.PredictedCard.setText(result)
        self.PredictedCard.setStyleSheet('background-color: rgba(0, 255, 0, 0.5);')
        self.LPlayedCard.setStyleSheet('background-color: rgba(0, 255, 0, 0);')
        self.RPlayedCard.setStyleSheet('background-color: rgba(0, 255, 0, 0);')

    def left_cards_display(self, result):
        self.LPlayedCard.setText(result)
        self.PredictedCard.setStyleSheet('background-color: rgba(0, 255, 0, 0);')
        self.LPlayedCard.setStyleSheet('background-color: rgba(0, 255, 0, 0.5);')
        self.RPlayedCard.setStyleSheet('background-color: rgba(0, 255, 0, 0);')

    def right_cards_display(self, result):
        self.RPlayedCard.setText(result)
        self.PredictedCard.setStyleSheet('background-color: rgba(0, 255, 0, 0);')
        self.LPlayedCard.setStyleSheet('background-color: rgba(0, 255, 0, 0);')
        self.RPlayedCard.setStyleSheet('background-color: rgba(0, 255, 0, 0.5);')

    def winrate_display(self, result):
        self.WinRate.setText(result)
        self.WinRate.setStyleSheet('background-color: rgba(255, 85, 0, 0.4);')

    def cards_recorder(self, result):
        for i in range(15):
            char = AllCards[i]
            num = result.count(char)
            newItem = QTableWidgetItem(str(num))
            newItem.setTextAlignment(Qt.AlignHCenter)
            self.tableWidget.setItem(0, i, newItem)

    def recorder2zero(self):
        for i in range(15):
            newItem = QTableWidgetItem("0")
            newItem.setTextAlignment(Qt.AlignHCenter)
            self.tableWidget.setItem(0, i, newItem)


def _tee_output(log_path):
    """把 stdout/stderr 同时写到控制台和日志文件。

    用 run.bat 提权启动时，原控制台会立即关闭、输出转到新开的窗口；一旦那个窗口
    一闪而过（崩溃/被关掉）就什么也看不到。这里始终保留一份日志，便于事后排查。
    """
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(log_path, "a", encoding="utf-8", buffering=1)
    except Exception:
        return None

    class _Tee:
        def __init__(self, *streams):
            self._streams = streams

        def write(self, s):
            for st in self._streams:
                try:
                    st.write(s)
                    st.flush()
                except Exception:
                    pass

        def flush(self):
            for st in self._streams:
                try:
                    st.flush()
                except Exception:
                    pass

        def isatty(self):
            return False

    sys.stdout = _Tee(sys.stdout, fh)
    sys.stderr = _Tee(sys.stderr, fh)
    return fh


def main():
    """程序入口（pyproject.toml 里注册为 gui-script: douzero-hlddz）。

    默认**启动即自动模式**；加 `--manual` 参数启动则停在手动模式等你点按钮。
    """
    _tee_output(LOGS_DIR / "console.log")
    print("=" * 64)
    print("  DouZero_For_HLDDZ_FullAuto v%s" % __version__)
    print("  项目根目录 : %s" % ROOT)
    print("  当前目录   : %s" % os.getcwd())
    print("  日志文件   : %s" % (LOGS_DIR / "console.log"))
    print("=" * 64)
    print("正在加载模板与模型（约 3-5 秒）...", flush=True)

    argv = sys.argv[1:]
    start_manual = "--manual" in argv or "-m" in argv

    app = QtWidgets.QApplication(sys.argv)
    window = MyPyQT_Form()
    # 旧代码没有 open() 就直接 readAll()，Qt 会报 "device not open"，样式表永远是空的
    style_file = QFile(str(STYLE_QSS))
    if style_file.open(QFile.ReadOnly):
        stream = QTextStream(style_file)
        style_sheet = stream.readAll()
        window.setStyleSheet(style_sheet)
        style_file.close()
    else:
        print("样式表未找到或被占用: {}".format(STYLE_QSS))
    window.show()

    if start_manual:
        print("界面已就绪：手动模式，点「手动」或「自动」开始", flush=True)
    else:
        print("界面已就绪：默认自动模式，0.5 秒后自动开始（停止后可手动切换）", flush=True)
        # 等界面画完再起线程，避免首帧卡顿
        QtCore.QTimer.singleShot(500, window.auto_start)
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
