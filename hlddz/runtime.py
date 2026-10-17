# -*- coding: utf-8 -*-
"""进程内共享的运行时对象。

GameHelper 会加载 95 张模板图，构造较重；全进程只保留一个实例，
供 worker（自动化）与 perception（识别）共同使用。
"""
from .perception.screen import GameHelper

helper = GameHelper()
