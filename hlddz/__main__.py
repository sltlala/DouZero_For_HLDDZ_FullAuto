# -*- coding: utf-8 -*-
"""允许 `python -m hlddz` 启动图形界面。"""
import sys

from .app import main

if __name__ == "__main__":
    sys.exit(main())
