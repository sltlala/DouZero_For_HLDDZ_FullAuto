# -*- coding: utf-8 -*-
"""兼容入口：等价于 `python -m hlddz`。

新代码请直接从 hlddz.app 导入，或使用 `python -m hlddz` / `douzero-hlddz`。
"""
import sys

from hlddz.app import main

if __name__ == "__main__":
    sys.exit(main())
