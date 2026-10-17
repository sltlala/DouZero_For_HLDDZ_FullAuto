"""DouZero 的 DMC（Deep Monte-Carlo）训练包。

`train` 采用延迟导入：包被 `import douzero.dmc` 时不立刻加载 dmc 模块，
这样 `python -m douzero.dmc.dmc` 不会触发
"found in sys.modules after import of package" 的 RuntimeWarning。
"""
from .arguments import parser

__all__ = ["train", "parser"]


def __getattr__(name):
    if name == "train":
        from .dmc import train
        return train
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
