"""手动冒烟：真正启动一次 dmc.train()（极小规模，CPU），验证 actor/learner 全链路。

不属于 pytest 用例（需要多进程、耗时数秒）。
用法： python tests/dmc_smoke.py
"""
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def main():
    from douzero.dmc.arguments import parser
    from douzero.dmc import train

    savedir = tempfile.mkdtemp(prefix="dmc_smoke_")
    sys.argv = [
        "dmc_smoke",
        "--xpid", "smoke",
        "--savedir", savedir,
        "--num_actor_devices", "1",
        "--num_actors", "1",
        "--num_buffers", "4",
        "--unroll_length", "4",
        "--batch_size", "2",
        "--num_threads", "1",
        "--total_frames", "240",
        "--disable_checkpoint",
        "--model_type", "resnet",
    ]
    flags = parser.parse_args()
    t0 = time.time()
    train(flags)
    dt = time.time() - t0

    log_path = Path(savedir) / "smoke" / "out.log"
    logs = Path(savedir) / "smoke" / "logs.csv"
    print("\n=== train() 正常返回，用时 %.1fs ===" % dt)
    print("out.log 存在:", log_path.exists(), " logs.csv 存在:", logs.exists())
    ok = True
    if logs.exists():
        lines = logs.read_text(encoding="utf-8").strip().splitlines()
        print("logs.csv 行数:", len(lines))
        print("表头:", lines[0][:120])
        print("最后一行:", lines[-1][:160])
        ok = ok and len(lines) > 1
        last = lines[-1].split(",")
        ok = ok and ("nan" not in lines[-1])
        print("nan 检查:", "通过" if "nan" not in lines[-1] else "存在 nan")
    else:
        ok = False
    print("smoke:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    main()
