# -*- coding: utf-8 -*-
"""健壮性回归测试：工作目录无关、阈值读写、常量单一来源、停止方式。

运行： python tests/test_robustness.py   或 pytest
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_resource_paths_are_absolute_and_exist():
    import app_paths

    for p in (app_paths.DATA_JSON, app_paths.PICS_DIR, app_paths.MUSIC_DIR,
              app_paths.WEIGHTS_DIR, app_paths.BASELINES_DIR, app_paths.STYLE_QSS):
        assert p.is_absolute(), "%s 不是绝对路径" % p
        assert p.exists(), "%s 不存在" % p
    assert app_paths.ROOT == ROOT


def test_works_from_any_working_directory():
    """旧代码用 './pics'、'./weights'、'data.json' 等相对路径，换工作目录就崩/静默失败。"""
    code = (
        "import sys; sys.path.insert(0, r'%s')\n"
        "import app_paths, BidModel, FarmerModel, LandlordModel\n"
        "h = __import__('GameHelper').GameHelper()\n"
        "assert len(h.Pics) > 50, len(h.Pics)\n"
        "assert __import__('os').path.isfile(BidModel.WEIGHTS_PATH)\n"
        "assert BidModel._loaded is True\n"
        "print('ok', len(h.Pics))\n"
    ) % ROOT
    with tempfile.TemporaryDirectory() as tmp:
        out = subprocess.run([sys.executable, "-c", code], cwd=tmp,
                             capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "ok " in out.stdout, out.stdout


def test_threshold_json_roundtrip():
    """阈值读写必须走同一份 data.json，并且键集合与界面消费的键一致。"""
    import GameHelper

    original = GameHelper.read_json()
    try:
        sample = dict(original)
        sample["bid1"] = "0.31"
        GameHelper.write_json(sample)
        assert GameHelper.read_json()["bid1"] == "0.31"
    finally:
        GameHelper.write_json(original)
    assert GameHelper.read_json() == original

    from app_paths import DATA_JSON
    keys = set(json.loads(DATA_JSON.read_text(encoding="utf-8")))
    assert keys == {"bid1", "bid2", "bid3", "jiabei1", "jiabei2", "jiabei3",
                    "jiabei4", "jiabei7", "jiabei8", "mingpai"}


def test_card_constants_have_single_source():
    """mian.py 里不再复制牌面映射/炸弹表，必须与引擎共用同一份对象。"""
    import mian
    import douzero.env.game as game

    assert mian.EnvCard2RealCard is game.EnvCard2RealCard
    assert mian.RealCard2EnvCard is game.RealCard2EnvCard
    assert mian.AllEnvCard is game.AllEnvCard
    assert mian.bombs is game.bombs


def test_stop_is_cooperative_first():
    """停止时先请求协作退出，只有超时才 terminate()，不再无条件强杀。"""
    src = (ROOT / "mian.py").read_text(encoding="utf-8")
    stop_block = src.split("def stop(self, result):", 1)[1].split("\n    def ", 1)[0]
    assert "self.thread.stop_sign = True" in stop_block
    assert "self.thread.wait(" in stop_block
    assert stop_block.index("self.thread.wait(") < stop_block.index("self.thread.terminate()"), \
        "必须先等待协作退出，再考虑 terminate"


def test_threshold_handshake_race_removed():
    """启动流程不再依赖 emit + sleep(0.1) 的跨线程握手。"""
    src = (ROOT / "mian.py").read_text(encoding="utf-8")
    assert "write_threshold.emit" not in src
    # worker 启动前由主线程落盘
    run_block = src.split("def hand_game(self, result):", 1)[1].split("\n    def ", 1)[0]
    assert "self.write_threshold(None)" in run_block


def _run_all():
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print("PASS  %s" % name)
        except Exception:
            import traceback
            failed.append(name)
            print("FAIL  %s" % name)
            traceback.print_exc()
    print("\n%d passed, %d failed, %d total" % (len(tests) - len(failed), len(failed), len(tests)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
