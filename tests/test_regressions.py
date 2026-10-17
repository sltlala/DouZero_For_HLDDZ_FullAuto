# -*- coding: utf-8 -*-
"""针对本次修复的回归测试。

运行方式（两种都支持）：
    python tests/test_regressions.py
    pytest                      # 依赖 pyproject.toml 里的 dev extra

覆盖的 6 个严重问题：
    1. dmc 训练链：Env <-> GameEnv 协议、观测编码与缓冲区/learner 形状不一致
    2. FileWriter 构造即 NameError（gather_metadata 被注释）
    3. search_actions 永远 return None、且模块间循环导入
    4. BidModel 权重装进 net2、predict() 却用没权重的 net
    5. 王炸未计入 bomb_num（bombs 里写成 [30, 20]）
    6. jiabei5/6 死阈值配置
"""
import json
import os
import subprocess
import sys
import threading
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # 方便定位 data.json / 权重（应用代码本身已改为绝对路径，见 app_paths.py）

UP_HAND = [3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8]
DOWN_HAND = [9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14]
THREE_CARDS = [17, 17, 17]


def _make_game_env(landlord_hand, heuristics=True, agent=None):
    from douzero.env.game import GameEnv
    from douzero.env.env import DummyAgent

    players = {p: DummyAgent(p) for p in ("landlord", "landlord_up", "landlord_down")}
    if agent is not None:
        players = ["landlord", agent]
    env = GameEnv(players, heuristics=heuristics)
    env.card_play_init({
        "landlord": sorted(landlord_hand),
        "landlord_up": list(UP_HAND),
        "landlord_down": list(DOWN_HAND),
        "three_landlord_cards": list(THREE_CARDS),
    })
    return env


# ---------------------------------------------------------------- 问题 5：王炸计数
def test_king_bomb_is_counted():
    """王炸必须计入 bomb_num；旧代码 bombs 里写的是 [30, 20]，导致永远匹配不上。"""
    from douzero.env.game import bomb_keys

    assert tuple(sorted([20, 30])) in bomb_keys
    assert tuple(sorted([30, 20])) in bomb_keys  # 顺序无关

    hand = [20, 30] + [3, 3, 3, 3] + list(range(4, 15)) + [17, 17, 17]
    assert len(hand) == 20
    env = _make_game_env(hand)
    assert env.bomb_num == 0

    env.step("landlord", [20, 30])
    assert env.bomb_num == 1, "王炸没有被计入 bomb_num"
    assert env.get_bomb_num() == 1

    # 普通炸弹
    env2 = _make_game_env([3, 3, 3, 3] + [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 17,
                                          20, 30, 4, 5, 6, 7, 8])
    env2.step("landlord", [3, 3, 3, 3])
    assert env2.bomb_num == 1


def test_adp_reward_uses_bomb_count():
    """ADP 回报 = ±2^bomb_num，bomb_num 数错会直接算错训练目标与结算分。"""
    from douzero.env.env import Env

    env = Env("adp")
    e = env._env
    e.bomb_num = 3
    e.info_sets["landlord"].player_hand_cards = []  # 地主先出完
    e.game_done()
    assert e.get_winner() == "landlord"
    assert env._get_reward() == 8.0                     # 2^3
    assert e.num_scores["landlord"] == 16              # base 2 * 2^3

    env = Env("wp")
    e = env._env
    e.bomb_num = 3
    e.info_sets["landlord"].player_hand_cards = []
    e.game_done()
    assert env._get_reward() == 1.0                    # wp 不含炸弹倍率


# ---------------------------------------------------------------- 问题 4：BidModel
def test_bidmodel_weights_actually_loaded():
    """bid_weights.pkl 的键名必须与 BidModel 里承载权重的网络完全一致，且确实加载成功。"""
    import torch
    import BidModel

    state = torch.load(BidModel.WEIGHTS_PATH, map_location="cpu")
    own = BidModel.net2.state_dict()
    assert set(state.keys()) == set(own.keys()), "bid_weights.pkl 与网络结构不匹配"
    assert BidModel._loaded is True

    # 加载后参数不应等于随机初始化：与全零/原始随机值比较意义不大，
    # 这里改为检查两个不同手牌能得到不同且有限的分数（真实前向可用）。
    a = BidModel.predict_score("333444569TTJJQKK2")
    b = BidModel.predict_score("3456789TJQKA2XD")
    assert a != b
    assert a == a and abs(a) < 1e6

    # 失效的旧 API 必须已删除，避免再被误调用
    assert not hasattr(BidModel, "predict")
    assert not hasattr(BidModel, "Net")


# ---------------------------------------------------------------- 问题 2：FileWriter
def test_file_writer_constructs_without_nameerror():
    """旧代码调用被注释掉的 gather_metadata()，构造 FileWriter 就 NameError。"""
    from douzero.dmc.file_writer import FileWriter, gather_metadata

    meta = gather_metadata()
    assert meta["successful"] is False
    assert "date_start" in meta and "env" in meta

    with tempfile.TemporaryDirectory() as tmp:
        writer = FileWriter(xpid="pytest", xp_args={"a": 1}, rootdir=tmp)
        writer.log({"loss_landlord": 0.5})
        writer.close(successful=True)
        meta_path = Path(writer.paths["meta"])
        assert meta_path.exists()
        saved = json.loads(meta_path.read_text(encoding="utf-8"))
        assert saved["xpid"] == "pytest"
        assert saved["args"] == {"a": 1}
        assert saved["successful"] is True
        assert Path(writer.paths["logs"]).exists()


# ---------------------------------------------------------------- 问题 3：search_utility
def test_search_actions_returns_path_list():
    """旧版永远 return None（靠副作用传结果），递归里的判空分支还是死代码。"""
    from douzero.env.game import GameEnv  # noqa: F401  先按 mian.py 的顺序导入一次
    import search_utility as su

    paths = []
    ret = su.search_actions([20, 30, 10, 10],
                            [4, 13, 5, 5, 5, 5, 8, 8, 8, 8, 12, 12, 13],
                            paths, rival_move=[13])
    assert ret is paths, "search_actions 必须把 path_list 返回给调用方"
    assert len(paths) == 1
    assert sorted(map(sorted, paths[0])) == [[10, 10], [20, 30]]

    # 找不到跑完路径时返回空列表（而不是 None）
    paths = []
    ret = su.search_actions([5, 5, 6, 6, 7, 7, 9, 9, 9, 12, 12, 12, 14, 17],
                            [3, 3, 3, 3, 4, 4, 5, 6, 7, 8, 8, 9, 10, 11, 13, 20, 30],
                            paths, rival_move=[17])
    assert ret is paths and paths == []

    # 超过 100 条时安全提前退出
    big = [None] * 101
    assert su.search_actions([3, 4], [5, 6], big) is big


def test_search_utility_can_be_imported_standalone():
    """旧版 game.py 顶层 import search_utility，而 search_utility 又反向 import douzero.env，
    形成循环导入：`python search_utility.py` / `import search_utility` 直接 ImportError。"""
    out = subprocess.run([sys.executable, "-c", "import search_utility; print('ok')"],
                         cwd=str(ROOT), capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout

    # 反方向（先 import douzero.env.*）也必须成立
    out = subprocess.run([sys.executable, "-c",
                          "from douzero.env.game import GameEnv; import search_utility; print('ok')"],
                         cwd=str(ROOT), capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


# ---------------------------------------------------------------- 问题 1：Env <-> GameEnv
def test_env_selfplay_terminates():
    """旧版 Env.step() 不把 position/action 交给引擎，整局永远 pass、永不结束。"""
    import numpy as np
    from douzero.env.env import Env

    rng = np.random.RandomState(0)
    for model_type in ("resnet", "old"):
        env = Env("adp", model_type=model_type)
        obs = env.reset()
        steps, done = 0, False
        while not done and steps < 1000:
            acts = obs["legal_actions"]
            assert acts
            obs, reward, done, _ = env.step(acts[rng.randint(len(acts))])
            steps += 1
        assert done, "model_type=%s 的对局没有正常结束" % model_type
        assert steps > 5, "步数异常少，说明动作没有被真正执行"
        assert reward in (1.0, -1.0) or abs(reward) >= 2.0


def test_gameenv_accepts_dict_players():
    """自对局时 players 是 dict（三家真实手牌），旧代码 self.players[0] 会 KeyError。"""
    hand = [3, 3, 3, 3] + [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 17, 17, 20, 30]
    env = _make_game_env(hand)
    env.step("landlord", [3, 3, 3, 3])  # 不抛 KeyError
    assert env.info_sets["landlord"].player_hand_cards == sorted(hand[4:])


def test_heuristics_flag_does_not_override_agent_action():
    """训练时 heuristics=False，引擎不得把模型选的牌改写成"可直接出完"的牌。"""
    class ScriptedAgent:
        def __init__(self, action, action_list):
            self.action = action
            self.action_list = action_list

        def act(self, infoset):
            return self.action, 1.0, self.action_list

    # 手牌只有 3,3,3,4：智能体只想出单张 4，但 3,3,3,4 是一手出完的牌
    # 注意 DeepAgent.act 返回的是 [(动作, 价值), ...]
    hand = [3, 3, 3, 4]
    agent = ScriptedAgent([4], [([4], 1.0), ([3, 3, 3, 4], 0.9)])

    env_train = _make_game_env(hand, heuristics=False, agent=agent)
    env_train.step("landlord")
    assert env_train.card_play_action_seq[-1][1] == [4], "训练模式下动作被启发式改写"

    env_live = _make_game_env(hand, heuristics=True, agent=agent)
    env_live.step("landlord")
    assert env_live.card_play_action_seq[-1][1] == [3, 3, 3, 4], "线上模式的「直接出完」兜底失效"


# ---------------------------------------------------------------- 问题 1：dmc 形状一致性
def _make_flags(**overrides):
    from douzero.dmc.arguments import parser
    flags = parser.parse_args([])
    flags.unroll_length = 4
    flags.batch_size = 2
    flags.num_buffers = 2
    flags.num_threads = 1
    flags.model_type = "resnet"
    for k, v in overrides.items():
        setattr(flags, k, v)
    return flags


def test_create_buffers_matches_model_inputs():
    """缓冲区形状必须与 learner 模型输入一致（旧版只按 old 编码建，resnet learner 必崩）。"""
    from douzero.dmc.utils import create_buffers
    from douzero.dmc.env_utils import get_device

    assert get_device(0).type == ("cuda" if _has_cuda() else "cpu")

    flags = _make_flags()
    buffers = create_buffers(flags)
    assert len(buffers) >= 1
    b = buffers[0]["landlord"]
    assert tuple(b["obs_x_no_action"][0].shape) == (flags.unroll_length, 15)
    assert tuple(b["obs_z"][0].shape) == (flags.unroll_length, 40, 54)
    assert "obs_action" not in b, "resnet 编码不需要 obs_action"

    old_flags = _make_flags(model_type="old")
    old_b = create_buffers(old_flags)[0]["landlord"]
    assert tuple(old_b["obs_x_no_action"][0].shape) == (old_flags.unroll_length, 319)
    assert tuple(old_b["obs_z"][0].shape) == (old_flags.unroll_length, 5, 162)
    assert "obs_action" in old_b


def _has_cuda():
    import torch
    return torch.cuda.is_available()


def test_actor_output_feeds_learner():
    """把 act() 里真正的采样流程跑一遍，再用同样形状的 batch 调 learn()：
    旧代码这里 x 维度 = 15+54 = 69、z = (T,5,162)，ResnetModel 会直接 shape error。"""
    import torch
    from douzero.dmc.utils import create_env
    from douzero.dmc.env_utils import Environment, get_device
    from douzero.dmc.models import Model
    from douzero.dmc.dmc import learn
    from douzero.dmc.utils import create_optimizers

    flags = _make_flags()
    model = Model(device=get_device(0))
    model.eval()

    env = Environment(create_env(flags), 0)
    position, obs, env_output = env.initial()

    records = []
    while len(records) < flags.unroll_length:
        with torch.no_grad():
            agent_output = model.forward(position, obs["z_batch"], obs["x_batch"], flags=flags)
        idx = int(agent_output["action"].cpu().detach().numpy())
        action = obs["legal_actions"][idx]
        records.append({
            "pos": position,
            "x": env_output["obs_x_no_action"],
            "z": obs["z_batch"][idx].cpu(),
        })
        position, obs, env_output = env.step(action)
        if obs is None:
            position, obs, env_output = env.initial()

    pos = records[0]["pos"]
    n = len(records)
    batch = {
        "obs_x_no_action": torch.stack([r["x"] for r in records]).unsqueeze(1),
        "obs_z": torch.stack([r["z"] for r in records]).unsqueeze(1),
        "target": torch.zeros(n, 1),
        "done": torch.zeros(n, 1, dtype=torch.bool),
        "episode_return": torch.zeros(n, 1),
    }
    batch["done"][0, 0] = True
    batch["episode_return"][0, 0] = 1.0

    assert tuple(batch["obs_x_no_action"].shape) == (n, 1, 15)
    assert tuple(batch["obs_z"].shape) == (n, 1, 40, 54)

    learner = Model(device=get_device(0))
    optimizers = create_optimizers(flags, learner)
    before = [p.detach().clone() for p in learner.get_model(pos).parameters()]

    stats = learn(pos, [], learner.get_model(pos), batch, optimizers[pos], flags, threading.Lock())
    assert "loss_" + pos in stats
    assert stats["loss_" + pos] == stats["loss_" + pos], "loss 是 NaN"

    after = [p.detach() for p in learner.get_model(pos).parameters()]
    assert any(not torch.equal(x, y) for x, y in zip(before, after)), "learn() 没有更新任何参数"


# ---------------------------------------------------------------- 问题 6：死配置
def test_dead_farmer_threshold_removed():
    data = json.loads((ROOT / "data.json").read_text(encoding="utf-8"))
    assert "jiabei5" not in data and "jiabei6" not in data
    assert "jiabei7" in data and "jiabei8" in data
    assert set(data) == {"bid1", "bid2", "bid3", "jiabei1", "jiabei2", "jiabei3",
                         "jiabei4", "jiabei7", "jiabei8", "mingpai"}

    mian_src = (ROOT / "mian.py").read_text(encoding="utf-8")
    # 注释里允许提到历史，但不能再有真正的读取/写入代码
    assert "data['jiabei5']" not in mian_src
    assert "data['jiabei6']" not in mian_src
    assert "jiabei_lineEdit_5" not in mian_src
    assert "jiabei_lineEdit_6" not in mian_src
    assert "FarmerJiabeiThresholdLow" not in mian_src

    ui_src = (ROOT / "MainWindow.py").read_text(encoding="utf-8")
    assert "jiabei_lineEdit_5" not in ui_src and "jiabei_lineEdit_6" not in ui_src
    assert "农民正常加倍" not in ui_src and "地主怂了" not in ui_src


# ---------------------------------------------------------------- 骨架
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
