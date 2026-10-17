# DouZero_For_HLDDZ_FullAuto

> 把 [DouZero](https://github.com/kwai/DouZero) 强化学习斗地主 AI 接到**新版《欢乐斗地主》客户端**上，实现全自动叫牌、加倍与出牌。

仓库地址：<https://github.com/sltlala/DouZero_For_HLDDZ_FullAuto>
基于 [DouZero](https://github.com/kwai/DouZero) 与 [DouZero_For_Happy_DouDiZhu](https://github.com/tianqiraf/DouZero_For_HappyDouDiZhu)，并感谢 girgle（QQ群马云）完成新版欢乐斗地主的适配。

---

## 功能

- **自动叫牌 / 抢地主**：用自训练的胜率网络给手牌打分，按阈值决定叫、抢或不叫。
- **自动加倍**：地主/农民分别判断普通加倍、超级加倍，满足条件时自动明牌。
- **自动出牌**：DouZero resnet 模型估值 + 残局「能否一手跑完」搜索 + 炸弹风控，界面同步显示 AI 推荐与候选分数。
- **手动 / 自动随时切换**：手动模式下只给建议并播放提示音，由你自己点牌。
- **后台窗口截图**：按窗口区域截图（`PrintWindow`），游戏窗口被其它窗口遮挡也能识别。
- **记牌器**：实时统计三方已出牌，界面顶部按 大/小/2/A/K/Q/J/10/9/8/7/6/5/4/3 显示剩余张数。

## 环境要求

| 项目 | 要求 |
|---|---|
| 操作系统 | Windows 10 / 11（依赖 `win32api` 与 Unity 窗口句柄） |
| Python | 3.8 – 3.11 |
| 游戏 | 欢乐斗地主以 **窗口模式** 运行，窗口保持默认 **1440×810** |
| 系统缩放 | **100%**（识别坐标按 1440×810 硬编码） |
| 其它 | 游戏窗口**不要最小化**；背景使用默认蓝色桌子 |

## 安装

依赖统一声明在 [pyproject.toml](pyproject.toml)（PEP 621 标准项目格式）。

```powershell
python -m venv .venv
.\.venv\Scripts\activate

# 可编辑安装（推荐：改代码立即生效）
pip install -e .

# 或者只装依赖、不安装本项目
pip install .

# 需要跑测试时
pip install -e ".[dev]"
```

> `torch` 在 PyPI 的 Windows 轮子默认是 CPU 版；需要 CUDA 版请按
> [pytorch.org](https://pytorch.org/get-started/locally/) 单独安装，程序检测到可用 GPU 会自动使用。

## 运行

```powershell
# 方式一
python mian.py

# 方式二（安装后生成的命令）
douzero-hlddz
```

资源路径（`data.json`、`pics/`、`music/`、`weights/`、`baselines/`）由 [app_paths.py](app_paths.py) 统一解析为绝对路径，因此不强制从项目根目录启动。

## 使用流程

1. 以**窗口模式**启动欢乐斗地主，进入「斗地主合集」→ 经典场 / 欢乐经典场（推荐经典场），选好场次，停在**开始游戏**界面。
2. 运行本程序，界面出现在桌面左上角。
3. 点 **手动**（只提示不代打）或 **自动**（全自动）；运行中可随时互切。
4. 需要停止点 **停止**；想重新开始再点一次手动/自动。
5. 程序会自动处理「继续游戏 / 快速开始 / 领奖 / 开始游戏」等弹窗，可连续对局。

> 图文版操作说明见仓库内的 `使用说明.docx`。

## 界面与阈值

界面右侧是 10 个决策阈值（保存在 `data.json`，点「重置」恢复默认）：

| 控件 | 默认值 | 作用 |
|---|---|---|
| 叫地主 `bid1` | 0.2 | 自己第一个叫地主时的分数阈值 |
| 一抢 `bid2` | 0.4 | 已经有人叫过、轮到自己抢时的阈值 |
| 二抢 `bid3` | 0.5 | 还没人叫过、轮到自己抢时的阈值 |
| 地主不是抢来的：超加 `jiabei1` / 加倍 `jiabei2` | 0.5 / 0.4 | 自己当地主且不是抢来的 |
| 地主是抢来的：超加 `jiabei3` / 加倍 `jiabei4` | 0.6 / 0.5 | 自己当地主且是抢来的 |
| 农民：超加 `jiabei7` / 加倍 `jiabei8` | 1.2 / 1.0 | 自己当农民时的加倍判断 |
| 明牌 `mingpai` | 1.5 | 地主手牌胜率高于该值且满 20 张时自动明牌 |

分数越高越激进；调低 = 更容易叫/加倍，调高 = 更保守。

## 模型文件

| 目录 | 内容 | 说明 |
|---|---|---|
| `baselines/resnet/*.ckpt` | DouZero 出牌模型（地主 / 上家 / 下家） | ADP 目标训练的 resnet 版本；`baselines/resnet/old/` 是备份 |
| `weights/*.pkl` | 叫牌、加倍用的胜率回归网络 | `bid_weights.pkl`、`farmer_weights.pkl`、`landlord_weights.pkl`、`landlord_{up,down}_weights.pkl` |

更换出牌模型：修改 [mian.py](mian.py) 中 `Worker.model_path_dict`：

```python
self.model_path_dict = {
    'landlord':      str(BASELINES_DIR / "resnet" / "resnet_landlord.ckpt"),
    'landlord_up':   str(BASELINES_DIR / "resnet" / "resnet_landlord_up.ckpt"),
    'landlord_down': str(BASELINES_DIR / "resnet" / "resnet_landlord_down.ckpt"),
}
```

> ⚠️ 模型类型由**路径字符串**判定：路径中含 `resnet` 才使用 resnet 观测编码，否则退回旧版 LSTM 编码。换目录/改名时务必保留这个关键字。

## 自动叫牌 / 加倍原理

用 DouZero 自我博弈 N 局，对随机到的每种手牌再随机生成若干种对手手牌，把「该牌型 + 赢的局数」喂给一个小的全连接/卷积网络训练，得到「手牌 → 预期胜率」的映射；实际对局时按预期胜率与阈值决定叫牌、抢地主、加倍与明牌。

## 目录结构

```
DouZero_For_HLDDZ_FullAuto/
├── mian.py            # 主程序：GUI 窗口 + Worker 自动化状态机
├── MainWindow.ui/.py  # Qt 界面（.py 由 pyuic5 从 .ui 生成）
├── GameHelper.py      # 窗口截图、模板匹配、后台点击
├── DetermineColor.py  # HSV 判色（区分大王/小王）
├── app_paths.py       # 资源路径统一解析
├── search_utility.py  # 残局「一手跑完」路径搜索
├── BidModel.py        # 叫牌胜率网络
├── FarmerModel.py     # 农民胜率网络
├── LandlordModel.py   # 地主胜率网络
├── douzero/           # 内嵌并适配过的 DouZero 库（env / dmc / evaluation）
├── baselines/resnet/  # DouZero 出牌模型
├── weights/           # 叫牌/加倍胜率模型
├── pics/              # 95 张模板图（牌面 + 按钮/弹窗）
├── music/             # 提示音
├── tests/             # 回归测试
└── data.json          # 决策阈值
```

## 常见问题

**识别不到牌 / 点击位置偏移**
- 游戏窗口被拉伸过 → 用句柄工具把窗口恢复成 1440×810（程序每次截图也会尝试强制改回）；
- 系统缩放不是 100%；
- 游戏窗口被最小化（必须保持窗口可见，只是允许被遮挡）；
- 换了皮肤或非默认蓝色桌子。

**提示找不到游戏窗口**
程序通过 `UnityWndClass` 查找窗口，需要游戏已以窗口模式启动；先开游戏、再点手动/自动。

**叫牌得分看起来是乱数**
`weights/bid_weights.pkl` 缺失或与网络结构不匹配时，启动会打印 `[BidModel] 警告: ...`，此时叫牌分数没有参考价值。

**导入 torch / cv2 报错**
没有激活虚拟环境，或装的是另一个 Python 解释器；用 `python -c "import sys; print(sys.executable)"` 确认。

**已知限制**
- 有较低几率把王炸识别为不出；
- 所有识别坐标与点击位置都按 1440×810 硬编码，仅支持该窗口尺寸；
- 默认 CPU 推理，出牌思考约数百毫秒。

## 开发与测试

```powershell
pip install -e ".[dev]"

pytest                              # 全量回归（18 个用例）
python tests/test_regressions.py    # 核心逻辑专项
python tests/test_robustness.py     # 健壮性专项
python tests/dmc_smoke.py           # 真实跑一次 DMC 训练（CPU 约 10s）
```

训练入口在 `douzero/dmc`，安装后直接用命令行的方式：

```powershell
douzero-train --model_type resnet --savedir douzero_checkpoints
# 或
python -m douzero.dmc.dmc --model_type resnet --savedir douzero_checkpoints
```

`--model_type` 可选 `resnet`（默认，对应 `ResnetModel`）或 `old`（LSTM），必须与实际使用的权重/观测编码一致。
`--help` 可查看全部参数（帧数、batch、学习率、存盘间隔等）。

## 免责声明

本项目仅供**学习与技术交流**，请勿用于任何商业或违规用途，否则后果自负。使用前请自行确认是否符合游戏服务条款。

## 鸣谢

- [DouZero](https://github.com/kwai/DouZero)：快手团队的开源斗地主 AI
- [DouZero_For_Happy_DouDiZhu](https://github.com/tianqiraf/DouZero_For_HappyDouDiZhu)：游戏对接与自动化思路的起点
- girgle（QQ群马云）：新版欢乐斗地主适配

## 交流

QQ 群：565142377（密码 `douzero`）
