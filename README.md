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
| 游戏 | 欢乐斗地主以 **窗口模式** 运行 |
| 窗口尺寸 | **可任意缩放**：识别前会把客户区画面归一化到 1440×810 再匹配（16:9 布局） |
| 系统缩放 | 任意（程序启动时切换为 Per-Monitor DPI 感知，按物理像素工作） |
| 权限 | 若游戏以管理员身份运行（新版 QQ 欢乐斗地主常见），本程序**也必须以管理员身份运行** |
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
# 推荐：双击 run.bat（会自动请求管理员权限；游戏以管理员运行时必须提权）
run.bat

# 或手动以管理员身份打开 PowerShell 后：
python -m hlddz

# 安装后生成的命令（同样需要管理员终端）
douzero-hlddz

# 兼容旧入口（等价于 python -m hlddz）
python mian.py
```

资源路径（`data.json`、`pics/`、`music/`、`weights/`、`baselines/`）由 [hlddz/paths.py](hlddz/paths.py) 统一解析为绝对路径，因此不强制从项目根目录启动。

启动后控制台会先打印横幅（版本 / 项目根目录 / 日志路径），并且**输出同时写入 `logs/console.log`**：

```
================================================================
  DouZero_For_HLDDZ_FullAuto v5.7.0
  项目根目录 : D:\...\DouZero_For_HLDDZ_FullAuto
  日志文件   : D:\...\logs\console.log
================================================================
正在加载模板与模型（约 3-5 秒）...
界面已就绪：进入对局后点「手动」或「自动」开始
```

> **看不到输出时**：`run.bat` 提权会**新开一个控制台窗口**（标题 `DouZero HLDDZ`），
> 原来那个窗口 3 秒后自动关闭——请到新窗口看，或直接看 `logs/console.log`。
> 双击启动后窗口会一直保留（程序退出时停住并显示退出码）。

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

更换出牌模型：修改 [hlddz/worker.py](hlddz/worker.py) 中 `Worker.model_path_dict`：

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
├── mian.py                  # 兼容入口（等价于 python -m hlddz）
├── pyproject.toml           # 依赖与打包配置
├── run.bat                  # 一键提权启动
├── data.json                # 决策阈值（界面可改）
├── hlddz/                   # ★ 应用代码
│   ├── __main__.py          # python -m hlddz 入口
│   ├── app.py               # GUI 外壳：MyPyQT_Form + main()
│   ├── worker.py            # 自动化编排：Worker 线程（状态机/识别/出牌）
│   ├── paths.py             # 资源路径统一解析（绝对路径）
│   ├── runtime.py           # 全局单例 helper（截图/模板）
│   ├── perception/
│   │   ├── screen.py        # 窗口截图、PrintWindow/BitBlt、点击、坐标映射
│   │   └── color.py         # HSV 判色（区分大王/小王）
│   ├── models/
│   │   ├── bid.py           # 叫牌胜率网络
│   │   ├── farmer.py        # 农民胜率网络
│   │   └── landlord.py      # 地主胜率网络
│   ├── engine/
│   │   └── search.py        # 残局「一手跑完」路径搜索
│   └── ui/
│       ├── main_window.ui   # Qt Designer 源文件
│       ├── main_window.py   # pyuic5 生成（改 .ui 后需重新生成）
│       └── resources_rc.py  # picture.qrc 编译产物
├── douzero/                 # 内嵌并适配过的 DouZero 库（env / dmc / evaluation）
├── baselines/resnet/        # DouZero 出牌模型
├── weights/                 # 叫牌/加倍胜率模型
├── pics/                    # 95 张模板图（牌面 + 按钮/弹窗）
├── music/                   # 提示音
├── tests/                   # 回归测试 + 环境自检脚本
├── tools/debug_screenshot.py# 坐标调试小工具
├── logs/                    # 运行日志与识别失败现场截图（不入库）
├── style.qss / picture.qrc / LICENSE / 使用说明.docx
```

改动界面后重新生成代码：

```powershell
# 改 hlddz/ui/main_window.ui 之后
python -m PyQt5.uic.pyuic hlddz/ui/main_window.ui -o hlddz/ui/main_window.py
```

### 代码分层

| 层 | 位置 | 职责 |
|---|---|---|
| 入口 | `hlddz/app.py:main`、`hlddz/__main__.py` | 创建 QApplication、装配窗口 |
| 界面 | `hlddz/app.py`、`hlddz/ui/` | 记牌器、出牌展示、阈值面板、按钮 |
| 编排 | `hlddz/worker.py:Worker` | QThread 状态机：等开局 → 叫牌/加倍 → 识别 → 出牌 → 跟牌 |
| 感知 | `hlddz/perception/` | 截图归一化、模板匹配、牌面识别、点击坐标换算 |
| 决策 | `douzero/`、`hlddz/models/`、`hlddz/engine/` | DouZero 出牌估值、胜率网络、残局搜索 |
```

## 常见问题

**识别不到牌 / 点击位置偏移**
- 游戏窗口被最小化（必须保持窗口可见，只是允许被遮挡）；
- 换了皮肤或非默认蓝色桌子；
- 窗口缩放本身**不再影响识别**：程序会把客户区画面等比归一化到 1440×810；
  出牌时点击坐标也按当前客户区尺寸反向换算，不依赖固定窗口大小。

## 识别与操控的兼容性设计

| 环节 | 做法 |
|---|---|
| 截图 | 按**客户区**尺寸 `PrintWindow(PW_CLIENTONLY\|PW_RENDERFULLCONTENT)`，失败依次退回 `PW_CLIENTONLY`、屏幕 `BitBlt` |
| 尺寸无关 | 截图统一缩放（LANCZOS）到规范画布 1440×810，所有区域常量与模板都基于该画布 |
| 高 DPI | 进程启动即设为 Per-Monitor DPI Aware，窗口坐标/光标位置都是物理像素，不再出现 1152×648 之类的虚拟化坐标 |
| 牌面识别 | 多尺度模板匹配 + 跨牌型按 x 聚类（一张牌只取最高分牌型），底牌自动用缩小后的手牌模板 |
| 点击 | 规范画布坐标 → 客户区坐标 → `ClientToScreen`；发送 `WM_MOUSEMOVE`(先更新悬停) + `WM_LBUTTONDOWN/UP` |
| 出牌定位 | 用每张牌**实际识别到的 x 坐标**，不再依赖固定牌间距 45.6 |
| 失败兜底 | 底牌/手牌连续 10 次识别失败自动存图到 `logs/`，控制台打印识别结果 |

**提示找不到游戏窗口**
程序通过 `UnityWndClass` 查找窗口，需要游戏已以窗口模式启动；先开游戏、再点手动/自动。

**提示 `SetActiveWindow` / `MoveWindow` 被拒绝，或截图全黑**
游戏以管理员身份运行，而本程序没有提权，Windows 的 UIPI 会拦截所有跨权限级别的窗口操作。
用 `run.bat` 启动（自动提权），或右键“以管理员身份运行”。

**提示“游戏窗口已最小化，无法识别”**
从任务栏把游戏窗口恢复出来即可；允许被其它窗口遮挡（截图走 `PrintWindow`），但不能最小化。

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
python tests/env_probe.py           # 环境自检：窗口/权限/截图/模板识别（需管理员）
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
