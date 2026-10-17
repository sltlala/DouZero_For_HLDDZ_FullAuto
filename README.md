# DouZero_For_HLDDZ_FullAuto: 将DouZero用于欢乐斗地主自动化
* 本项目基于[DouZero](https://github.com/kwai/DouZero) 和  [DouZero_For_Happy_DouDiZhu](https://github.com/tianqiraf/DouZero_For_HappyDouDiZhu)
* 感谢 girgle（QQ群马云）对新版欢乐斗地主进行了适配，现在仓库代码已支持新版欢乐斗地主

* 环境配置见下方「安装」，依赖统一声明在 `pyproject.toml` 中

* 模型默认为ADP，更换模型请修改 `mian.py` 中的模型路径（`Worker.model_path_dict`），如果需要加载Resnet新模型，请保证游戏路径或文件名中存在关键词 "resnet"

  ```python
  self.model_path_dict = {
      'landlord': "baselines/resnet/resnet_landlord.ckpt",
      'landlord_up': "baselines/resnet/resnet_landlord_up.ckpt",
      'landlord_down': "baselines/resnet/resnet_landlord_down.ckpt"
  }
  ```

* 运行 `python mian.py` 即可（或安装后直接运行 `douzero-hlddz`）

## 安装

依赖统一声明在 [pyproject.toml](pyproject.toml) 中（PEP 621 标准项目格式），不再使用 requirements.txt。

```bash
# 建议先建虚拟环境
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # Linux/macOS

# 方式一：可编辑安装（推荐，改代码后立即生效）
pip install -e .

# 方式二：只装依赖、不安装本项目
pip install .

# 另需编辑 MainWindow.ui 时再装开发依赖
pip install -e ".[dev]"
```

> 说明：`torch` 在 PyPI 上的 Windows 轮子默认就是 CPU 版；需要 CUDA 版请按
> [pytorch.org](https://pytorch.org/get-started/locally/) 的指引单独安装，本项目在有 GPU 时会自动使用。

命令行入口（等价于 `python mian.py`）：

```bash
douzero-hlddz
```

> 资源路径（`data.json`、`pics/`、`music/`、`weights/`、`baselines/`）已统一由
> [app_paths.py](app_paths.py) 以项目根目录解析成绝对路径，因此不再要求必须在项目根目录启动。

* 在原 [DouZero_For_Happy_DouDiZhu](https://github.com/tianqiraf/DouZero_For_HappyDouDiZhu) 的基础上加入了自动出牌，基于手牌自动叫牌，加倍，同时修改截屏方式为窗口区域截屏，游戏原窗口遮挡不影响游戏进行。

*   **请勿把游戏界面最小化，否则无法使用**

## 说明
*   欢乐斗地主使用 **窗口** 模式运行
*   **如果觉得这个项目有用，请给一个Star谢谢！**
*   **本项目仅供学习以及技术交流，请勿用于其它目的，否则后果自负。**

## 使用步骤（最新版本跳过此步骤）
1. ~~先使用 `debug_screenshot.py` 确认屏幕缩放比~~（新版已改为窗口区域截屏并强制 1440×810，因此系统缩放保持 100% 即可，无需再改缩放比）

2. 点击游戏中开始游戏后点击程序的 `自动开始`

3. 如果需要自动继续下一把，点击单局按钮，使其变为自动

## 自动叫牌/加倍原理

用DouZero自我博弈N局，对于随机到的每种手牌，随机生成若干种对手手牌，把该牌型和赢的局数扔进一个简单的全连接网络进行训练，得到手牌与胜率之间的关系，最后根据预期胜率，以一定阈值进行叫牌和加倍。

## 潜在Bug
*   有较低几率把王炸识别为不出

## 修复记录（v5.7 之后）

| # | 问题 | 影响 | 修复 |
|---|---|---|---|
| 1 | `douzero/dmc` 训练链与推理链不一致：`Env.step()` 不把 `position/action` 交给引擎；缓冲区按 old 编码（319/430、z=(5,162)）建立，learner 却是 `ResnetModel`（x=15、z=(40,54)） | 自对局整局永远 pass、永不结束；跑训练必然 shape 报错 | `Env` 传 `(position, action)`；按 `--model_type` 建缓冲区；resnet 保存被选动作的 `(40,54)` z；`learn()` 按编码决定是否拼接动作；设备自适应 CPU/GPU |
| 2 | `FileWriter.__init__` 调用被注释掉的 `gather_metadata()` | 训练一启动就 `NameError` | 恢复 `gather_metadata()`（git 元数据改为可选），并在 `close()` 释放日志句柄 |
| 3 | `search_actions()` 永远 `return None`，且 `game.py` 顶层导入 `search_utility` 形成循环导入 | 结果只能靠副作用传参；`import search_utility` 直接 `ImportError` | 返回 `path_list`；删除死代码；改为函数内延迟导入（两条导入路径都可用） |
| 4 | `BidModel` 把 `bid_weights.pkl` 装进 `net2`，`predict()` 却用从未加载权重的 `net` | 叫牌得分等于随机数 | 只保留与权重键名匹配的网络，删除失效 `predict()`，加载失败/不匹配时显式告警 |
| 5 | `game.py` 的 `bombs` 把王炸写成 `[30, 20]`，而生成器产出 `[20, 30]` | 王炸不计入 `bomb_num`，ADP 回报与炸弹特征被低估 | 修正字面量并统一按排序后的 key 比较 |
| 6 | 农民加倍阈值 `jiabei5/6` 先赋值、随后被 `jiabei7/8` 无条件覆盖 | 界面上两个输入框永远无效 | 删除死配置（数据/界面/代码三处），农民只保留一套阈值 |

**第二轮（健壮性）**

| # | 问题 | 修复 |
|---|---|---|
| 7 | 所有资源用相对路径（`data.json`、`./pics`、`./weights`、`music/*.wav`、`baselines/*.ckpt`），换工作目录就崩或静默退回随机权重 | 新增 [app_paths.py](app_paths.py) 统一以项目根解析绝对路径 |
| 8 | 阈值传参靠 `emit` + `sleep(0.1)` 跨线程握手，主线程忙时读到旧值 | 点击「手动/自动」时先在主线程落盘，worker 直接读文件 |
| 9 | 停止用 `QThread.terminate()` 无条件强杀 | 先置 `stop_sign/RunGame` 请求协作退出，1 秒未退出才强杀 |
| 10 | `mian.py` 复制了一份牌面映射/炸弹表；`initialBeishu` 未定义；`change_mode` 死属性；`debug_screenshot.py` 无窗口时崩；`DetermineColor` 演示图路径写错 | 统一从 `douzero.env.game` 导入常量，清掉死代码/潜在 NameError，补空值保护 |

回归测试：

```bash
pytest                                  # 18 个用例
python tests/test_regressions.py        # 6 个严重问题的专项
python tests/test_robustness.py         # 健壮性专项
python tests/dmc_smoke.py               # 真实跑一次 dmc.train()（CPU 约 10s）
```


## 鸣谢
*   本项目基于[DouZero](https://github.com/kwai/DouZero)  [DouZero_For_Happy_DouDiZhu](https://github.com/tianqiraf/DouZero_For_HappyDouDiZhu) 

## 其他

欢迎加入QQ群交流自动化相关：565142377  密码 douzero

也可以在原作者[DouZero_For_Happy_DouDiZhu](https://github.com/tianqiraf/DouZero_For_HappyDouDiZhu) 的QQ群交流


                                                       使用说明
                                                       
第一步：安装完python环境之后在pycharm上运行mian.py（游戏设置为窗口模式，不需要改分辨率和缩放比。

  注意：
  
  1.不要拉伸默认的窗口大小，一不小心拉伸窗口的朋友，请自行从网上找句柄工具将窗口恢复到默认的1440×810大小。
  
  2.游戏背景要使用默认的蓝色桌子，不然识别会有问题）。  
  
第二步：全屏进入新版QQ欢乐斗地主。

第三步：点击《斗地主合集》，再进入欢乐经典场或经典场（推荐经典场，因为欢乐经典场炸弹太多）。

第四步：选择相对应的场次。

第五步：进入到游戏的开始界面。

第六步：点击软件的手动或自动开始（优点：出牌的时候可以在手动和自动之间切换）。

最后感谢大家的理解和支持，一个人的力量有限，希望大家多多反馈BUG，确认后我会尽快处理。
