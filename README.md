# TE-Speed-MiniMaxH3-OSS

纯 Python 开源版 MiniMax H3 加速插件。原版 TE-Speed-MiniMaxH3 把核心逻辑编译成了
`nodes.pyd`（Cython），本包用等价的 Python 代码重新实现了同样的加速算法，
不依赖任何二进制文件，参数行为与原版一致，并额外提供 `cache_depth` 调优项。

## 安装

1. 把 `TE-Speed-MiniMaxH3-OSS` 文件夹放入 `ComfyUI\custom_nodes`。
2. 给 ComfyUI 的 MiniMax H3 模型文件打钩子补丁（只需一次）：

   ```
   python patch_model.py
   ```

   自动定位 ComfyUI（也可 `--comfy-ui <ComfyUI根目录>` 指定）。脚本会先备份
   原文件为 `model.py.te_speed.bak`，再做两处最小改动：
   - 新增 `MiniMaxH3Model._run_blocks(start, end)`（支持部分块区间执行）
   - 在 forward 的块循环处增加 `("block_loop", 0)` 钩子分支

   回退：`python patch_model.py --revert`
3. 重启 ComfyUI。

## 使用

工作流中把 `UNETLoader` 输出的 model 接入 `TE-Speed-MiniMaxH3 (OSS)` 节点，
再接到 `BasicScheduler` / `BasicGuider`。节点内部：

```
model -> TESpeedMiniMaxH3(OSS) -> BasicScheduler
                              \-> BasicGuider
```

示例工作流 `minimax_h3_TE_Speed加速高达45%(示例工作流).json` 可直接加载
（节点显示名不同但类型名 `TESpeedMiniMaxH3` 一致）。

## 参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| processing_control_value | 0.12 | 相邻两步 sigma 差小于该值时允许走缓存步（0 = 全部完整计算，结果与官流一致） |
| processing_percent_1 | 0.1 | 缓存窗口起点：前 10% 步始终完整计算 |
| processing_percent_2 | 0.9 | 缓存窗口终点：后 10% 步始终完整计算 |
| mcs | 2 | 最多连续缓存步数，超过后强制完整步，防止误差累积（0 = 关闭缓存） |
| device | auto | 缓存残差存放位置：auto/gpu 留在计算设备，cpu 省显存（有搬运开销） |
| cache_depth *(扩展)* | 0.75 | 缓存步中从缓存取用的尾部块占比：0.75 ≈ 原版 45% 提速；调低更稳画质，调高更快；0 = 关闭缓存（与 threshold=0、mcs=0 等效） |

## 加速原理

每个去噪步，插件通过 `("block_loop", 0)` 钩子接管 50 层 DiT 的块循环：

- **完整步 (FULL)**：跑全部块，并保存残差 `residual = h_full - h_warm`，
  其中 `h_warm` 是前 `(1-cache_depth)*block_count` 个"热身块"的输出；
- **缓存步 (CACHE)**：只重算热身块，再加上一步保存的残差。相邻步 sigma 差很小时，
  被缓存尾部块的贡献几乎不变，残差校正即可补偿漂移。

三步条件同时满足才允许走缓存步：位于调度窗口内、sigma 差小于阈值、连续缓存未超过
mcs。每轮采样的第一步强制完整步；CFG（正负条件）同一步的两次调用共享同一决策。
运行结束后控制台打印 `TE-Speed-MiniMaxH3(OSS): acceleration xx.x%` 统计。

默认参数下在参考工作流（30 步、8s 视频）中约提速 45%，与官方 pyd 版本一致。

## 与原版的差异

- 全部逻辑为 Python 源码，可审计、可修改；
- 新增 `cache_depth` 参数（原版为内部常量，行为按 0.75 还原）；
- 报错信息更明确：未打钩子补丁时会警告并以原速运行，不会静默失效；
- 采样中途无法注入统计回调，加速百分比在最后一步/下一轮开始时打印。
