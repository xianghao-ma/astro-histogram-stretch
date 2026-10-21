---
name: astro-histogram-stretch
description: 用 PixInsight HistogramTransformation 文档中的裁切、MTF 和动态范围扩展算法，通过独立 Python 脚本拉伸 FITS/TIFF 天文图像。适用于显示拉伸、彩色图线性背景中性化与配色、黑白点和中间调调整；不用于测光校准、直方图均衡化或生成替代天体图像。
---

# 天文图像直方图拉伸

用随附脚本执行确定性的像素变换。算法、输入尺度和参数策略见
[references/algorithm.md](references/algorithm.md)。首次处理或解释算法时先读该文件。

## 输入与执行

1. 先运行 `scripts/stretch.py inspect INPUT`，检查图像扩展、形状、单位、
   有限像素范围和已有非线性处理标记。FITS 有多个图像扩展时选择实际科学图像，
   不把 ERR、DQ、WHT 当作科学图像；必要时问用户选择哪一幅。
2. 明确输入是线性数据还是已拉伸图。三维 FITS 必须确认是 RGB，并显式给出
   `--channel-axis`；不要把光谱立方体自动解释为 RGB。若有有效区掩膜，
   检查其语义再使用 `--mask-hdu`（有限且大于零表示有效）；不能直接把 DQ
   当作这种掩膜。
3. 彩色图默认先完成[线性配色流程](references/color-balance.md)：选取面积充足、
   排除天体及可见弥散光的固定背景区，先对齐 RGB 背景中位数，再调整正线性增益。
   配色只允许逐通道乘常数和加减常数，使用 `I'_c = a_c × (I_c − b_c) + B`，
   保持固定背景区的 RGB 中位数均为共同值 B（默认 0）。保存校色后的线性文件，
   再执行下述共享拉伸；不得在已拉伸图上改颜色。检查目标颜色的合理性，
   不强制所有星系或透镜弧同色，也不把代表色显示当作测光校准。
   灰度图跳过配色；用户明确要求保留既有配色时遵从用户要求。
4. 默认执行 `scripts/adaptive_stretch.py` 的自适应流程，详见
   [references/adaptive.md](references/adaptive.md)：
   - 先用稳健背景统计和空间邻域筛出暗端、亮端离群候选；不要按固定百分位
     截掉亮星或星系核心。保留候选掩膜，候选不等于已确认坏点。
   - 从非离群样本中选择背景以上 5σ、至少 9 个相连像素的较亮结构。
     在此拉伸阶段，背景仅从中间调选参样本中排除，不再次扣背景或删除背景像素。
   - RGB 必须共用一个黑点、一个白点和一个 MTF 中间调参数。
     不把各通道的背景减 3σ 分别作为黑点。新请求明确要求分通道时才覆盖此约定。
   - 以 RGB 平均强度估计的背景减 3σ 为起始黑点；非离群样本的共同最大值为白点。
     每次调整黑点后，重新计算裁切重标度后亮结构平均强度的中位数作为 `m`，
     让此输入位置映射至 0.5。这里的中位数不是旧版 `--target-median` 输出目标。
   - 用固定背景掩膜测量输出 RGB 平均强度中位数，默认限制在 **0.1 以下**。
     若初始结果超限，提高共享黑点并重算 `m`，通常求到 0.099 留出余量；
     同时检查保存的浮点图与 PNG。背景本来较暗时无需强行提亮至 0.099。
   - 黑点求解完成后，固定黑白点，默认将基础 `m` **乘以 2** 得到最终 MTF 参数，
     使图像更暗；不在放大后重求黑点，不累计乘上一次试验的倍率。
     可用 `--midtones-factor` 明确覆盖，设为 1 恢复基础曲线。最终 m 必须在 (0,1)；
     超界时报告失败，不静默截断。最终浮点图与 PNG 仍须满足背景上限。
   - 此过程允许裁切背景尾部，不能再声称“只剔除离群点”或“背景完全未截断”。
     分别报告 RGB 裁切比例、背景中位数、实际黑白点和 `m`；不同通道裁切比例可不同。
   - 灰度图使用同一拉伸流程。RGB 在完成前述线性配色后固定通道增益；
     拉伸阶段不再独立改动通道颜色。
     若没有可靠背景、结构掩膜不足或求解失败，检查掩膜并改用用户指定参数，
     不把失败静默替换成全图固定目标中位数。
   用户明确指定参数时，使用 `scripts/stretch.py apply`；其 MTF 数值核心保持不变。
5. 每次使用新的输出目录。检查预览、直方图和 JSON；留意暗弱结构、饱和核心、
   噪声和色偏。调整时从原始输入重跑，避免无意中叠加拉伸。
6. 交付拉伸图像、PNG 预览和参数 JSON 的链接，简述参数与裁切结果。
   彩色图还交付校色后的线性文件、背景掩膜/区域图及线性系数记录。
   拉伸 FITS/TIFF 是非线性显示产品，不能作为保留原始通量的测光数据。

## 命令

以下 `SKILL_DIR` 表示本文件所在目录，`PYTHON` 表示已装依赖的解释器；
实际调用时替换为绝对路径。脚本依赖 `scripts/requirements.txt`，优先复用
已有环境；缺依赖时在工作目录建虚拟环境，不修改系统 Python。

彩色示例中的 RGB 输入须先完成上述背景中性化与线性配色。
`adaptive_stretch.py` 是拉伸执行器，不会自动选择无弥散光背景区或完成配色；
使用 skill 时必须先执行并验证配色步骤，不能用直接调用脚本代替完整流程。

```bash
PYTHON SKILL_DIR/scripts/stretch.py inspect /absolute/path/image.fits

PYTHON SKILL_DIR/scripts/adaptive_stretch.py /absolute/path/image.fits \
  --hdu SCI --mask-hdu VALID --background-cap 0.1 \
  --outdir /absolute/path/result

PYTHON SKILL_DIR/scripts/adaptive_stretch.py /absolute/path/rgb.fits \
  --hdu 0 --mask-hdu VALID --channel-axis 0 \
  --outdir /absolute/path/rgb-result

PYTHON SKILL_DIR/scripts/stretch.py apply /absolute/path/rgb.tiff \
  --midtones 0.2 --outdir /absolute/path/rgb-result

PYTHON SKILL_DIR/scripts/stretch.py apply /absolute/path/rgb.fits \
  --hdu 0 --channel-axis 0 --input-range -0.1 50 \
  --shadows 0 --highlights 1 --midtones 0.2 \
  --outdir /absolute/path/rgb-result
```

仅在输入确有相应扩展时使用示例中的 SCI / VALID。参数和格式完整说明用
`scripts/adaptive_stretch.py --help` 或 `scripts/stretch.py apply --help` 查看。回归检查：
`PYTHON -m unittest discover -s SKILL_DIR/scripts -p 'test_*.py'`。

## 输出约定与边界

- 不覆盖输入或已有输出目录。保留有限负值到显式归一化阶段；NaN、Inf 和
  掩膜无效样本不进入统计，浮点结果中写为 NaN，PNG 中显示为透明。
- 输出与输入保持同类格式：FITS 为 float64 并保留所选图像 HDU 的 WCS；
  TIFF 为 float32。另有 `preview.png`、`diagnostics.png`、`parameters.json`。
  TIFF 不携带 FITS WCS，PNG 为量化预览。自适应模式另存
  `adaptive_parameters.json` 与 `selection_masks.npz`，记录选择依据与固定评估掩膜。
- FITS 仅输出变换后的科学图像及 VALID 掩膜，不复制未经传播的误差数组。
  BUNIT 改为无量纲，历史记录标明显示用途。图像网格不变。
- 默认 PNG/诊断图按 FITS 的 lower 原点或 TIFF 的 upper 原点显示，
  不改变数据文件的数组顺序；可用 `--origin` 覆盖显示方向。
- 本实现不支持 XISF、alpha、多页 TIFF、复数数据、依赖外部表的 WCS 或图像配准。三维数据仅支持
  已确认的三通道 RGB。一次调用只执行一组共享或逐通道参数，
  不复刻 PixInsight 全部界面功能或逐通道后再叠加 RGB/K 的过程实例。
- 核心公式已按文档实现；未与本机 PixInsight 原生输出逐像素交叉验证时，
  不宣称已实现完整原生兼容。
