# astro-histogram-stretch

用于 Codex 的天文图像显示拉伸技能。支持 FITS/TIFF，使用 PixInsight HistogramTransformation 文档中的裁剪、MTF 和动态范围扩展公式。

默认彩色流程：检查线性 RGB → 选取干净且面积充分的背景区域 → 背景 RGB 中位数对齐 → 在固定背景参考下进行线性配色 → 共享黑白点及 MTF 拉伸 → 检查输出。默认最终中间调参数为亮结构基础 m 的 2 倍，输出背景中位数限制在 0.1 以下。

## 安装

需要可运行 Python 和本地脚本的 Codex 环境，以及 Python 3.10 或更新版本。私有仓库需要 GitHub 访问权限。

将仓库克隆到 Codex 个人技能目录（默认位置如下；自定义 CODEX_HOME 时使用相应的 skills 目录）：

```bash
git clone https://github.com/xianghao-ma/astro-histogram-stretch.git ~/.codex/skills/astro-histogram-stretch
cd ~/.codex/skills/astro-histogram-stretch
python3 -m venv .venv
.venv/bin/python -m pip install -r scripts/requirements.txt
```

若目标目录已存在，先检查其内容，避免覆盖。Windows 用户使用对应的虚拟环境解释器路径。让 Codex 重新发现已安装技能后，提供数据绝对路径并调用：

> 使用 $astro-histogram-stretch 处理这份线性 FITS，生成预览和参数记录。使用 skill 目录下 .venv 的 Python。

## 执行范围

- `scripts/stretch.py`：图像检查、显式参数像素变换和输出校验。
- `scripts/adaptive_stretch.py`：离群候选、结构样本、共享黑白点、背景上限及默认 m×2 的自动拉伸。
- `references/color-balance.md`：背景选择、中性化、仅在线性文件上的通道增益与常数偏移、颜色合理性和验证规范。

背景区域选择和线性配色由 Codex 按规范执行，目前没有封装成一个全自动配色命令。单独调用 adaptive_stretch.py 只执行拉伸阶段。先保存并验证校色后的线性 RGB，再运行拉伸脚本。

## 直接运行脚本

将示例输入路径替换为实际文件。RGB 示例要求输入为三通道线性数据，已完成背景中性化与配色，且确有 PRIMARY 图像和 VALID 有效区扩展：

```bash
.venv/bin/python scripts/stretch.py inspect /path/to/image.fits
.venv/bin/python scripts/adaptive_stretch.py /path/to/linear_rgb.fits --hdu 0 --mask-hdu VALID --channel-axis 0 --outdir /path/to/new-output
.venv/bin/python -m unittest discover -s scripts -p 'test_*.py'
```

FITS 扩展与掩膜须按实际输入选择；详细参数见脚本 --help 和 SKILL.md。

## 数据与限制

不需要安装 PixInsight。仓库包含自编脚本和算法摘要，不包含 PixInsight 手册 PDF、观测数据、测试图像或本机 Python 环境。未与 PixInsight 原生输出逐像素验证完整兼容性。

拉伸输出是非线性显示产品，不保留原始测光通量语义。多波段代表色配色不等于自然真彩色或测光定标。XISF、图像配准和实测 PSF 匹配不属于随附拉伸脚本的功能。
