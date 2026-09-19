# 停车车辆 + 中文车牌识别系统（V10）

基于深度学习的停车场景车辆检测、跟踪与中文车牌识别系统。对视频中的车辆进行
YOLO 检测与 BoTSORT 多目标跟踪，判断车辆是否停放，并对其中的中文车牌进行
检测、透视矫正、双 OCR 识别与逐字投票聚合，最终输出标注视频、车辆截图归档
和可视化 HTML 报告。

> 主程序：`测试-5.py`（版本 V10）

## 功能特性

- **车辆检测与跟踪**：YOLOv8x + BoTSORT，支持跨帧 ReID 特征匹配（HSV 直方图）。
- **车牌检测**：独立车牌检测模型，支持全局检测与车辆 ROI 局部检测两级策略。
- **双 OCR 识别**（P23）：PaddleOCR（SVTR_LCNet）与 HyperLPR3 双引擎投票，结果一致时加分。
- **严格格式校验**（P22）：省简称、第二位字母、后缀字符集、新能源 8 位牌等多重过滤。
- **透视矫正**（P24）：对倾斜车牌做最小外接矩形 + `warpPerspective` 拉正。
- **旋转增强**（P25）：±8°/±15° 副本扩充识别样本。
- **逐字投票聚合**：带时间衰减、地区先验加权的字符级投票，产出最终车牌号。
- **停放判定**：基于运动历史中位数距离阈值，判定车辆是否停放并计时。
- **相似车牌目录合并**：后处理按编辑距离/省份易混组归并相似车牌。
- **HTML 报告**（P26）：自动生成可视化报告 `report.html`（内嵌 base64 图片）。
- **可视化 UI**：HUD 进度条、近期识别车牌 PIP 面板、四角括号框、自适应字体、
  车牌/颜色标签粘滞（F1~F4，避免高频闪烁）。

## 目录结构

```
.
├── 测试-5.py            # 主程序（V10）
├── botsort_custom.yaml  # BoTSORT 跟踪器配置（运行时自动生成）
├── report.html          # 生成的识别报告（HTML，内嵌图片）
├── vehicles/            # 车辆截图归档（按车牌号/UNKNOWN 分目录）
└── README.md
```

## 环境依赖

- Python 3.10（推荐虚拟环境 `plate310`）
- PyTorch（CUDA 可选，代码自动回退 CPU）
- Ultralytics（YOLO）
- PaddlePaddle + PaddleOCR
- OpenCV（`opencv-python`）
- NumPy、Pillow、tqdm
- HyperLPR3（可选，安装后启用双 OCR）

```bash
pip install ultralytics paddleocr paddlepaddle opencv-python numpy pillow tqdm
pip install hyperlpr3      # 可选，安装后启用双 OCR 投票
```

## 模型准备

程序内固定加载以下两个模型路径，**需先放置对应权重文件**：

| 用途 | 代码内路径 | 说明 |
|------|-----------|------|
| 车辆检测 | `/mnt/workspace/models/yolov8x.pt` | YOLOv8x 权重 |
| 车牌检测 | `/mnt/workspace/models/chinese_plate_yolov8.pt` | 中文车牌检测权重 |

> 若在 Windows / 其他机器运行，请将 `测试-5.py` 中这两处路径改为本机实际路径，
> 或将模型放到对应位置。

## 使用方法

```bash
# 完整处理（检测 + 跟踪 + 识别 + 报告）
python 测试-5.py -i 输入视频.mp4 -o 输出目录

# 只对已有 vehicles/ 目录做相似车牌合并 + 生成报告
python 测试-5.py --merge-only -o 输出目录

# 只基于已有 vehicles/ 目录重新生成 HTML 报告
python 测试-5.py --report-only -o 输出目录
```

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-i, --input` | 无 | 输入视频路径（`--merge-only`/`--report-only` 时可不传） |
| `-o, --output` | `./output4` | 输出目录 |
| `--stride` | `1` | 抽帧步长 |
| `--out-height` | `1080` | 输出视频高度 |
| `--no-pip` | 关 | 关闭近期车牌 PIP 面板 |
| `--no-bracket` | 关 | 关闭四角括号框 |
| `--no-sr` | 关 | 关闭停放车辆超分识别 |
| `--no-dual-ocr` | 关 | 关闭双 OCR，仅用 PaddleOCR |
| `--no-perspective` | 关 | 关闭透视矫正 |
| `--no-rotation` | 关 | 关闭旋转增强 |
| `--merge-only` | 关 | 仅执行相似车牌目录合并 |
| `--report-only` | 关 | 仅生成 HTML 报告 |

## 输出说明

处理完成后，输出目录下会生成：

- `result.mp4`：带检测框、车牌标签、HUD 与 PIP 面板的标注视频。
- `vehicles/`：按车牌号归档的车辆截图（`车牌号/` 与 `UNKNOWN_xxx/`）。
- `report.html`：可视化识别报告，展示每辆车车牌、图片数与代表图。
- `botsort_custom.yaml`：本次运行使用的跟踪器配置。

## 版本说明（V10）

- **F1~F4**：修复输出视频「UI 高频闪烁导致看似跳帧」问题（标签粘滞、颜色粘滞、秒数取整、PIP 不滚动）。
- **P22**：严格车牌格式验证，淘汰垃圾串。
- **P23**：HyperLPR3 双 OCR 投票。
- **P24**：车牌透视矫正。
- **P25**：±8°/±15° 角度增强副本。
- **P26**：HTML 报告自动生成。

> 更早版本（P1~P21、V1~V7）改动已合入当前代码，逻辑保持一致。

## 许可

项目仅供学习与研究使用。车牌数据请遵守当地法律法规，勿用于非法用途。
