# =========================================================
# 中国停车车辆 + 中文车牌识别系统  (V10  测试-5.py)
#
# 基于 测试-4.py (V9), 修复输出视频「UI 高频闪烁导致看似跳帧」问题:
#
#   F1  标签粘滞: 一旦试探性识别过, 后续 OCR 失败不退回 ID-XX
#   F2  颜色粘滞: sticky 记录最高级状态, 不每帧从空判
#   F3  秒数取整: ⏱ 已停 12s (不再每 0.1s 抖动)
#   F4  PIP 不滚动: 满了就不再 push, 避免左下角整块刷新
#
# 识别逻辑 (P1~P26) 完全保留, 跟 V9 一致
#
# 上代 V9 改动 (背景说明):
#   P22  严格车牌格式验证 (淘汰垃圾串 如 皖O0700S8)
#   P23  hyperlpr3 双 OCR 投票
#   P24  车牌透视矫正
#   P25  ±8/±15° 角度增强副本
#   P26  HTML 报告自动生成
#
# 全部 P1~P21 + V1~V7 + F1~F4 保留
#
# 推荐虚拟环境: plate310
# 补装依赖:
#   pip install hyperlpr3      # 可选, 装了会启用双 OCR
#   pip install tqdm Pillow    # 通常已有
# =========================================================

import os
import gc
import re
import cv2
import sys
import math
import time
import shutil
import uuid
import json
import base64
import torch
import argparse
import traceback
import numpy as np

from tqdm import tqdm
from collections import defaultdict

from ultralytics import YOLO
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont

# P23: hyperlpr3 可选, 没装就降级用单 OCR
HYPERLPR_AVAILABLE = False
try:
    import hyperlpr3 as lpr3
    HYPERLPR_AVAILABLE = True
except Exception:
    pass


# =========================================================
# ENV
# =========================================================
os.environ["FLAGS_allocator_strategy"] = "auto_growth"
os.environ["FLAGS_fraction_of_gpu_memory_to_use"] = "0.30"
os.environ["FLAGS_eager_delete_tensor_gb"] = "0.0"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


# =========================================================
# CONFIG
# =========================================================
CONFIG = {
    # ---- VEHICLE ----
    "vehicle_conf": 0.35,
    "detect_size": 1280,
    "min_vehicle_area": 80 * 80,

    # ---- PLATE ----
    "plate_conf": 0.20,
    "plate_imgsz": 1920,
    "plate_detect_interval": 5,

    # ---- OCR ----
    "ocr_interval": 10,
    "ocr_interval_parked": 30,
    "vote_min_count": 3,
    "min_ocr_score": 0.30,                # P22: 0.25 -> 0.30 提高
    "max_vote_history": 60,
    "vote_decay": 0.95,

    # ---- PARK ----
    "park_frame_threshold": 50,
    "park_distance_threshold": 5,

    # ---- SAVE ----
    "save_interval": 30,
    "angle_min_iou_change": 0.20,
    "angle_min_feat_change": 0.08,
    "max_angle_per_vehicle": 12,
    "max_files_per_dir": 16,

    # ---- PLATE SIZE ----
    "min_plate_w": 16,
    "min_plate_h": 8,

    # ---- GPU ----
    "use_fp16": True,
    "cuda_clear_interval": 100,

    # ---- ReID ----
    "reid_threshold": 0.78,
    "reid_threshold_close": 0.70,
    "reid_close_gap": 30,
    "reid_max_gap": 120,
    "reid_max_dist_ratio": 0.30,

    "frame_stride": 1,
    "out_height": 1080,
    "video_codec": "avc1",

    # ---- UI ----
    "ui_adaptive_font": True,
    "ui_pip_max": 6,
    "ui_pip_enabled": True,
    "ui_hud_enabled": True,
    "ui_corner_bracket": True,
    "ui_show_park_seconds": True,

    # ---- 车牌聚合 ----
    "plate_suffix_match_len": 5,
    "plate_edit_distance_threshold": 2,
    "province_confusion_groups": [
        set("湘皖鄂豫"),
        set("沪浙苏赣"),
        set("京津冀晋"),
        set("粤桂闽琼"),
        set("川渝云贵"),
        set("辽吉黑"),
        set("陕甘宁青"),
        set("鲁皖"),
    ],

    # ---- 超分 ----
    "sr_enable_for_parked": True,
    "sr_max_attempts_per_vehicle": 5,

    # ---- 地区先验 ----
    "region_prior_province": "湘",
    "region_prior_weight": 1.5,

    # ---- P22 严格验证 ----
    "strict_format": True,
    "min_digits_in_suffix": 2,            # 后 5 位至少含 2 个数字
    "blacklist_second_char": set("OQ"),   # 第二位禁止字符 (O 常被错认)

    # ---- P23 双 OCR ----
    "dual_ocr_enabled": True,
    "dual_ocr_agree_bonus": 1.5,          # 两个 OCR 一致时分数加成

    # ---- P24 透视矫正 ----
    "perspective_enabled": True,
    "perspective_min_skew_deg": 3,        # 倾斜超过 N° 才矫正

    # ---- P25 旋转增强 ----
    "rotation_aug_enabled": True,
    "rotation_aug_angles": [-15, -8, 8, 15],

    # ---- P26 HTML 报告 ----
    "report_enabled": True,
}


# =========================================================
# PLATE
# =========================================================
PROVINCE_CHARS = "京沪津渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼"
SPECIAL_TAIL = "使领警学港澳"
LETTER_DIGIT = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
PLATE_CHARS = PROVINCE_CHARS + SPECIAL_TAIL + LETTER_DIGIT
PLATE_REGEX = re.compile(r'[一-龥][A-Z][A-Z0-9]{5,6}')


# =========================================================
# 字体
# =========================================================
_FONT_PATH = None
_FONT_CACHE = {}


def _detect_font_path():
    global _FONT_PATH
    if _FONT_PATH is not None:
        return _FONT_PATH
    for p in [
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]:
        if os.path.exists(p):
            _FONT_PATH = p; return p
    _FONT_PATH = ""; return ""


def get_font(size):
    size = max(10, int(size))
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    p = _detect_font_path()
    try:
        f = ImageFont.truetype(p, size) if p else ImageFont.load_default()
    except Exception:
        f = ImageFont.load_default()
    _FONT_CACHE[size] = f
    return f


# =========================================================
# 几何 / 特征
# =========================================================
def center_distance(a, b):
    ax = (a[0] + a[2]) / 2; ay = (a[1] + a[3]) / 2
    bx = (b[0] + b[2]) / 2; by = (b[1] + b[3]) / 2
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def box_iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    u = area_a + area_b - inter
    return inter / u if u > 0 else 0.0


def calc_feature(image):
    image = cv2.resize(image, (224, 224))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    v = cv2.calcHist([hsv], [2], None, [256], [0, 256])
    feat = np.concatenate([h.flatten(), s.flatten(), v.flatten()])
    return cv2.normalize(feat, None).flatten().astype(np.float32)


def feature_similarity(f1, f2):
    return float(cv2.compareHist(
        f1.astype(np.float32), f2.astype(np.float32),
        cv2.HISTCMP_CORREL))


def image_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def image_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv[..., 2].mean()


def is_good_plate_crop(crop):
    if crop is None or crop.size == 0:
        return False
    h, w = crop.shape[:2]
    if w < CONFIG["min_plate_w"] or h < CONFIG["min_plate_h"]:
        return False
    if image_sharpness(crop) < 25:
        return False
    b = image_brightness(crop)
    if b < 30 or b > 235:
        return False
    return True


# =========================================================
# 车牌相似度
# =========================================================
def _edit_distance(a, b):
    n, m = len(a), len(b)
    if abs(n - m) > 4:
        return max(n, m)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]; dp[0] = i
        for j in range(1, m + 1):
            tmp = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = tmp
    return dp[m]


def _province_confusable(p1, p2):
    if p1 == p2:
        return True
    for g in CONFIG["province_confusion_groups"]:
        if p1 in g and p2 in g:
            return True
    return False


def plate_similar(p1, p2):
    if not p1 or not p2:
        return False
    if p1 == p2:
        return True
    if abs(len(p1) - len(p2)) > 1:
        return False
    L = CONFIG["plate_suffix_match_len"]
    if len(p1) >= L + 1 and len(p2) >= L + 1:
        if p1[-L:] == p2[-L:] and _province_confusable(p1[0], p2[0]):
            return True
    if _province_confusable(p1[0], p2[0]):
        if _edit_distance(p1[1:], p2[1:]) <= CONFIG["plate_edit_distance_threshold"]:
            return True
    return False


# =========================================================
# 绘图原语
# =========================================================
def pil_draw_text(img_bgr, text, xy, color_bgr, font, fill_alpha=255):
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil, "RGBA")
    draw.text(xy, text, font=font,
              fill=(color_bgr[2], color_bgr[1], color_bgr[0], fill_alpha))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def overlay_rect(img, p1, p2, color_bgr, alpha):
    overlay = img.copy()
    cv2.rectangle(overlay, p1, p2, color_bgr, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    return img


def draw_corner_bracket(img, x1, y1, x2, y2, color, length, thickness):
    cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)
    cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)
    cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)
    cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)


# =========================================================
# BoTSORT
# =========================================================
def write_botsort_yaml(path):
    content = (
        "tracker_type: botsort\n"
        "track_high_thresh: 0.5\n"
        "track_low_thresh: 0.1\n"
        "new_track_thresh: 0.6\n"
        "track_buffer: 120\n"
        "match_thresh: 0.85\n"
        "fuse_score: True\n"
        "gmc_method: sparseOptFlow\n"
        "proximity_thresh: 0.5\n"
        "appearance_thresh: 0.25\n"
        "with_reid: False\n"
    )
    with open(path, "w") as f:
        f.write(content)


# =========================================================
# P24: 透视矫正
# =========================================================
def perspective_correct(plate_crop):
    """
    用最小外接矩形 + warpPerspective 把斜车牌拉正
    """
    try:
        h, w = plate_crop.shape[:2]
        if h < 10 or w < 20:
            return plate_crop

        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        # 自适应二值化突出字符
        bw = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 25, 10
        )
        # 找最大轮廓
        contours, _ = cv2.findContours(
            bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return plate_crop

        # 用所有点合并求最小外接矩形
        all_pts = np.vstack([c.reshape(-1, 2) for c in contours
                             if cv2.contourArea(c) > 5])
        if len(all_pts) < 4:
            return plate_crop
        rect = cv2.minAreaRect(all_pts)
        (cx, cy), (rw, rh), ang = rect
        if rw < 10 or rh < 5:
            return plate_crop

        # 让长边水平
        if rw < rh:
            rw, rh = rh, rw
            ang = ang + 90

        # 倾斜过小不矫正
        if abs(ang) < CONFIG["perspective_min_skew_deg"]:
            return plate_crop

        box = cv2.boxPoints(rect)
        # 按 (左上, 右上, 右下, 左下) 排序
        box = sorted(box, key=lambda p: p[1])
        top = sorted(box[:2], key=lambda p: p[0])
        bot = sorted(box[2:], key=lambda p: p[0])
        src = np.array([top[0], top[1], bot[1], bot[0]], dtype=np.float32)

        dst_w = int(max(rw, 100))
        dst_h = int(max(rh, 30))
        dst = np.array(
            [[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]],
            dtype=np.float32,
        )
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(plate_crop, M, (dst_w, dst_h))
        return warped
    except Exception:
        return plate_crop


# =========================================================
# P25: 旋转增强
# =========================================================
def rotation_variants(plate_crop):
    if not CONFIG["rotation_aug_enabled"]:
        return []
    out = []
    h, w = plate_crop.shape[:2]
    if h < 10 or w < 20:
        return out
    center = (w / 2, h / 2)
    for ang in CONFIG["rotation_aug_angles"]:
        M = cv2.getRotationMatrix2D(center, ang, 1.0)
        # 计算旋转后画布
        cos = abs(M[0, 0]); sin = abs(M[0, 1])
        nw = int((h * sin) + (w * cos))
        nh = int((h * cos) + (w * sin))
        M[0, 2] += (nw / 2) - center[0]
        M[1, 2] += (nh / 2) - center[1]
        try:
            rotated = cv2.warpAffine(
                plate_crop, M, (nw, nh),
                borderValue=(127, 127, 127))
            out.append(rotated)
        except Exception:
            pass
    return out


# =========================================================
# Vehicle
# =========================================================
class Vehicle:

    def __init__(self, uid):
        self.uid = uid
        self.track_ids = set()
        self.last_box = None
        self.last_seen_frame = -1

        self.char_table = defaultdict(lambda: defaultdict(float))
        self.vote_count = 0
        self.plate_votes = []
        self.best_plate_crop = None       # P26 用于报告

        self.final_plate = None
        self.confirmed = False

        self.last_ocr_frame = -999
        self.last_save_frame = -999

        self.stationary_frames = 0
        self.is_parked = False
        self.park_start_frame = -1
        self.motion_history = []

        self.feature = None
        self.feature_count = 0

        self.saved_boxes = []
        self.saved_features = []
        self.saved_count = 0

        self.sr_attempts = 0

        # F1/F2: 粘滞渲染状态 - 一旦升级就不退回
        # 0 = ID-XX 移动绿, 1 = 已停红, 2 = 试探性识别青, 3 = 已确认橙
        self.sticky_render_state = 0
        self.last_plate_for_label = None     # 上次显示过的车牌 (即使现在 None)


# =========================================================
# 主系统
# =========================================================
class PlateRecognitionSystem:

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.vehicle_dir = os.path.join(output_dir, "vehicles")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.vehicle_dir, exist_ok=True)

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print("=" * 60)
        print("DEVICE:", self.device)
        if torch.cuda.is_available():
            print("GPU :", torch.cuda.get_device_name(0))
            print("CUDA:", torch.version.cuda)
            torch.cuda.empty_cache()
        print(f"HyperLPR3 可用: {HYPERLPR_AVAILABLE}")
        print("=" * 60)

        self.botsort_yaml = os.path.join(output_dir, "botsort_custom.yaml")
        write_botsort_yaml(self.botsort_yaml)

        print("Loading vehicle model...")
        self.vehicle_model = YOLO("/mnt/workspace/models/yolov8x.pt")
        print("Loading plate model...")
        self.plate_model = YOLO("/mnt/workspace/models/chinese_plate_yolov8.pt")

        print("Loading PaddleOCR (GPU)...")
        try:
            self.ocr = PaddleOCR(
                use_angle_cls=True, lang='ch', use_gpu=True, show_log=False,
                det_db_box_thresh=0.2, det_db_thresh=0.2,
                rec_algorithm='SVTR_LCNet',
                rec_batch_num=4, cpu_threads=4, enable_mkldnn=False,
            )
            print(" PaddleOCR on GPU ok")
        except Exception as e:
            print(" PaddleOCR GPU failed, fallback CPU:", e)
            self.ocr = PaddleOCR(
                use_angle_cls=True, lang='ch', use_gpu=False, show_log=False,
                det_db_box_thresh=0.2, det_db_thresh=0.2,
                rec_algorithm='SVTR_LCNet',
                rec_batch_num=1, cpu_threads=4, enable_mkldnn=True,
            )

        # P23: hyperlpr3
        self.lpr3 = None
        if HYPERLPR_AVAILABLE and CONFIG["dual_ocr_enabled"]:
            try:
                self.lpr3 = lpr3.LicensePlateCatcher()
                print(" hyperlpr3 on ok")
            except Exception as e:
                print(" hyperlpr3 init failed:", e)
                self.lpr3 = None

        self.vehicles = {}
        self.track_to_uid = {}
        self.recognized_plates = set()

        self.frame_w = 0; self.frame_h = 0
        self.video_fps = 25.0

        self.recent_plates = []
        self.fps_proc = 0.0
        self._ema_dt = 0.04

        self.cuda_warmup()

    def _scale(self, val):
        return max(1, int(val * self.frame_h / 1080))

    def cuda_warmup(self):
        if not torch.cuda.is_available():
            return
        dummy = torch.zeros((1, 3, 640, 640), device=self.device)
        for _ in range(3):
            _ = dummy * 2.0
        torch.cuda.synchronize()

    # =====================================================
    # P22: 严格车牌格式
    # =====================================================
    def soft_validate_plate(self, plate):
        if plate is None:
            return None
        if len(plate) not in (7, 8):
            return None
        if plate[0] not in PROVINCE_CHARS:
            return None

        # 第 2 位必须字母, 且不能在黑名单
        if not plate[1].isalpha():
            d2l = {"0": "O", "1": "I", "8": "B", "5": "S", "2": "Z"}
            if plate[1] in d2l:
                plate = plate[0] + d2l[plate[1]] + plate[2:]
            else:
                return None
        if CONFIG["strict_format"]:
            if plate[1] in CONFIG["blacklist_second_char"]:
                return None

        # 后缀字符全合法
        for c in plate[2:]:
            if c not in LETTER_DIGIT:
                return None

        # P22 后缀至少含 N 个数字
        if CONFIG["strict_format"]:
            digit_count = sum(1 for c in plate[2:] if c.isdigit())
            if digit_count < CONFIG["min_digits_in_suffix"]:
                return None

            # 8 位牌第 3 位通常是 D/F/A 等新能源前缀
            if len(plate) == 8:
                if plate[2] not in "DFA0123456789":
                    return None

        return plate

    # -----------------------------------------------------
    def preprocess_variants(self, plate_crop):
        try:
            h, w = plate_crop.shape[:2]
            scale = max(2, int(120 / max(h, 1)))
            big = cv2.resize(plate_crop, (w * scale, h * scale),
                             interpolation=cv2.INTER_CUBIC)
            big = cv2.detailEnhance(big, sigma_s=10, sigma_r=0.15)
            gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            gray_eq = clahe.apply(gray)
            gray_eq = cv2.fastNlMeansDenoising(gray_eq, h=10)
            sharpen_k = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpen = cv2.filter2D(gray_eq, -1, sharpen_k)
            binary = cv2.adaptiveThreshold(
                sharpen, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
            return [
                cv2.cvtColor(big, cv2.COLOR_BGR2RGB),
                cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2RGB),
                cv2.cvtColor(sharpen, cv2.COLOR_GRAY2RGB),
                cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB),
            ]
        except Exception as e:
            print("PREPROCESS ERR:", e)
            return [plate_crop]

    def super_resolve(self, crop):
        h, w = crop.shape[:2]
        if h == 0 or w == 0:
            return crop
        big = cv2.resize(crop, (w * 6, h * 6),
                         interpolation=cv2.INTER_LANCZOS4)
        big = cv2.bilateralFilter(big, 9, 75, 75)
        big = cv2.detailEnhance(big, sigma_s=14, sigma_r=0.2)
        return big

    # =====================================================
    # OCR 子调用: PaddleOCR
    # =====================================================
    def _paddle_ocr(self, img):
        try:
            results = self.ocr.ocr(img, cls=True)
        except Exception:
            return None, 0
        if not results:
            return None, 0
        best_text, best_score = None, 0
        for line in results:
            if line is None:
                continue
            for item in line:
                try:
                    text = item[1][0]; score = float(item[1][1])
                    if score < CONFIG["min_ocr_score"]:
                        continue
                    text = text.upper().replace(" ", "")
                    filtered = "".join(c for c in text if c in PLATE_CHARS)
                    m = PLATE_REGEX.search(filtered)
                    if not m:
                        continue
                    plate = self.soft_validate_plate(m.group())
                    if plate is None:
                        continue
                    if score > best_score:
                        best_text, best_score = plate, score
                except Exception:
                    pass
        return best_text, best_score

    # =====================================================
    # OCR 子调用: HyperLPR3
    # =====================================================
    def _hyperlpr_ocr(self, img_bgr):
        if self.lpr3 is None:
            return None, 0
        try:
            results = self.lpr3(img_bgr)
        except Exception:
            return None, 0
        if not results:
            return None, 0
        best_text, best_score = None, 0
        for r in results:
            try:
                # hyperlpr3 返回 (plate_text, score, plate_type, [box])
                if isinstance(r, (list, tuple)) and len(r) >= 2:
                    text = str(r[0]); score = float(r[1])
                else:
                    continue
                if score < CONFIG["min_ocr_score"]:
                    continue
                text = text.upper().replace(" ", "").replace("·", "")
                filtered = "".join(c for c in text if c in PLATE_CHARS)
                m = PLATE_REGEX.search(filtered)
                if not m:
                    continue
                plate = self.soft_validate_plate(m.group())
                if plate is None:
                    continue
                if score > best_score:
                    best_text, best_score = plate, score
            except Exception:
                pass
        return best_text, best_score

    # =====================================================
    # P23 + P24 + P25: 主识别入口
    # =====================================================
    def recognize_plate(self, plate_crop, allow_sr=False):
        try:
            if plate_crop is None or plate_crop.size == 0:
                return None, 0
            if not is_good_plate_crop(plate_crop):
                return None, 0

            # P24: 先尝试透视矫正
            crops_to_test = [plate_crop]
            if CONFIG["perspective_enabled"]:
                warped = perspective_correct(plate_crop)
                if warped is not plate_crop:
                    crops_to_test.append(warped)

            # P25: 加旋转副本
            if CONFIG["rotation_aug_enabled"]:
                crops_to_test.extend(rotation_variants(plate_crop))

            best_text_paddle, best_score_paddle = None, 0
            best_text_lpr, best_score_lpr = None, 0

            # P23: 两路 OCR 各自跑所有 crop, 选最高分
            for base in crops_to_test:
                # ---- PaddleOCR (吃多种预处理) ----
                candidates = self.preprocess_variants(base)
                if allow_sr:
                    sr = self.super_resolve(base)
                    candidates = self.preprocess_variants(sr) + candidates
                for img in candidates:
                    img = np.ascontiguousarray(img)
                    t, s = self._paddle_ocr(img)
                    if t and s > best_score_paddle:
                        best_text_paddle, best_score_paddle = t, s

                # ---- HyperLPR3 (直接吃原 RGB) ----
                if self.lpr3 is not None:
                    t, s = self._hyperlpr_ocr(base)
                    if t and s > best_score_lpr:
                        best_text_lpr, best_score_lpr = t, s
                    if allow_sr:
                        sr = self.super_resolve(base)
                        t, s = self._hyperlpr_ocr(sr)
                        if t and s > best_score_lpr:
                            best_text_lpr, best_score_lpr = t, s

            # ---- 融合两路结果 ----
            if best_text_paddle and best_text_lpr:
                if best_text_paddle == best_text_lpr:
                    # 双 OCR 一致, 高度可信
                    return best_text_paddle, min(
                        1.0,
                        max(best_score_paddle, best_score_lpr)
                        * CONFIG["dual_ocr_agree_bonus"]
                    )
                # 不一致, 各打八折防止单边过度自信
                if best_score_paddle * 0.8 > best_score_lpr * 0.8:
                    return best_text_paddle, best_score_paddle * 0.8
                else:
                    return best_text_lpr, best_score_lpr * 0.8
            if best_text_paddle:
                return best_text_paddle, best_score_paddle
            if best_text_lpr:
                return best_text_lpr, best_score_lpr
            return None, 0

        except Exception as e:
            print("OCR ERR:", e)
            return None, 0

    # -----------------------------------------------------
    def get_or_create_vehicle(self, track_id, crop, current_box, frame_idx):
        if track_id in self.track_to_uid:
            v = self.vehicles[self.track_to_uid[track_id]]
            v.last_seen_frame = frame_idx
            return v

        new_feat = calc_feature(crop)
        cx = (current_box[0] + current_box[2]) / 2
        cy = (current_box[1] + current_box[3]) / 2
        max_dist = self.frame_w * CONFIG["reid_max_dist_ratio"]

        best_v = None; best_score = 0.0; best_gap = 9999
        for v in self.vehicles.values():
            if v.feature is None or track_id in v.track_ids:
                continue
            gap = frame_idx - v.last_seen_frame
            if gap > CONFIG["reid_max_gap"]:
                continue
            s = feature_similarity(v.feature, new_feat)
            if v.last_box is not None:
                vx = (v.last_box[0] + v.last_box[2]) / 2
                vy = (v.last_box[1] + v.last_box[3]) / 2
                d = math.sqrt((vx - cx) ** 2 + (vy - cy) ** 2)
                if d > max_dist:
                    s *= 0.5
            if s > best_score:
                best_score = s; best_v = v; best_gap = gap

        thr = (CONFIG["reid_threshold_close"]
               if best_gap <= CONFIG["reid_close_gap"]
               else CONFIG["reid_threshold"])
        if best_v is not None and best_score >= thr:
            best_v.track_ids.add(track_id)
            best_v.last_seen_frame = frame_idx
            self.track_to_uid[track_id] = best_v.uid
            return best_v

        uid_ = uuid.uuid4().hex[:8]
        v = Vehicle(uid_)
        v.feature = new_feat; v.feature_count = 1
        v.track_ids.add(track_id); v.last_seen_frame = frame_idx
        self.vehicles[uid_] = v
        self.track_to_uid[track_id] = uid_
        return v

    # -----------------------------------------------------
    def migrate_dir(self, old_name, new_name):
        if old_name == new_name:
            return
        old_dir = os.path.join(self.vehicle_dir, old_name)
        new_dir = os.path.join(self.vehicle_dir, new_name)
        if not os.path.isdir(old_dir):
            return
        if not os.path.isdir(new_dir):
            try:
                os.rename(old_dir, new_dir)
                print(f"RENAME: {old_name} -> {new_name}")
                return
            except Exception as e:
                print("RENAME ERR:", e); return
        try:
            for f in os.listdir(old_dir):
                src = os.path.join(old_dir, f)
                dst = os.path.join(new_dir, f)
                if os.path.exists(dst):
                    base, ext = os.path.splitext(f)
                    dst = os.path.join(new_dir, f"{base}_dup{ext}")
                shutil.move(src, dst)
            try: os.rmdir(old_dir)
            except OSError: pass
            print(f"MERGE: {old_name} -> {new_name}")
        except Exception as e:
            print("MERGE ERR:", e)

    # -----------------------------------------------------
    def save_vehicle_image(self, frame, vehicle, box, frame_idx):
        x1, y1, x2, y2 = box
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return
        plate_name = vehicle.final_plate or f"UNKNOWN_{vehicle.uid}"
        save_dir = os.path.join(self.vehicle_dir, plate_name)
        existing = 0
        if os.path.isdir(save_dir):
            existing = len([f for f in os.listdir(save_dir)
                            if f.lower().endswith(".jpg")])
        if existing >= CONFIG["max_files_per_dir"]:
            return
        if vehicle.saved_count >= CONFIG["max_angle_per_vehicle"]:
            return

        cur_feat = calc_feature(crop)
        if vehicle.saved_count > 0:
            for sb in vehicle.saved_boxes:
                if box_iou(box, sb) > (1.0 - CONFIG["angle_min_iou_change"]):
                    for sf in vehicle.saved_features:
                        if (1.0 - feature_similarity(cur_feat, sf)) \
                                < CONFIG["angle_min_feat_change"]:
                            return

        os.makedirs(save_dir, exist_ok=True)
        filename = f"frame{frame_idx:07d}_a{vehicle.saved_count + 1}.jpg"
        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, crop)
        vehicle.saved_boxes.append(box)
        vehicle.saved_features.append(cur_feat)
        vehicle.saved_count += 1
        print(f"SAVE [{plate_name}] #{vehicle.saved_count}: {save_path}")

    # -----------------------------------------------------
    def update_plate_vote(self, vehicle, text, score, frame_idx,
                          plate_crop=None):
        if not text:
            return
        vehicle.plate_votes.append((text, score, frame_idx))
        if len(vehicle.plate_votes) > CONFIG["max_vote_history"]:
            vehicle.plate_votes.pop(0)
        vehicle.vote_count += 1

        # P26: 保存最高分车牌特写, 用于报告
        if plate_crop is not None and plate_crop.size > 0:
            cur_best = max(
                (s for _, s, _ in vehicle.plate_votes), default=0)
            if score >= cur_best:
                try:
                    vehicle.best_plate_crop = plate_crop.copy()
                except Exception:
                    pass

        decay = CONFIG["vote_decay"]
        for pos in vehicle.char_table:
            for ch in vehicle.char_table[pos]:
                vehicle.char_table[pos][ch] *= decay

        prior_prov = CONFIG.get("region_prior_province", "")
        prior_w = CONFIG.get("region_prior_weight", 1.0)
        for i, ch in enumerate(text):
            w = score
            if i == 0 and ch == prior_prov and prior_w > 1.0:
                w *= prior_w
            vehicle.char_table[i][ch] += w

        length_score = defaultdict(float)
        for t, s, f in vehicle.plate_votes:
            gap = frame_idx - f
            w = decay ** max(0, gap // 5)
            length_score[len(t)] += s * w
        best_len = max(length_score, key=length_score.get)

        plate_chars = []
        for pos in range(best_len):
            if pos not in vehicle.char_table:
                plate_chars = None; break
            tbl = vehicle.char_table[pos]
            if not tbl:
                plate_chars = None; break
            ch = max(tbl, key=tbl.get)
            plate_chars.append(ch)

        if plate_chars is None:
            return
        new_plate = "".join(plate_chars)
        new_plate = self.soft_validate_plate(new_plate)
        if new_plate is None:
            return

        old = vehicle.final_plate
        if vehicle.vote_count >= CONFIG["vote_min_count"]:
            vehicle.final_plate = new_plate
            vehicle.confirmed = True
        elif not vehicle.final_plate:
            vehicle.final_plate = new_plate

        if old != vehicle.final_plate:
            self.recognized_plates.discard(old or "")
            self.recognized_plates.add(vehicle.final_plate)
            if old:
                self.migrate_dir(old, vehicle.final_plate)
            else:
                self.migrate_dir(f"UNKNOWN_{vehicle.uid}", vehicle.final_plate)
            if plate_crop is not None and plate_crop.size > 0:
                self._add_recent_plate(vehicle.final_plate, plate_crop)

    # -----------------------------------------------------
    def _add_recent_plate(self, text, plate_crop):
        for t, _ in self.recent_plates:
            if t == text:
                return
        # F4: 满了就不再 push, 避免 PIP 整块滚动刷新
        if len(self.recent_plates) >= CONFIG["ui_pip_max"]:
            return
        cell_h = self._scale(70)
        h, w = plate_crop.shape[:2]
        nh = cell_h; nw = max(int(w * nh / max(h, 1)), 1)
        try:
            thumb = cv2.resize(plate_crop, (nw, nh),
                               interpolation=cv2.INTER_CUBIC)
        except Exception:
            return
        self.recent_plates.append((text, thumb))

    # -----------------------------------------------------
    def detect_plates_global(self, frame):
        try:
            with torch.inference_mode():
                res = self.plate_model.predict(
                    source=frame, conf=CONFIG["plate_conf"],
                    imgsz=CONFIG["plate_imgsz"], verbose=False,
                    device=self.device,
                    half=(CONFIG["use_fp16"] and torch.cuda.is_available()),
                )
        except Exception as e:
            print("PLATE DETECT ERR:", e); return []
        if not res or res[0].boxes is None or len(res[0].boxes) == 0:
            return []
        out = []
        for pbox, pconf in zip(
            res[0].boxes.xyxy.cpu().numpy(),
            res[0].boxes.conf.cpu().numpy(),
        ):
            out.append((pbox.astype(int).tolist(), float(pconf)))
        return out

    # =====================================================
    # 后处理合并
    # =====================================================
    def _vote_best_plate_name(self, group, counts):
        prior_prov = CONFIG.get("region_prior_province", "")
        prior_w = CONFIG.get("region_prior_weight", 1.0)
        len_score = defaultdict(float)
        for name in group:
            len_score[len(name)] += counts.get(name, 0)
        if not len_score:
            return max(group, key=lambda d: counts.get(d, 0))
        best_len = max(len_score, key=len_score.get)
        char_votes = [defaultdict(float) for _ in range(best_len)]
        for name in group:
            if len(name) != best_len:
                continue
            weight = counts.get(name, 1)
            for i, ch in enumerate(name):
                w = float(weight)
                if i == 0 and ch == prior_prov and prior_w > 1.0:
                    w *= prior_w
                char_votes[i][ch] += w
        chars = []
        for i in range(best_len):
            if not char_votes[i]:
                return max(group, key=lambda d: counts.get(d, 0))
            chars.append(max(char_votes[i], key=char_votes[i].get))
        candidate = "".join(chars)
        if self.soft_validate_plate(candidate):
            return candidate
        return max(group, key=lambda d: counts.get(d, 0))

    def post_merge_similar_dirs(self):
        print("\n=== 后处理: 相似车牌目录合并 (含地区先验) ===")
        prior_prov = CONFIG.get("region_prior_province", "")
        if prior_prov:
            print(f" 地区先验: '{prior_prov}' 权重 x"
                  f"{CONFIG.get('region_prior_weight', 1.0)}")
        try:
            items = sorted([d for d in os.listdir(self.vehicle_dir)
                            if os.path.isdir(os.path.join(self.vehicle_dir, d))
                            and not d.startswith("UNKNOWN_")])
        except Exception as e:
            print(" 扫描失败:", e); return

        counts = {d: len([f for f in os.listdir(
            os.path.join(self.vehicle_dir, d))
            if f.lower().endswith(".jpg")]) for d in items}

        # 单图目录降级
        demoted = 0
        for d, n in list(counts.items()):
            if n < 2:
                try:
                    new_name = f"UNKNOWN_{uuid.uuid4().hex[:8]}"
                    self.migrate_dir(d, new_name)
                    counts.pop(d, None); demoted += 1
                except Exception:
                    pass
        if demoted:
            print(f" 单图目录降级为 UNKNOWN: {demoted} 个")

        items = [d for d in items if d in counts]

        merged = 0
        used = set()
        for i, a in enumerate(items):
            if a in used or a not in counts:
                continue
            group = [a]
            for b in items[i + 1:]:
                if b in used or b not in counts:
                    continue
                if plate_similar(a, b):
                    group.append(b)
            if len(group) == 1:
                continue
            target = self._vote_best_plate_name(group, counts)
            target_existed = target in group
            if not target_existed:
                os.makedirs(
                    os.path.join(self.vehicle_dir, target), exist_ok=True)
            for d in group:
                if d == target:
                    continue
                self.migrate_dir(d, target)
                used.add(d); merged += 1
            used.add(target)
            print(f" GROUP {group} -> {target}"
                  f"{' (新合成)' if not target_existed else ''}")
        print(f" 合并完成: 共合并 {merged} 个目录\n")

    # =====================================================
    # P26: HTML 报告
    # =====================================================
    def generate_html_report(self):
        if not CONFIG["report_enabled"]:
            return
        report_path = os.path.join(self.output_dir, "report.html")
        try:
            dirs = sorted([d for d in os.listdir(self.vehicle_dir)
                           if os.path.isdir(
                               os.path.join(self.vehicle_dir, d))])
        except Exception as e:
            print("REPORT ERR:", e); return

        known = [d for d in dirs if not d.startswith("UNKNOWN_")]
        unknown = [d for d in dirs if d.startswith("UNKNOWN_")]

        def img_to_b64(path, max_w=320):
            try:
                img = cv2.imread(path)
                if img is None: return ""
                h, w = img.shape[:2]
                if w > max_w:
                    nh = int(h * max_w / w)
                    img = cv2.resize(img, (max_w, nh))
                _, buf = cv2.imencode(".jpg", img,
                                      [cv2.IMWRITE_JPEG_QUALITY, 70])
                return "data:image/jpeg;base64," + \
                    base64.b64encode(buf.tobytes()).decode()
            except Exception:
                return ""

        rows = []
        for idx, d in enumerate(known):
            dpath = os.path.join(self.vehicle_dir, d)
            files = sorted([f for f in os.listdir(dpath)
                            if f.lower().endswith(".jpg")])
            n = len(files)
            rep = os.path.join(dpath, files[0]) if files else ""
            b64 = img_to_b64(rep)
            rows.append(f"""
            <tr>
              <td>{idx + 1}</td>
              <td class="plate">{d}</td>
              <td>{n}</td>
              <td><img src="{b64}" /></td>
              <td><span class="badge ok">已确认</span></td>
            </tr>""")

        for idx, d in enumerate(unknown):
            dpath = os.path.join(self.vehicle_dir, d)
            files = sorted([f for f in os.listdir(dpath)
                            if f.lower().endswith(".jpg")])
            n = len(files)
            rep = os.path.join(dpath, files[0]) if files else ""
            b64 = img_to_b64(rep)
            rows.append(f"""
            <tr>
              <td>{len(known) + idx + 1}</td>
              <td class="plate unknown">{d}</td>
              <td>{n}</td>
              <td><img src="{b64}" /></td>
              <td><span class="badge unk">未识别</span></td>
            </tr>""")

        html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>车牌识别报告</title>
<style>
  body {{ font-family: -apple-system, "PingFang SC", "Microsoft YaHei",
          sans-serif; background:#1a1a23; color:#eee; padding:20px; }}
  h1 {{ color:#ffc850; }}
  .stats {{ background:#22222d; padding:14px; border-radius:8px;
            margin-bottom:16px; }}
  table {{ width:100%; border-collapse:collapse; background:#22222d; }}
  th, td {{ padding:10px; border-bottom:1px solid #333;
           vertical-align:middle; }}
  th {{ background:#2d2d3a; color:#ffc850; text-align:left; }}
  td.plate {{ font-size:18px; font-weight:bold; color:#7fdfff; }}
  td.plate.unknown {{ color:#888; font-size:13px; }}
  img {{ max-width:320px; max-height:200px;
        border:1px solid #444; border-radius:4px; }}
  .badge {{ padding:4px 10px; border-radius:12px; font-size:12px; }}
  .badge.ok {{ background:#1d6f3f; color:#9be8b9; }}
  .badge.unk {{ background:#5a3a1a; color:#ffd9a4; }}
</style>
</head>
<body>
  <h1>车牌识别报告</h1>
  <div class="stats">
    <p>总车辆目录: <b>{len(dirs)}</b>　|
       已识别: <b style="color:#9be8b9">{len(known)}</b>　|
       UNKNOWN: <b style="color:#ffd9a4">{len(unknown)}</b></p>
    <p style="color:#888;font-size:12px">
       生成时间: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
  </div>
  <table>
    <thead>
      <tr><th>#</th><th>车牌</th><th>图片数</th>
          <th>代表图</th><th>状态</th></tr>
    </thead>
    <tbody>
      {"".join(rows)}
    </tbody>
  </table>
</body>
</html>"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"REPORT: {report_path}")

    # =====================================================
    # UI 绘制
    # =====================================================
    def draw_hud(self, frame, frame_idx, total):
        if not CONFIG["ui_hud_enabled"]:
            return frame
        h, w = frame.shape[:2]
        hud_h = self._scale(110)
        overlay_rect(frame, (0, 0), (w, hud_h), (15, 15, 25), 0.65)
        cv2.line(frame, (0, hud_h), (w, hud_h),
                 (180, 140, 60), self._scale(2))
        parked = sum(1 for v in self.vehicles.values() if v.is_parked)
        progress = frame_idx / max(total, 1) * 100
        frame = pil_draw_text(
            frame, "车牌识别系统  V10",
            (self._scale(20), self._scale(8)),
            (255, 200, 80), get_font(self._scale(28)))
        kpi = (f"帧 {frame_idx}/{total}    "
               f"进度 {progress:5.1f}%    "
               f"车辆 {len(self.vehicles)}    "
               f"停放 {parked}    "
               f"识别 {len(self.recognized_plates)}    "
               f"FPS {self.fps_proc:5.1f}")
        frame = pil_draw_text(
            frame, kpi,
            (self._scale(20), self._scale(46)),
            (230, 230, 240), get_font(self._scale(24)))
        bar_x = self._scale(20); bar_w = w - 2 * bar_x
        bar_h = self._scale(6); bar_y = hud_h - bar_h - self._scale(8)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), (60, 60, 80), -1)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + int(bar_w * progress / 100), bar_y + bar_h),
                      (80, 200, 255), -1)
        return frame

    def draw_label_card(self, img, anchor_xy, lines, status_color, font_size):
        font = get_font(font_size)
        pad_x = self._scale(10); pad_y = self._scale(6)
        bar_w = self._scale(6); line_gap = self._scale(2)
        pil_tmp = Image.new("RGB", (10, 10))
        d_tmp = ImageDraw.Draw(pil_tmp)
        sizes = []
        for t in lines:
            bbox = d_tmp.textbbox((0, 0), t, font=font)
            sizes.append((bbox[2] - bbox[0], bbox[3] - bbox[1]))
        text_w = max(s[0] for s in sizes)
        text_h = sum(s[1] for s in sizes) + line_gap * (len(lines) - 1)
        ax, ay = anchor_xy
        card_w = bar_w + pad_x * 2 + text_w
        card_h = text_h + pad_y * 2
        cx1 = ax; cy2 = ay - self._scale(4); cy1 = cy2 - card_h
        if cy1 < self._scale(45):
            cy1 = ay + self._scale(4); cy2 = cy1 + card_h
        cx2 = cx1 + card_w
        cx1 = max(0, cx1); cy1 = max(0, cy1)
        cx2 = min(img.shape[1], cx2); cy2 = min(img.shape[0], cy2)
        overlay_rect(img, (cx1, cy1), (cx2, cy2), (20, 20, 28), 0.72)
        cv2.rectangle(img, (cx1, cy1), (cx1 + bar_w, cy2), status_color, -1)
        tx = cx1 + bar_w + pad_x; ty = cy1 + pad_y
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil, "RGBA")
        for i, t in enumerate(lines):
            draw.text((tx, ty), t, font=font, fill=(255, 255, 255, 255))
            ty += sizes[i][1] + line_gap
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    def draw_vehicle_box(self, img, box, color, box_h):
        x1, y1, x2, y2 = box
        thickness = max(self._scale(2),
                        min(self._scale(6), int(box_h * 0.012)))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        if CONFIG["ui_corner_bracket"]:
            l = max(self._scale(12), int((x2 - x1) * 0.08))
            draw_corner_bracket(img, x1, y1, x2, y2, color, l, thickness + 1)

    def draw_pip(self, frame):
        if not CONFIG["ui_pip_enabled"] or not self.recent_plates:
            return frame
        h, w = frame.shape[:2]
        cell_h = self._scale(70); title_h = self._scale(34)
        pad = self._scale(10); panel_w = self._scale(380)
        panel_h = title_h + (cell_h + pad) * len(self.recent_plates) + pad
        x1 = self._scale(20); y2 = h - self._scale(20)
        y1 = y2 - panel_h; x2 = x1 + panel_w
        if y1 < self._scale(120):
            return frame
        overlay_rect(frame, (x1, y1), (x2, y2), (15, 15, 28), 0.72)
        cv2.rectangle(frame, (x1, y1), (x2, y2),
                      (180, 140, 60), self._scale(2))
        frame = pil_draw_text(
            frame, "近期识别车牌",
            (x1 + pad, y1 + self._scale(4)),
            (255, 200, 80), get_font(self._scale(22)))
        cy = y1 + title_h + pad // 2
        text_font = get_font(self._scale(26))
        for text, thumb in self.recent_plates:
            th = thumb.shape[0]; tw = thumb.shape[1]
            ix1 = x1 + pad; iy1 = cy
            ix2 = ix1 + tw; iy2 = iy1 + th
            if ix2 > x2 - pad:
                tw_new = x2 - pad - ix1
                if tw_new > 10:
                    thumb = cv2.resize(thumb, (tw_new, th))
                    ix2 = ix1 + tw_new
                else:
                    cy += cell_h + pad; continue
            try:
                frame[iy1:iy2, ix1:ix2] = thumb
            except Exception:
                pass
            cv2.rectangle(frame, (ix1, iy1), (ix2, iy2),
                          (200, 200, 200), 1)
            frame = pil_draw_text(
                frame, text,
                (ix2 + pad, iy1 + (th - self._scale(26)) // 2),
                (255, 255, 255), text_font)
            cy += cell_h + pad
        return frame

    # =====================================================
    # 主循环
    # =====================================================
    def process(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("无法打开视频:", video_path); return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_w = width; self.frame_h = height; self.video_fps = fps
        print(f"VIDEO: {width}x{height} {fps:.2f}fps total={total}")

        out_h = min(CONFIG["out_height"], height)
        out_w = int(round(width * out_h / height))
        if out_w % 2 == 1: out_w -= 1
        output_video = os.path.join(self.output_dir, "result.mp4")
        fourcc = cv2.VideoWriter_fourcc(*CONFIG["video_codec"])
        out = cv2.VideoWriter(output_video, fourcc, fps, (out_w, out_h))
        if not out.isOpened():
            print("avc1 编码不可用, fallback mp4v")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_video, fourcc, fps, (out_w, out_h))
        print(f"OUTPUT: {out_w}x{out_h} -> {output_video}")

        pbar = tqdm(total=total, desc="处理中")
        frame_idx = 0
        t0 = time.time()
        plate_bboxes_global = []
        last_t = time.time()

        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
            pbar.update(1)

            now = time.time()
            dt = now - last_t; last_t = now
            self._ema_dt = self._ema_dt * 0.9 + dt * 0.1
            self.fps_proc = 1.0 / max(self._ema_dt, 1e-6)

            if CONFIG["frame_stride"] > 1 and \
               frame_idx % CONFIG["frame_stride"] != 0:
                out.write(self._resize_out(frame, out_w, out_h))
                continue

            annotated = frame.copy()

            try:
                with torch.inference_mode():
                    results = self.vehicle_model.track(
                        frame, persist=True, tracker=self.botsort_yaml,
                        conf=CONFIG["vehicle_conf"], classes=[2, 3, 5, 7],
                        imgsz=CONFIG["detect_size"], verbose=False,
                        device=self.device,
                        half=(CONFIG["use_fp16"] and torch.cuda.is_available()),
                    )
            except Exception as e:
                print("TRACK ERR:", e)
                out.write(self._resize_out(annotated, out_w, out_h))
                continue

            if (results is None or results[0].boxes is None
                    or results[0].boxes.id is None):
                annotated = self.draw_hud(annotated, frame_idx, total)
                annotated = self.draw_pip(annotated)
                out.write(self._resize_out(annotated, out_w, out_h))
                continue

            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            if frame_idx % CONFIG["plate_detect_interval"] == 0:
                plate_bboxes_global = self.detect_plates_global(frame)

            for box, track_id in zip(boxes, track_ids):
                try:
                    x1, y1, x2, y2 = map(int, box)
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(width, x2); y2 = min(height, y2)
                    if x2 <= x1 or y2 <= y1: continue
                    area = (x2 - x1) * (y2 - y1)
                    if area < CONFIG["min_vehicle_area"]: continue
                    current_box = [x1, y1, x2, y2]
                    vehicle_crop = frame[y1:y2, x1:x2]
                    if vehicle_crop.size == 0: continue

                    vehicle = self.get_or_create_vehicle(
                        track_id, vehicle_crop, current_box, frame_idx)

                    if vehicle.last_box is not None:
                        d = center_distance(current_box, vehicle.last_box)
                        vehicle.motion_history.append(d)
                        if len(vehicle.motion_history) > 20:
                            vehicle.motion_history.pop(0)
                        avg = float(np.median(vehicle.motion_history))
                        if avg < CONFIG["park_distance_threshold"]:
                            vehicle.stationary_frames += 1
                        else:
                            vehicle.stationary_frames = max(
                                0, vehicle.stationary_frames - 2)
                    was_parked = vehicle.is_parked
                    vehicle.is_parked = (
                        vehicle.stationary_frames
                        >= CONFIG["park_frame_threshold"])
                    if vehicle.is_parked and not was_parked:
                        vehicle.park_start_frame = frame_idx
                    elif not vehicle.is_parked and was_parked:
                        vehicle.park_start_frame = -1
                    vehicle.last_box = current_box

                    if vehicle.feature_count < 5:
                        nf = calc_feature(vehicle_crop)
                        vehicle.feature = (
                            vehicle.feature * vehicle.feature_count + nf
                        ) / (vehicle.feature_count + 1)
                        vehicle.feature_count += 1

                    interval = (CONFIG["ocr_interval_parked"]
                                if vehicle.final_plate
                                else CONFIG["ocr_interval"])
                    if vehicle.is_parked and not vehicle.final_plate:
                        interval = max(5, CONFIG["ocr_interval"] // 2)
                    need_ocr = (
                        (frame_idx - vehicle.last_ocr_frame) >= interval)

                    if need_ocr:
                        vehicle.last_ocr_frame = frame_idx
                        matched_plates = []
                        for pbox, pconf in plate_bboxes_global:
                            cx = (pbox[0] + pbox[2]) / 2
                            cy = (pbox[1] + pbox[3]) / 2
                            if x1 <= cx <= x2 and y1 <= cy <= y2:
                                matched_plates.append((pbox, pconf))

                        if not matched_plates:
                            try:
                                with torch.inference_mode():
                                    lr = self.plate_model.predict(
                                        source=vehicle_crop,
                                        conf=CONFIG["plate_conf"],
                                        imgsz=960, verbose=False,
                                        device=self.device, half=False)
                            except Exception:
                                lr = None
                            if (lr and lr[0].boxes is not None
                                    and len(lr[0].boxes) > 0):
                                for pbox, pconf in zip(
                                    lr[0].boxes.xyxy.cpu().numpy(),
                                    lr[0].boxes.conf.cpu().numpy(),
                                ):
                                    px1, py1, px2, py2 = pbox.astype(int)
                                    matched_plates.append((
                                        [px1 + x1, py1 + y1,
                                         px2 + x1, py2 + y1], float(pconf)))

                        allow_sr = (CONFIG["sr_enable_for_parked"]
                                    and vehicle.is_parked
                                    and not vehicle.confirmed
                                    and vehicle.sr_attempts
                                    < CONFIG["sr_max_attempts_per_vehicle"])
                        if allow_sr:
                            vehicle.sr_attempts += 1

                        for pbox, pconf in matched_plates[:2]:
                            px1, py1, px2, py2 = pbox
                            pad_x = int((px2 - px1) * 0.30)
                            pad_y = int((py2 - py1) * 0.45)
                            px1 = max(0, px1 - pad_x)
                            py1 = max(0, py1 - pad_y)
                            px2 = min(width, px2 + pad_x)
                            py2 = min(height, py2 + pad_y)
                            if px2 <= px1 or py2 <= py1: continue
                            plate_crop = frame[py1:py2, px1:px2]
                            if not is_good_plate_crop(plate_crop): continue
                            text, score = self.recognize_plate(
                                plate_crop, allow_sr=allow_sr)
                            if text:
                                print(f"OCR uid={vehicle.uid} tid={track_id}: "
                                      f"{text} {score:.3f}"
                                      f"{' [SR]' if allow_sr else ''}")
                                self.update_plate_vote(
                                    vehicle, text, score, frame_idx,
                                    plate_crop=plate_crop)

                    if vehicle.is_parked:
                        if (frame_idx - vehicle.last_save_frame) \
                                >= CONFIG["save_interval"]:
                            self.save_vehicle_image(
                                frame, vehicle, current_box, frame_idx)
                            vehicle.last_save_frame = frame_idx

                    # F2: 粘滞状态 - 只升不降, 防止颜色频繁切换
                    if vehicle.confirmed and vehicle.final_plate:
                        cur_state = 3
                    elif vehicle.final_plate:
                        cur_state = 2
                    elif vehicle.is_parked:
                        cur_state = 1
                    else:
                        cur_state = 0
                    if cur_state > vehicle.sticky_render_state:
                        vehicle.sticky_render_state = cur_state
                    # 退化条件: 仅当车辆离开停放再次确认移动 OR 跨段长时间无观测
                    elif (cur_state == 0 and
                          vehicle.sticky_render_state == 1):
                        vehicle.sticky_render_state = 0  # 停放→重新移动

                    state = vehicle.sticky_render_state
                    color = {
                        0: (90, 220, 90),     # 移动绿
                        1: (60, 60, 235),     # 停放红
                        2: (80, 220, 220),    # 试探性识别青
                        3: (60, 180, 255),    # 已确认金橙
                    }[state]

                    box_h = y2 - y1
                    self.draw_vehicle_box(annotated, current_box, color, box_h)

                    # F1: 标签粘滞 - 一旦显示过车牌就一直显示, 不退回 ID
                    if vehicle.final_plate:
                        vehicle.last_plate_for_label = vehicle.final_plate
                    if vehicle.last_plate_for_label:
                        prefix = "" if vehicle.confirmed else "? "
                        line1 = f"{prefix}{vehicle.last_plate_for_label}"
                    else:
                        line1 = f"ID-{track_id}"

                    # F3: 秒数取整, 每秒只变 1 次, 避免标签卡每帧重画
                    if vehicle.is_parked and CONFIG["ui_show_park_seconds"]:
                        if vehicle.park_start_frame > 0:
                            secs = int(
                                (frame_idx - vehicle.park_start_frame) / fps
                            )
                        else:
                            secs = 0
                        line2 = f"⏱ 已停 {secs}s"
                    elif state >= 1:
                        line2 = f"⏱ 已停 ?s"  # 粘滞已停状态但当前未在停放
                    else:
                        line2 = "● 移动中"

                    if CONFIG["ui_adaptive_font"]:
                        base_size = int(box_h * 0.10)
                        base_size = max(self._scale(18),
                                        min(self._scale(40), base_size))
                    else:
                        base_size = self._scale(28)

                    annotated = self.draw_label_card(
                        annotated, (x1, y1),
                        [line1, line2], color, base_size)

                except Exception as e:
                    print("OBJECT ERR:", e); continue

            for pbox, pconf in plate_bboxes_global:
                cv2.rectangle(
                    annotated, (pbox[0], pbox[1]),
                    (pbox[2], pbox[3]), (0, 220, 220), self._scale(1))

            annotated = self.draw_hud(annotated, frame_idx, total)
            annotated = self.draw_pip(annotated)
            out.write(self._resize_out(annotated, out_w, out_h))

            if frame_idx % CONFIG["cuda_clear_interval"] == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if torch.cuda.is_available():
                mem = torch.cuda.memory_allocated() / 1024 ** 3
                if mem > 12:
                    print(f"WARN GPU MEM {mem:.2f} GB -> empty_cache")
                    torch.cuda.empty_cache()

        cap.release(); out.release(); pbar.close()
        elapsed = time.time() - t0

        # 后处理 + 报告
        self.post_merge_similar_dirs()
        self.generate_html_report()

        try:
            final_dirs = [d for d in os.listdir(self.vehicle_dir)
                          if os.path.isdir(
                              os.path.join(self.vehicle_dir, d))]
        except Exception:
            final_dirs = []
        final_known = [d for d in final_dirs if not d.startswith("UNKNOWN_")]
        final_unknown = [d for d in final_dirs if d.startswith("UNKNOWN_")]
        parked_now = sum(1 for v in self.vehicles.values() if v.is_parked)

        print("\n========== 处理完成 ==========")
        print(f"耗时         : {elapsed:.1f} s")
        print(f"输出视频     : {output_video}  ({out_w}x{out_h})")
        print(f"车辆目录     : {self.vehicle_dir}")
        print(f"HTML 报告    : {os.path.join(self.output_dir, 'report.html')}")
        print(f"唯一车辆数   : {len(self.vehicles)}")
        print(f"track_id 总数: {len(self.track_to_uid)} "
              f"-> 归并为 {len(self.vehicles)} 辆")
        print(f"识别车牌目录 : {len(final_known)}")
        print(f"UNKNOWN 目录 : {len(final_unknown)}")
        print(f"最终停放数   : {parked_now}")
        if final_known:
            print("车牌列表     :")
            for p in sorted(final_known):
                print(f"   {p}")

    def _resize_out(self, frame, out_w, out_h):
        if frame.shape[1] == out_w and frame.shape[0] == out_h:
            return frame
        return cv2.resize(frame, (out_w, out_h),
                          interpolation=cv2.INTER_AREA)


# =========================================================
# 单独后处理入口
# =========================================================
def merge_only(output_dir):
    print(f"=== 单独后处理: 合并 {output_dir}/vehicles 内相似车牌 ===")
    sys_ = PlateRecognitionSystem.__new__(PlateRecognitionSystem)
    sys_.output_dir = output_dir
    sys_.vehicle_dir = os.path.join(output_dir, "vehicles")
    if not os.path.isdir(sys_.vehicle_dir):
        print("vehicles 目录不存在"); return
    sys_.post_merge_similar_dirs()
    sys_.generate_html_report()


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default=None)
    parser.add_argument("-o", "--output", default="./output4")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--out-height", type=int, default=1080)
    parser.add_argument("--no-pip", action="store_true")
    parser.add_argument("--no-bracket", action="store_true")
    parser.add_argument("--no-sr", action="store_true")
    parser.add_argument("--no-dual-ocr", action="store_true")
    parser.add_argument("--no-perspective", action="store_true")
    parser.add_argument("--no-rotation", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--report-only", action="store_true",
                        help="只生成 HTML 报告, 不做合并")
    args = parser.parse_args()

    CONFIG["frame_stride"] = args.stride
    CONFIG["out_height"] = args.out_height
    if args.no_pip: CONFIG["ui_pip_enabled"] = False
    if args.no_bracket: CONFIG["ui_corner_bracket"] = False
    if args.no_sr: CONFIG["sr_enable_for_parked"] = False
    if args.no_dual_ocr: CONFIG["dual_ocr_enabled"] = False
    if args.no_perspective: CONFIG["perspective_enabled"] = False
    if args.no_rotation: CONFIG["rotation_aug_enabled"] = False

    if args.report_only:
        sys_ = PlateRecognitionSystem.__new__(PlateRecognitionSystem)
        sys_.output_dir = args.output
        sys_.vehicle_dir = os.path.join(args.output, "vehicles")
        sys_.generate_html_report(); sys.exit(0)

    if args.merge_only:
        merge_only(args.output); sys.exit(0)

    if not args.input:
        print("必须提供 -i 输入视频, 除非用 --merge-only / --report-only")
        sys.exit(1)

    try:
        system = PlateRecognitionSystem(args.output)
        system.process(args.input)
    except Exception:
        print(traceback.format_exc())
