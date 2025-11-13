import os
import sys
import glob
import time
import argparse
from pathlib import Path
import numpy as np

# 将 src 加入 Python 路径，方便示例脚本直接导入项目模块
ROOT = Path(__file__).resolve().parents[1]
# 将包含 src 的父目录加入 sys.path，使得 `import src` 可用
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

try:
    import cv2
    CV_AVAILABLE = True
except Exception:
    cv2 = None
    CV_AVAILABLE = False

# 注意：features 包位于 src 包下，需以 src.features 导入
from src.features.tracker import FeatureDetector, OpticalFlowTracker


def list_images(data_dir: str) -> list:
    files = glob.glob(os.path.join(data_dir, '*.png'))
    files.sort()  # 按文件名排序（时间戳）
    return files


def draw_points(image_gray: np.ndarray, points: np.ndarray, color=(0, 255, 0)) -> np.ndarray:
    if not CV_AVAILABLE:
        # 无 OpenCV 情况下直接返回原图（不绘制）
        return image_gray
    img_color = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    for p in points:
        cv2.circle(img_color, (int(p[0]), int(p[1])), 2, color, -1)
    return img_color


def run_demo(args):
    if not CV_AVAILABLE:
        print('[ERROR] 未检测到 OpenCV(cv2)。请安装: pip install opencv-python opencv-contrib-python')
        return 1

    data_dir = args.data_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    files = list_images(data_dir)
    if len(files) < 2:
        print(f'[ERROR] 图像数量不足(<2)。请检查目录: {data_dir}')
        return 1

    # 读取首帧并转换灰度
    prev_img = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)
    if prev_img is None:
        print(f'[ERROR] 无法读取图像: {files[0]}')
        return 1

    # 初始化检测器与光流跟踪器
    detector = FeatureDetector(
        detector_type=args.detector,
        num_features=args.features,
        quality_level=args.quality,
        min_distance=args.min_distance,
        fast_threshold=args.fast_threshold,
    )
    of_tracker = OpticalFlowTracker(
        window_size=(args.window_size, args.window_size),
        max_level=args.max_level,
        min_eigen_threshold=args.min_eig_threshold,
    )

    # 首帧特征检测
    if args.grid:
        prev_points = detector.detect_grid(prev_img, grid_size=(args.grid_rows, args.grid_cols), features_per_cell=max(1, args.features // max(1, args.grid_rows * args.grid_cols)))
    else:
        prev_points = detector.detect(prev_img)

    # 若 FAST 没有检测到点，尝试回退到 Shi-Tomasi
    if prev_points is None or len(prev_points) == 0:
        detector.detector_type = 'GFTT'
        prev_points = detector.detect(prev_img)

    print(f'[INFO] 首帧检测到特征点数量: {len(prev_points)}')

    if len(prev_points) == 0:
        print('[ERROR] 首帧未检测到任何特征。请调整参数或检查图像质量。')
        return 1

    # 保存首帧可视化
    if args.save_viz:
        img0_viz = draw_points(prev_img, prev_points, color=(0, 255, 0))
        cv2.imwrite(os.path.join(output_dir, 'frame_0000_features.png'), img0_viz)

    # 逐帧跟踪
    ratios = []
    disp_medians = []
    processed = 0
    t_start = time.time()

    for idx, f in enumerate(files[1:], start=1):
        if args.limit_frames > 0 and idx >= args.limit_frames:
            break

        curr_img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if curr_img is None:
            print(f'[WARN] 跳过无法读取的图像: {f}')
            continue

        # 使用 KLT 进行光流跟踪
        curr_points, status = of_tracker.track(prev_img, curr_img, prev_points)
        tracked_points = curr_points[status]

        tracked_ratio = (tracked_points.shape[0] / max(1, prev_points.shape[0]))
        ratios.append(tracked_ratio)

        # 位移统计（仅成功跟踪的点）
        disps = np.linalg.norm((tracked_points - prev_points[status]), axis=1) if tracked_points.shape[0] > 0 else np.array([])
        disp_median = float(np.median(disps)) if disps.size > 0 else 0.0
        disp_medians.append(disp_median)

        if args.save_viz:
            img_viz = draw_points(curr_img, tracked_points, color=(0, 255, 0))
            out_name = os.path.join(output_dir, f'frame_{idx:04d}_tracked.png')
            cv2.imwrite(out_name, img_viz)

        # 若特征数量过低，可选择在当前帧补充检测（简单策略）
        if tracked_points.shape[0] < max(20, args.features // 4) and args.replenish:
            # 检测新点并与现有点合并（不做去重，仅用于验证管线）
            new_pts = detector.detect(curr_img) if not args.grid else detector.detect_grid(
                curr_img, grid_size=(args.grid_rows, args.grid_cols), features_per_cell=max(1, args.features // max(1, args.grid_rows * args.grid_cols))
            )
            if new_pts is not None and len(new_pts) > 0:
                # 合并并限制总量
                combined = np.vstack([tracked_points, new_pts]) if tracked_points.size > 0 else new_pts
                if combined.shape[0] > args.features:
                    combined = combined[:args.features]
                prev_points = combined
            else:
                prev_points = tracked_points
        else:
            prev_points = tracked_points

        prev_img = curr_img
        processed += 1

        if idx % 50 == 0:
            print(f'[INFO] 处理到帧 {idx}, 跟踪成功率: {tracked_ratio:.3f}, 中位位移: {disp_median:.2f}px, 活跃点数: {prev_points.shape[0]}')

    t_elapsed = time.time() - t_start

    # 汇总统计
    avg_ratio = float(np.mean(ratios)) if len(ratios) > 0 else 0.0
    med_ratio = float(np.median(ratios)) if len(ratios) > 0 else 0.0
    med_disp = float(np.median(disp_medians)) if len(disp_medians) > 0 else 0.0

    print('===== 跟踪统计结果 =====')
    print(f'- 处理帧数: {processed}')
    print(f'- 初始特征点: {len(detector.detect(prev_img)) if prev_img is not None else len(prev_points)}')
    print(f'- 平均跟踪成功率: {avg_ratio:.3f}')
    print(f'- 中位跟踪成功率: {med_ratio:.3f}')
    print(f'- 中位位移: {med_disp:.2f} 像素')
    print(f'- 总耗时: {t_elapsed:.2f} 秒')
    print(f'- 可视化输出目录: {output_dir} (save_viz={args.save_viz})')

    # 简单正确性判断（启发式）：
    # - 相邻帧的平均跟踪成功率通常应在 0.5 以上（视运动与场景而定）
    if processed >= 10 and avg_ratio > 0.5:
        print('[OK] KLT 跟踪成功率合理，tracker.py 管线看起来工作正常。')
        return 0
    else:
        print('[WARN] 跟踪成功率较低或处理帧过少，请检查参数、图像质量或依赖安装。')
        return 0  # 仍返回 0，但提示用户


def build_argparser():
    p = argparse.ArgumentParser(description='GEOVINS FAST+KLT 跟踪示例 (TD_07/export 数据)')
    p.add_argument('--data-dir', type=str, default=str(ROOT.parent / 'TD_07' / 'export' / 'images' / '_camera_array_cam0_image_raw_compressed'), help='图像数据目录')
    p.add_argument('--output-dir', type=str, default=str(ROOT.parent / 'TD_07' / 'export' / 'images' / 'tracker_output'), help='输出可视化目录')
    p.add_argument('--detector', type=str, default='FAST', choices=['FAST', 'GFTT', 'KLT', 'SHI-TOMASI'], help='特征检测器类型')
    p.add_argument('--features', type=int, default=400, help='初始检测的最大特征数量')
    p.add_argument('--fast-threshold', type=int, default=20, help='FAST 阈值')
    p.add_argument('--quality', type=float, default=0.01, help='GFTT 质量阈值')
    p.add_argument('--min-distance', type=float, default=10.0, help='特征最小距离')
    p.add_argument('--window-size', type=int, default=21, help='KLT 窗口大小（正方形）')
    p.add_argument('--max-level', type=int, default=3, help='金字塔层数')
    p.add_argument('--min-eig-threshold', dest='min_eig_threshold', type=float, default=1e-3, help='最小特征值阈值')
    p.add_argument('--grid', action='store_true', default=True, help='是否使用网格化检测以均匀分布特征')
    p.add_argument('--grid-rows', type=int, default=5, help='网格行数')
    p.add_argument('--grid-cols', type=int, default=5, help='网格列数')
    p.add_argument('--replenish', action='store_true', default=True, help='当活跃点过少时是否补充检测')
    p.add_argument('--save-viz', action='store_true', default=True, help='是否保存每帧的跟踪可视化')
    p.add_argument('--limit-frames', type=int, default=300, help='最多处理的帧数（0 表示全部）')
    return p


def main():
    args = build_argparser().parse_args()
    code = run_demo(args)
    sys.exit(code)


if __name__ == '__main__':
    main()