#!/usr/bin/env python3
"""
GEOVINS Demo (Real Data): Visual-Inertial Navigation using IMU CSV + Images

改造版 demo.py：删除 GPS 相关内容，读取指定 IMU CSV 与图像目录，
并使用自定义特征提取与跟踪（src/features/tracker.py）驱动 MSCKF 更新。
"""
import os
import sys
import glob
import csv
import time
import numpy as np

# 允许从 examples/ 运行，加入项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.state import create_initial_state
from src.core.types import IMUMeasurement, Quaternion, Pose
from src.sensors.imu_model import IMUModel
from src.sensors.camera_model import PinholeCameraModel
from src.filter.msckf import MSCKF
from src.features.tracker import FeatureTracker, OpticalFlowTracker, FeatureDetector


# ==== 数据源路径（按用户要求） ====
IMU_CSV_PATH = os.path.normpath(r"c:/Users/Admin/Desktop/msckf-geo/TD_07/export/imu.csv")
IMAGE_DIR = os.path.normpath(r"c:/Users/Admin/Desktop/msckf-geo/TD_07/export/images/_camera_array_cam0_image_raw_compressed/")
RESIZE_TO_WIDTH = 512  # 目标缩放宽度（保持纵横比）；若原始宽度<=此值则不缩放


def load_imu_csv(csv_path: str):
    """加载 IMU CSV，返回 (measurements, first_quaternion, t0)。

    CSV 列：topic,timestamp,qx,qy,qz,qw,wx,wy,wz,ax,ay,az
    """
    measurements: list[IMUMeasurement] = []
    q_first = None
    t0 = None

    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = float(row['timestamp'])
                wx = float(row['wx']); wy = float(row['wy']); wz = float(row['wz'])
                ax = float(row['ax']); ay = float(row['ay']); az = float(row['az'])

                measurements.append(IMUMeasurement(
                    timestamp=ts,
                    angular_velocity=np.array([wx, wy, wz], dtype=np.float64),
                    linear_acceleration=np.array([ax, ay, az], dtype=np.float64),
                ))

                if q_first is None:
                    qx = float(row['qx']); qy = float(row['qy']); qz = float(row['qz']); qw = float(row['qw'])
                    q_first = Quaternion(w=qw, x=qx, y=qy, z=qz)
                    t0 = ts
            except Exception:
                # 跳过异常行
                continue

    # 依据时间排序（稳妥起见）
    measurements.sort(key=lambda m: m.timestamp)
    if t0 is None and measurements:
        t0 = measurements[0].timestamp

    return measurements, q_first, t0


def list_images_with_timestamps(image_dir: str):
    """列出图像文件并解析文件名中的纳秒时间戳为秒。返回 [(t_sec, path), ...]。"""
    exts = ('*.png', '*.jpg', '*.jpeg')
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(image_dir, ext)))

    items: list[tuple[float, str]] = []
    for p in files:
        base = os.path.splitext(os.path.basename(p))[0]
        try:
            # 文件名形如 1679129146878229316.png → 纳秒
            ts_ns = int(base)
            ts_sec = ts_ns / 1e9
            items.append((ts_sec, p))
        except Exception:
            # 非时间戳命名，忽略
            continue

    items.sort(key=lambda x: x[0])
    return items


def read_image(path: str):
    """读取灰度图，优先使用 OpenCV；失败时返回 None。"""
    try:
        import cv2
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return img
    except Exception:
        return None


def pixels_to_normalized(camera_model: PinholeCameraModel, pixels: np.ndarray) -> np.ndarray:
    """像素 (N,2) → 归一化坐标 (N,2)，考虑去畸变。"""
    if pixels is None or len(pixels) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    uvn = []
    for u, v in pixels:
        ray = camera_model.unproject(np.array([u, v], dtype=np.float64), depth=1.0, undistort=True)
        uvn.append([ray[0], ray[1]])
    return np.asarray(uvn, dtype=np.float64)


def resize_keep_aspect(img: np.ndarray, scale: float) -> np.ndarray:
    """按比例缩放图像，保持纵横比。scale=1.0 时返回原图。"""
    if img is None or scale is None or abs(scale - 1.0) < 1e-8:
        return img
    try:
        import cv2
        h, w = img.shape[:2]
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    except Exception:
        return img


def pixels_resized_to_normalized(camera_model: PinholeCameraModel, pixels_resized: np.ndarray, scale: float) -> np.ndarray:
    """将缩放后图像上的像素坐标先映射回原始尺寸，再转换为归一化坐标。"""
    if pixels_resized is None or len(pixels_resized) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if scale is None or scale <= 0:
        scale = 1.0
    # 映射回原始像素坐标
    pixels_orig = np.asarray(pixels_resized, dtype=np.float64) / float(scale)
    return pixels_to_normalized(camera_model, pixels_orig)


def run_demo():
    print("=" * 70)
    print("GEOVINS Demo (IMU CSV + Images)")
    print("=" * 70)
    print()

    print("1. 初始化与数据加载")
    print("-" * 70)
    print(f"   - 读取 IMU: {IMU_CSV_PATH}")
    imu_measurements, q_first, t0 = load_imu_csv(IMU_CSV_PATH)
    print(f"   - IMU 数据条数: {len(imu_measurements)}")

    print(f"   - 列出图像: {IMAGE_DIR}")
    image_items = list_images_with_timestamps(IMAGE_DIR)
    print(f"   - 图像帧数: {len(image_items)}")

    if len(image_items) == 0:
        raise RuntimeError("图像目录为空或无法访问；请检查路径与权限。")

    # 用首帧尺寸推断内参中心；焦距若未知，取 width 作为尺度（近似）
    first_img = read_image(image_items[0][1])
    if first_img is None:
        width, height = 640, 480
        print("   - 警告: 无法读取首帧图像，使用默认分辨率 640x480")
    else:
        height, width = first_img.shape[:2]
        print(f"   - 图像分辨率: {width}x{height}")

    # 计算缩放因子（仅当原始宽度大于目标宽度时进行下采样）
    if width > RESIZE_TO_WIDTH:
        resize_scale = RESIZE_TO_WIDTH / float(width)
        print(f"   - 图像缩放: 宽 {width} → {RESIZE_TO_WIDTH}，scale={resize_scale:.3f}")
    else:
        resize_scale = 1.0
        if width < RESIZE_TO_WIDTH:
            print(f"   - 图像宽度 {width} ≤ {RESIZE_TO_WIDTH}，不缩放")

    # 初始化滤波器
    print("   - 初始化 MSCKF 组件")
    initial_orientation = q_first if q_first is not None else Quaternion.identity()
    initial_state = create_initial_state(
        position=np.zeros(3),
        velocity=np.zeros(3),
        orientation=initial_orientation,
        timestamp=t0 if t0 is not None else 0.0
    )

    imu_model = IMUModel(
        gyro_noise_density=1.6968e-04,
        accel_noise_density=2.0000e-03
    )

    # 简化内参：fx=fy≈width，cx,cy≈图像中心。如有标定文件，建议替换为真实值。
    camera_model = PinholeCameraModel(
        width=width, height=height,
        fx=float(width), fy=float(width),
        cx=float(width)/2.0, cy=float(height)/2.0
    )

    T_cam_imu = Pose(position=np.zeros(3), orientation=Quaternion.identity())

    msckf = MSCKF(
        initial_state=initial_state,
        imu_model=imu_model,
        camera_model=camera_model,
        T_cam_imu=T_cam_imu,
        max_camera_states=15
    )

    feature_tracker = FeatureTracker(max_track_length=15, min_track_length=3, max_features=300)
    optical_flow = OpticalFlowTracker()
    detector = FeatureDetector(detector_type="GFTT", num_features=300, quality_level=0.01, min_distance=8.0)

    # 标定与时间同步策略确认
    fx, fy, cx, cy = camera_model.fx, camera_model.fy, camera_model.cx, camera_model.cy
    cam_wh = (camera_model.width, camera_model.height)
    R_ci = T_cam_imu.orientation.to_rotation_matrix()
    det_R = np.linalg.det(R_ci)
    t_ci = T_cam_imu.position

    print("   - 初始化完成！")
    print(f"   - 相机内参: fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}, 分辨率={cam_wh}")
    print(f"   - 外参 T_cam_imu: |t|={np.linalg.norm(t_ci):.3f} m, det(R)={det_R:.6f}")
    if abs(det_R - 1.0) > 1e-3:
        print("   - 警告: 外参旋转矩阵行列式偏离 1，请复核方向/单位")
    if np.linalg.norm(t_ci) > 1.0:
        print("   - 注意: 相机-IMU 平移超过 1m，请确认单位是否为米")
    print("   - 时间策略: 本示例按 IMU<=图像时间戳批量传播，不做插值。如出现较大时间间隔将提示。")
    print("   - 跟踪缩放: 追踪阶段按宽 512 像素缩放，归一化前映射回原尺寸，保持几何一致。")
    print()

    print("2. 按时间顺序处理 IMU 与图像")
    print("-" * 70)

    # 进度控制与统计
    start_time = time.time()
    PROGRESS_EVERY = 25  # 每处理多少帧打印一次进度
    imu_events_propagated = 0

    imu_idx = 0
    imu_batch: list[IMUMeasurement] = []
    pixel_map: dict[int, np.ndarray] = {}  # feature_id → 最近像素坐标

    prev_image = None
    trajectory = []

    for frame_idx, (t_img, img_path) in enumerate(image_items, start=1):
        # 进度提示：开始处理该帧
        if frame_idx % PROGRESS_EVERY == 0 or frame_idx <= 3:
            print(f"   - [进度] 开始处理帧 {frame_idx}/{len(image_items)} @ {t_img:.3f}s", flush=True)

        # 收集 IMU 到当前图像时间戳
        while imu_idx < len(imu_measurements) and imu_measurements[imu_idx].timestamp <= t_img:
            imu_batch.append(imu_measurements[imu_idx])
            imu_idx += 1

        # 时间同步检查：最近 IMU 与当前图像的时间差
        if imu_idx > 0:
            last_imu_time = imu_measurements[imu_idx - 1].timestamp
            dt_img_imu = t_img - last_imu_time
            if dt_img_imu > 0.02:
                print(f"     · 注意: 图像与最近 IMU 间隔 {dt_img_imu:.3f}s，建议做插值/对齐", flush=True)

        if len(imu_batch) >= 2:
            batch_len = len(imu_batch)
            msckf.propagate_imu(imu_batch)
            imu_events_propagated += batch_len
            imu_batch = []
            # 进度提示：IMU 传播
            if frame_idx % PROGRESS_EVERY == 0 or frame_idx <= 3:
                print(f"     · 传播 IMU 样本 {batch_len}，累计传播 {imu_events_propagated}", flush=True)

        # 相机状态扩充
        msckf.augment_state(timestamp=t_img)
        curr_cam_id = msckf.camera_state_counter - 1  # 新增相机状态 ID
        if frame_idx % PROGRESS_EVERY == 0 or frame_idx <= 3:
            print(f"     · 扩充相机状态 ID={curr_cam_id}，当前滑窗={msckf.state.num_camera_states}", flush=True)

        # 读取当前图像并按比例缩放
        curr_image_full = read_image(img_path)
        curr_image = resize_keep_aspect(curr_image_full, resize_scale)
        if curr_image is None:
            print(f"   - 警告: 无法读取图像 {img_path}，跳过特征更新")
            prev_image = curr_image
            continue

        if prev_image is None:
            # 首帧：检测初始特征并加入跟踪（像素→归一化）
            pts_px = detector.detect_grid(curr_image, grid_size=(6, 8), features_per_cell=8)
            pts_norm = pixels_resized_to_normalized(camera_model, pts_px, resize_scale)
            new_features = feature_tracker.add_features([(u, v) for u, v in pts_norm], camera_state_id=curr_cam_id)
            for feat, px in zip(new_features, pts_px):
                pixel_map[feat.feature_id] = np.array(px, dtype=np.float32)
            if frame_idx % PROGRESS_EVERY == 0 or frame_idx <= 3:
                print(f"     · 首帧检测并加入特征 {len(new_features)}", flush=True)
        else:
            # 后续帧：用光流跟踪上一帧的活跃特征
            active_features = feature_tracker.get_active_features()
            if len(active_features) > 0:
                prev_points = []
                ordered_features = []
                for feat in active_features:
                    if feat.feature_id in pixel_map:
                        prev_points.append(pixel_map[feat.feature_id])
                        ordered_features.append(feat)
                prev_points = np.asarray(prev_points, dtype=np.float32)

                if len(prev_points) > 0:
                    curr_points, status = optical_flow.track(prev_image, curr_image, prev_points)
                    curr_obs = []
                    keep_idx = 0
                    for i, feat in enumerate(ordered_features):
                        if bool(status[i]):
                            u_px, v_px = curr_points[i]
                            uvn = pixels_resized_to_normalized(camera_model, np.array([[u_px, v_px]], dtype=np.float64), resize_scale)
                            curr_obs.append((feat.feature_id, float(uvn[0, 0]), float(uvn[0, 1])))
                            pixel_map[feat.feature_id] = curr_points[i]
                        # 若未跟上，在 track_features 中会被判定 LOST 并移除

                    tracked, lost = feature_tracker.track_features(ordered_features, curr_obs, camera_state_id=curr_cam_id)

                    # 移除丢失特征的像素缓存
                    for feat in lost:
                        if feat.feature_id in pixel_map:
                            pixel_map.pop(feat.feature_id, None)
                    if frame_idx % PROGRESS_EVERY == 0 or frame_idx <= 3:
                        print(f"     · 光流跟踪：跟上 {len(tracked)}，丢失 {len(lost)}", flush=True)

            # 若活跃特征不足，补充新检测
            if feature_tracker.num_active_tracks < feature_tracker.max_features * 0.6:
                pts_px_new = detector.detect_grid(curr_image, grid_size=(6, 8), features_per_cell=6)
                pts_norm_new = pixels_resized_to_normalized(camera_model, pts_px_new, resize_scale)
                added = feature_tracker.add_features([(u, v) for u, v in pts_norm_new], camera_state_id=curr_cam_id)
                for feat, px in zip(added, pts_px_new):
                    pixel_map[feat.feature_id] = np.array(px, dtype=np.float32)
                if frame_idx % PROGRESS_EVERY == 0 or frame_idx <= 3:
                    print(f"     · 补充新特征 {len(added)}，活跃={feature_tracker.num_active_tracks}", flush=True)

        # 滤波更新：取满足条件的特征并进行 MSCKF 更新
        ready_features = feature_tracker.get_features_for_update()
        if len(ready_features) > 0:
            bad_ids = msckf.update_features(ready_features)
            if len(bad_ids) > 0:
                feature_tracker.remove_features_by_id(bad_ids)
                if frame_idx % PROGRESS_EVERY == 0 or frame_idx <= 3:
                    print(f"     · 质量剪枝：移除劣质特征 {len(bad_ids)}", flush=True)
            if frame_idx % PROGRESS_EVERY == 0 or frame_idx <= 3:
                print(f"     · MSCKF 更新特征 {len(ready_features)}", flush=True)

        # 同步清理不再存在的相机观测
        valid_cam_ids = list(msckf.state.camera_states.keys())
        feature_tracker.prune_tracks(valid_cam_ids)

        # 记录轨迹（每帧）
        st = msckf.get_state()
        trajectory.append({
            'timestamp': t_img,
            'position': st.imu_state.position.copy(),
            'velocity': st.imu_state.velocity.copy(),
            'num_camera_states': st.num_camera_states
        })

        prev_image = curr_image

        # 进度提示：完成该帧
        if frame_idx % PROGRESS_EVERY == 0 or frame_idx <= 3:
            elapsed = time.time() - start_time
            print(f"   - [进度] 完成帧 {frame_idx}/{len(image_items)}，滑窗={st.num_camera_states}，耗时 {elapsed:.1f}s", flush=True)

    print(f"   - 处理 IMU 数量: {imu_idx}")
    print(f"   - 处理图像帧数: {len(image_items)}")
    print()

    print("3. 结果与统计")
    print("-" * 70)
    final_state = msckf.get_state()
    final_cov = msckf.get_covariance()

    print("   最终状态：")
    print(f"   - 位置:     {final_state.imu_state.position}")
    print(f"   - 速度:     {final_state.imu_state.velocity}")
    print(f"   - 陀螺偏置: {final_state.imu_state.gyro_bias}")
    print(f"   - 加计偏置: {final_state.imu_state.accel_bias}")
    print(f"   - 相机窗口: {final_state.num_camera_states}")

    pos_cov = final_cov[3:6, 3:6]
    pos_std = np.sqrt(np.diag(pos_cov))
    print(f"   位置不确定度（std）：{pos_std}")

    # 采样打印轨迹
    print("   轨迹采样：")
    print("   " + "-" * 66)
    print("   {:>12s}  {:>12s}  {:>12s}  {:>12s}  {:>8s}".format(
        "Time(s)", "X(m)", "Y(m)", "Z(m)", "CamWin"
    ))
    print("   " + "-" * 66)
    for point in trajectory[::max(1, len(trajectory)//10 or 1)]:
        pos = point['position']
        print("   {:12.3f}  {:12.3f}  {:12.3f}  {:12.3f}  {:8d}".format(
            point['timestamp'], pos[0], pos[1], pos[2], point['num_camera_states']
        ))
    print("   " + "-" * 66)
    print()

    print("=" * 70)
    print("Demo 完成！")
    print("=" * 70)


def main():
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\n\n用户中断。")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
