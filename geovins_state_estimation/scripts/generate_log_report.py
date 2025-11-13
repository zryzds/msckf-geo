import argparse
import csv
import json
import math
import os
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np


def _safe_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _find_indices(header: List[str]):
    def idx(name, default=None):
        try:
            return header.index(name)
        except ValueError:
            return default

    ts_i = idx("timestamp", 1)
    wx_i = idx("wx", 5)
    wy_i = idx("wy", 6)
    wz_i = idx("wz", 7)
    ax_i = idx("ax", 8)
    ay_i = idx("ay", 9)
    az_i = idx("az", 10)
    return ts_i, wx_i, wy_i, wz_i, ax_i, ay_i, az_i


def parse_imu_csv(path: str, max_rows: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse IMU CSV with columns: timestamp, wx, wy, wz, ax, ay, az.
    Returns (timestamps, angular_velocities[N,3], linear_accelerations[N,3]).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    ts_list = []
    w_list = []
    a_list = []

    with open(path, 'r', newline='') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return np.array([]), np.empty((0, 3)), np.empty((0, 3))

        ts_i, wx_i, wy_i, wz_i, ax_i, ay_i, az_i = _find_indices(header)

        count = 0
        for row in reader:
            if max_rows is not None and count >= max_rows:
                break

            vals = [
                _safe_float(row[ts_i]) if ts_i is not None and ts_i < len(row) else None,
                _safe_float(row[wx_i]) if wx_i is not None and wx_i < len(row) else None,
                _safe_float(row[wy_i]) if wy_i is not None and wy_i < len(row) else None,
                _safe_float(row[wz_i]) if wz_i is not None and wz_i < len(row) else None,
                _safe_float(row[ax_i]) if ax_i is not None and ax_i < len(row) else None,
                _safe_float(row[ay_i]) if ay_i is not None and ay_i < len(row) else None,
                _safe_float(row[az_i]) if az_i is not None and az_i < len(row) else None,
            ]

            if any(v is None for v in vals):
                continue

            t, wx, wy, wz, ax, ay, az = vals
            ts_list.append(t)
            w_list.append([wx, wy, wz])
            a_list.append([ax, ay, az])
            count += 1

    # Strictly increasing timestamps
    ts = np.array(ts_list, dtype=np.float64)
    w = np.array(w_list, dtype=np.float64)
    a = np.array(a_list, dtype=np.float64)
    if len(ts) == 0:
        return ts, w, a
    # filter non-increasing
    mask = np.ones_like(ts, dtype=bool)
    last_t = ts[0]
    for i in range(1, len(ts)):
        mask[i] = ts[i] > last_t
        if mask[i]:
            last_t = ts[i]
    ts = ts[mask]
    w = w[mask]
    a = a[mask]
    return ts, w, a


def compute_imu_stats(ts: np.ndarray, w: np.ndarray, a: np.ndarray) -> dict:
    dts = np.diff(ts) if len(ts) > 1 else np.array([])
    acc_mag = np.linalg.norm(a, axis=1) if len(a) > 0 else np.array([])
    gyro_mag = np.linalg.norm(w, axis=1) if len(w) > 0 else np.array([])

    stats = {
        "num_measurements": int(len(ts)),
        "dt_mean": float(np.mean(dts)) if len(dts) else 0.0,
        "dt_median": float(np.median(dts)) if len(dts) else 0.0,
        "dt_min": float(np.min(dts)) if len(dts) else 0.0,
        "dt_max": float(np.max(dts)) if len(dts) else 0.0,
        "sample_rate_mean_hz": float(1.0 / np.mean(dts)) if len(dts) and np.mean(dts) > 0 else 0.0,
        "sample_rate_median_hz": float(1.0 / np.median(dts)) if len(dts) and np.median(dts) > 0 else 0.0,
        "dt_outliers_lt_zero": int(np.sum(dts <= 0)) if len(dts) else 0,
        "dt_outliers_gt_20ms": int(np.sum(dts > 0.02)) if len(dts) else 0,
        "gyro_mean": w.mean(axis=0).tolist() if len(w) else [0.0, 0.0, 0.0],
        "gyro_std": w.std(axis=0).tolist() if len(w) else [0.0, 0.0, 0.0],
        "gyro_mag_max": float(np.max(gyro_mag)) if len(gyro_mag) else 0.0,
        "gyro_mag_median": float(np.median(gyro_mag)) if len(gyro_mag) else 0.0,
        "gyro_mag_p95": float(np.percentile(gyro_mag, 95)) if len(gyro_mag) else 0.0,
        "acc_mean": a.mean(axis=0).tolist() if len(a) else [0.0, 0.0, 0.0],
        "acc_std": a.std(axis=0).tolist() if len(a) else [0.0, 0.0, 0.0],
        "acc_mag_max": float(np.max(acc_mag)) if len(acc_mag) else 0.0,
        "acc_mag_median": float(np.median(acc_mag)) if len(acc_mag) else 0.0,
        "acc_mag_p95": float(np.percentile(acc_mag, 95)) if len(acc_mag) else 0.0,
    }
    return stats


def quat_multiply(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = r
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float64)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    if n == 0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def euler_from_quat(q: np.ndarray) -> Tuple[float, float, float]:
    """Return roll, pitch, yaw in radians from quaternion [w,x,y,z]."""
    w, x, y, z = q
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def integrate_orientation(ts: np.ndarray, w: np.ndarray) -> dict:
    if len(ts) == 0:
        return {"final_rpy_deg": [0.0, 0.0, 0.0], "yaw_total_abs_deg": 0.0, "yaw_net_deg": 0.0}

    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    yaw_abs = 0.0
    yaw_net = 0.0

    for i in range(1, len(ts)):
        dt = ts[i] - ts[i - 1]
        if dt <= 0:
            continue
        wx, wy, wz = w[i]
        omega = np.array([wx, wy, wz], dtype=np.float64)
        theta = float(np.linalg.norm(omega) * dt)
        if theta < 1e-12:
            continue
        axis = omega / np.linalg.norm(omega)
        half = 0.5 * theta
        dq = np.array([
            math.cos(half),
            axis[0] * math.sin(half),
            axis[1] * math.sin(half),
            axis[2] * math.sin(half),
        ], dtype=np.float64)
        q = quat_multiply(q, dq)
        q = quat_normalize(q)
        # accumulate yaw angle approximation using wz*dt
        yaw_abs += abs(wz * dt)
        yaw_net += wz * dt

    roll, pitch, yaw = euler_from_quat(q)
    return {
        "final_rpy_deg": [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)],
        "yaw_total_abs_deg": math.degrees(yaw_abs),
        "yaw_net_deg": math.degrees(yaw_net),
    }


def load_metrics_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def write_log(out_path: str, csv_path: str, imu_stats: dict, ori_stats: dict, metrics: dict):
    lines = []
    lines.append("GEOVINS 测试与 IMU 统计日志")
    lines.append(f"时间: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append(f"CSV 文件: {csv_path}")
    lines.append("")

    # IMU stats
    lines.append("=== 原始 IMU 统计 ===")
    lines.append(f"样本数: {imu_stats.get('num_measurements', 0)}")
    lines.append(f"dt: mean {imu_stats.get('dt_mean', 0.0):.6f}s, median {imu_stats.get('dt_median', 0.0):.6f}s, min {imu_stats.get('dt_min', 0.0):.6f}s, max {imu_stats.get('dt_max', 0.0):.6f}s")
    lines.append(f"采样率: mean {imu_stats.get('sample_rate_mean_hz', 0.0):.2f}Hz, median {imu_stats.get('sample_rate_median_hz', 0.0):.2f}Hz")
    lines.append(f"dt 异常: <=0 共 {imu_stats.get('dt_outliers_lt_zero', 0)} 条, >20ms 共 {imu_stats.get('dt_outliers_gt_20ms', 0)} 条")
    gyro_mean = imu_stats.get('gyro_mean', [0.0, 0.0, 0.0])
    gyro_std = imu_stats.get('gyro_std', [0.0, 0.0, 0.0])
    lines.append(f"角速度均值 [wx, wy, wz]: [{gyro_mean[0]:.6f}, {gyro_mean[1]:.6f}, {gyro_mean[2]:.6f}] rad/s")
    lines.append(f"角速度标准差 [wx, wy, wz]: [{gyro_std[0]:.6f}, {gyro_std[1]:.6f}, {gyro_std[2]:.6f}] rad/s")
    lines.append(f"角速度模值: max {imu_stats.get('gyro_mag_max', 0.0):.6f}, median {imu_stats.get('gyro_mag_median', 0.0):.6f}, p95 {imu_stats.get('gyro_mag_p95', 0.0):.6f}")
    acc_mean = imu_stats.get('acc_mean', [0.0, 0.0, 0.0])
    acc_std = imu_stats.get('acc_std', [0.0, 0.0, 0.0])
    lines.append(f"加速度均值 [ax, ay, az]: [{acc_mean[0]:.6f}, {acc_mean[1]:.6f}, {acc_mean[2]:.6f}] m/s^2")
    lines.append(f"加速度标准差 [ax, ay, az]: [{acc_std[0]:.6f}, {acc_std[1]:.6f}, {acc_std[2]:.6f}] m/s^2")
    lines.append(f"加速度模值: max {imu_stats.get('acc_mag_max', 0.0):.6f}, median {imu_stats.get('acc_mag_median', 0.0):.6f}, p95 {imu_stats.get('acc_mag_p95', 0.0):.6f}")
    lines.append("")

    # Orientation integration
    lines.append("=== 姿态积分摘要 ===")
    rpy = ori_stats.get('final_rpy_deg', [0.0, 0.0, 0.0])
    lines.append(f"最终姿态 [roll, pitch, yaw] (deg): [{rpy[0]:.3f}, {rpy[1]:.3f}, {rpy[2]:.3f}]")
    lines.append(f"累计绝对 yaw 变化 (deg): {ori_stats.get('yaw_total_abs_deg', 0.0):.3f}")
    lines.append(f"净 yaw 变化 (deg): {ori_stats.get('yaw_net_deg', 0.0):.3f}")
    lines.append("")

    # Filter metrics if available
    if metrics:
        lines.append("=== MSCKF 传播指标 ===")
        fp = metrics.get('final_position', [0.0, 0.0, 0.0])
        fv = metrics.get('final_velocity', [0.0, 0.0, 0.0])
        disp = metrics.get('displacement', 0.0)
        eig = metrics.get('cov_min_eigenvalue', 0.0)
        lines.append(f"末端位置: [{fp[0]:.6f}, {fp[1]:.6f}, {fp[2]:.6f}] m")
        lines.append(f"末端速度: [{fv[0]:.6f}, {fv[1]:.6f}, {fv[2]:.6f}] m/s")
        lines.append(f"位移: {disp:.3f} m")
        lines.append(f"协方差最小特征值: {eig:.6e}")
        lines.append("")

    content = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser(description="Generate .log report with extended IMU stats and filter metrics")
    parser.add_argument('--csv', required=False, help='IMU CSV file path')
    parser.add_argument('--metrics', required=False, help='Metrics JSON created by tests')
    parser.add_argument('--out', required=False, default=os.path.normpath(os.path.join('geovins_state_estimation', 'core_test_report.log')), help='Output .log path')
    parser.add_argument('--max-rows', type=int, default=None, help='Limit number of rows parsed from CSV')
    args = parser.parse_args()

    metrics = load_metrics_json(args.metrics) if args.metrics else {}
    csv_path = args.csv or metrics.get('csv_path')
    if not csv_path:
        raise SystemExit("CSV path not provided and not found in metrics JSON")

    # If metrics have num_measurements, align CSV parsing rows to keep comparability
    max_rows = args.max_rows if args.max_rows is not None else metrics.get('num_measurements')
    ts, w, a = parse_imu_csv(csv_path, max_rows=max_rows)
    imu_stats = compute_imu_stats(ts, w, a)
    ori_stats = integrate_orientation(ts, w)

    write_log(args.out, csv_path, imu_stats, ori_stats, metrics)
    print(f"Wrote log report: {args.out}")


if __name__ == '__main__':
    main()