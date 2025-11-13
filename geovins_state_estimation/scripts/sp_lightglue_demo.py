import os
import glob
import argparse

import numpy as np
import sys

try:
    import cv2
except Exception:
    cv2 = None

# 将项目根目录加入 sys.path，便于脚本内导入 src 包
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.matcher import SuperPointLightGlueMatcher


def find_images(folder: str):
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff',
            '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.TIF', '*.TIFF')
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(folder, '**', ext), recursive=True))
    files = sorted(set(files))
    return files


def main():
    parser = argparse.ArgumentParser(description='Run SuperPoint+LightGlue matching on dataset images.')
    parser.add_argument('--dir', type=str,
                        default=r"C:\Users\Admin\Desktop\msckf-geo\TD_07\export",
                        help='Directory containing input images.')
    parser.add_argument('--output', type=str,
                        default=r"C:\Users\Admin\Desktop\msckf-geo\geovins_state_estimation\output\sp_lightglue",
                        help='Directory to save visualization results.')
    parser.add_argument('--max_kpts', type=int, default=2048, help='Max keypoints per image for SuperPoint.')
    args = parser.parse_args()

    img_paths = find_images(args.dir)
    if len(img_paths) < 2:
        raise RuntimeError(f"在目录中未找到足够的图片: {args.dir}")

    os.makedirs(args.output, exist_ok=True)

    # 使用前两张图做示例匹配（可扩展为成对遍历）
    path0, path1 = img_paths[0], img_paths[1]
    print(f"匹配两张图像:\n  {path0}\n  {path1}")

    matcher = SuperPointLightGlueMatcher(max_num_keypoints=args.max_kpts)
    result = matcher.match_paths(path0, path1)

    # 可视化输出
    save_path = os.path.join(args.output, 'sp_lightglue_match_0_1.png')
    vis = matcher.visualize(result, save_path=save_path)

    k = result['matches_idx'].shape[0]
    print(f"匹配完成：共 {k} 对匹配。")
    print(f"可视化已保存：{save_path}")


if __name__ == '__main__':
    main()