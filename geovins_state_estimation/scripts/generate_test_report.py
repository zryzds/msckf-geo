import argparse
import json
import os
import xml.etree.ElementTree as ET


def parse_junit(xml_path: str):
    summary = {
        "tests": 0,
        "errors": 0,
        "failures": 0,
        "skipped": 0,
        "time": 0.0,
        "suites": [],
    }
    if not os.path.exists(xml_path):
        return summary

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Pytest writes either <testsuite> or <testsuites>
    suites = []
    if root.tag == "testsuite":
        suites = [root]
    elif root.tag == "testsuites":
        suites = list(root)
    else:
        suites = []

    for s in suites:
        s_sum = {
            "name": s.attrib.get("name", "suite"),
            "tests": int(s.attrib.get("tests", 0)),
            "errors": int(s.attrib.get("errors", 0)),
            "failures": int(s.attrib.get("failures", 0)),
            "skipped": int(s.attrib.get("skipped", 0)),
            "time": float(s.attrib.get("time", 0.0)),
            "cases": [],
        }
        for tc in s.findall("testcase"):
            case = {
                "classname": tc.attrib.get("classname", ""),
                "name": tc.attrib.get("name", ""),
                "time": float(tc.attrib.get("time", 0.0)),
                "status": "passed",
            }
            if tc.find("failure") is not None:
                case["status"] = "failure"
            elif tc.find("error") is not None:
                case["status"] = "error"
            elif tc.find("skipped") is not None:
                case["status"] = "skipped"
            s_sum["cases"].append(case)

        summary["tests"] += s_sum["tests"]
        summary["errors"] += s_sum["errors"]
        summary["failures"] += s_sum["failures"]
        summary["skipped"] += s_sum["skipped"]
        summary["time"] += s_sum["time"]
        summary["suites"].append(s_sum)

    return summary


def load_metrics(metrics_path: str):
    if not os.path.exists(metrics_path):
        return None
    try:
        with open(metrics_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def write_markdown(out_path: str, junit: dict, metrics: dict):
    lines = []
    lines.append("# GEOVINS 滤波器测试报告")
    lines.append("")
    lines.append("## 总览")
    lines.append(f"- 测试用例数: {junit['tests']}")
    lines.append(f"- 失败: {junit['failures']}  错误: {junit['errors']}  跳过: {junit['skipped']}")
    lines.append(f"- 总用时: {junit['time']:.2f}s")

    lines.append("")
    lines.append("## 详细用例")
    for s in junit.get("suites", []):
        lines.append(f"- 套件 `{s['name']}`: {s['tests']} 个用例, 失败 {s['failures']}, 错误 {s['errors']}, 跳过 {s['skipped']}, 用时 {s['time']:.2f}s")
        for c in s.get("cases", []):
            lines.append(f"  - [{c['status']}] {c['classname']}::{c['name']} ({c['time']:.3f}s)")

    if metrics:
        lines.append("")
        lines.append("## IMU CSV 指标摘要")
        lines.append(f"- 数据文件: `{metrics.get('csv_path', '')}`")
        lines.append(f"- 采样条数: {metrics.get('num_measurements', 0)}")
        lines.append(f"- dt 中位数: {metrics.get('dt_median', 0.0):.6f}s (min {metrics.get('dt_min', 0.0):.6f}s, max {metrics.get('dt_max', 0.0):.6f}s)")
        lines.append(f"- 末端位置: {metrics.get('final_position', [])}")
        lines.append(f"- 末端速度: {metrics.get('final_velocity', [])}")
        lines.append(f"- 位移: {metrics.get('displacement', 0.0):.3f} m")
        lines.append(f"- 协方差最小特征值: {metrics.get('cov_min_eigenvalue', 0.0):.6e}")

    content = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser(description="Generate Markdown test report from JUnit XML and metrics JSON.")
    parser.add_argument("--xml", required=True, help="Path to JUnit XML report")
    parser.add_argument("--metrics", required=False, default="", help="Path to IMU metrics JSON (optional)")
    parser.add_argument("--out", required=True, help="Output Markdown path")
    args = parser.parse_args()

    junit = parse_junit(args.xml)
    metrics = load_metrics(args.metrics) if args.metrics else None

    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    write_markdown(args.out, junit, metrics)


if __name__ == "__main__":
    main()