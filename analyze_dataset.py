"""
EchoRisk Task1 数据集统计分析脚本

统计维度:
    - 样本级: 训练/验证各分组的记录数、患者数
    - 患者级: 每患者时间点分布 (T1-T5 覆盖)
    - LVEF:   均值/标准差/最小值/最大值/四分位数/分布直方图
    - 分类标签: biomarker_elevated 患病率 (全样本 + 按时间点分层)
    - 缺失值:  video_a4c / video_a2c / biomarker_elevated 缺失统计
    - 切面:   双切面 vs 单切面 vs 仅 A4C / 仅 A2C 分布
    - DICOM:  随机抽样若干文件，统计帧数 / 分辨率 / 帧率 / 文件大小

使用方法:
    python analyze_dataset.py
    python analyze_dataset.py --dicom_sample 10   # 随机抽样 10 个 DICOM 文件
"""

import os
import csv
import random
import argparse
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import TASK1_TRAIN_CSV, TASK1_VAL_CSV, TASK1_TRAIN_DICOM, TASK1_VAL_DICOM, OUTPUT_ROOT

# ============================================================
# 工具函数
# ============================================================

def load_csv(path):
    """读取标签 CSV，返回 row 列表 (list of dict)"""
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["lvef"] = float(row["lvef"]) if row["lvef"].strip() else None
            val = row.get("biomarker_elevated", "").strip()
            row["biomarker_elevated"] = int(val) if val in ("0", "1") else None
            rows.append(row)
    return rows


def safe_int(x):
    try:
        return int(float(x))
    except (ValueError, TypeError):
        return None


# ============================================================
# 单分割统计
# ============================================================

def analyze_split(csv_path, dicom_root, split_name):
    """对 train 或 val 做全面统计，返回汇总 dict 用于报告"""
    rows = load_csv(csv_path)
    data = {}
    patients = sorted(set(r["patient_id"] for r in rows))

    # ---- 样本级 ----
    data["csv_path"] = csv_path
    data["split"] = split_name
    data["n_records"] = len(rows)
    data["n_patients"] = len(patients)

    # ---- 每个患者的时间点分布 ----
    tp_per_patient = Counter()
    tp_sequences = Counter()
    for pid in patients:
        pts = sorted([r["timepoint"] for r in rows if r["patient_id"] == pid])
        tp_per_patient[len(pts)] += 1
        tp_sequences["→".join(pts)] += 1
    data["records_per_patient"] = dict(tp_per_patient)
    data["top_timepoint_sequences"] = tp_sequences.most_common(10)

    # 各时间点记录数
    tp_counts = Counter(r["timepoint"] for r in rows)
    data["records_by_timepoint"] = dict(sorted(tp_counts.items()))

    # ---- LVEF ----
    lvefs = [r["lvef"] for r in rows if r["lvef"] is not None]
    if lvefs:
        data["lvef_mean"] = np.mean(lvefs)
        data["lvef_std"] = np.std(lvefs)
        data["lvef_min"] = np.min(lvefs)
        data["lvef_max"] = np.max(lvefs)
        data["lvef_q25"] = np.percentile(lvefs, 25)
        data["lvef_median"] = np.percentile(lvefs, 50)
        data["lvef_q75"] = np.percentile(lvefs, 75)
        data["lvef_n"] = len(lvefs)
    else:
        data["lvef_mean"] = float("nan")

    # ---- LVEF 分层 ----
    data["lvef_below_40"] = sum(1 for v in lvefs if v < 40)
    data["lvef_40_50"] = sum(1 for v in lvefs if 40 <= v < 50)
    data["lvef_50_70"] = sum(1 for v in lvefs if 50 <= v < 70)
    data["lvef_above_70"] = sum(1 for v in lvefs if v >= 70)

    # ---- biomarker_elevated ----
    bio = [r["biomarker_elevated"] for r in rows if r["biomarker_elevated"] is not None]
    data["biomarker_elevated_known"] = len(bio)
    data["biomarker_elevated_missing"] = len(rows) - len(bio)
    if bio:
        data["biomarker_elevated_pos"] = sum(bio)
        data["biomarker_elevated_neg"] = len(bio) - sum(bio)
        data["biomarker_elevated_rate"] = sum(bio) / len(bio) * 100

    # 按时间点分层的 biomarker
    bio_by_tp = {}
    for tp in ["T1", "T2", "T3", "T4", "T5"]:
        vals = [r["biomarker_elevated"] for r in rows
                if r["timepoint"] == tp and r["biomarker_elevated"] is not None]
        if vals:
            bio_by_tp[tp] = {"n": len(vals), "pos": sum(vals), "rate": sum(vals) / len(vals) * 100}
    data["biomarker_by_timepoint"] = bio_by_tp

    # ---- 缺失统计 ----
    missing_a4c = sum(1 for r in rows if not r["video_a4c"].strip())
    missing_a2c = sum(1 for r in rows if not r["video_a2c"].strip())
    data["missing_a4c"] = missing_a4c
    data["missing_a2c"] = missing_a2c
    data["missing_a4c_pct"] = missing_a4c / len(rows) * 100
    data["missing_a2c_pct"] = missing_a2c / len(rows) * 100
    data["missing_either"] = sum(1 for r in rows if not r["video_a4c"].strip() or not r["video_a2c"].strip())
    data["missing_both"] = sum(1 for r in rows if not r["video_a4c"].strip() and not r["video_a2c"].strip())

    # ---- 切面分布 ----
    both = sum(1 for r in rows if r["video_a4c"].strip() and r["video_a2c"].strip())
    only_a4c = sum(1 for r in rows if r["video_a4c"].strip() and not r["video_a2c"].strip())
    only_a2c = sum(1 for r in rows if not r["video_a4c"].strip() and r["video_a2c"].strip())
    data["view_both"] = both
    data["view_only_a4c"] = only_a4c
    data["view_only_a2c"] = only_a2c

    return data


# ============================================================
# DICOM 抽样统计
# ============================================================

def scan_dicom_headers(csv_path, dicom_root, sample_size=10):
    """随机抽样 DICOM 文件，返回帧数/分辨率/帧率/大小等统计"""
    rows = load_csv(csv_path)
    all_files = []
    for r in rows:
        pid = r["patient_id"]
        tp = r["timepoint"]
        for view_key in ["video_a4c", "video_a2c"]:
            fname = r[view_key].strip()
            if fname:
                fpath = os.path.join(dicom_root, pid, tp, fname)
                if os.path.exists(fpath):
                    all_files.append((fpath, view_key))

    if len(all_files) == 0:
        return {"error": f"No DICOM files found in {dicom_root}"}

    sample = random.sample(all_files, min(sample_size, len(all_files)))

    n_frames = []
    fps_list = []
    file_size_mb = []
    resolutions = []
    errors = 0

    try:
        import pydicom
    except ImportError:
        return {"error": "pydicom not installed. Run: pip install pydicom"}

    for fpath, view_key in sample:
        try:
            ds = pydicom.dcmread(fpath, stop_before_pixels=False)
            nf = getattr(ds, "NumberOfFrames", 1)
            n_frames.append(nf)

            fps = safe_int(getattr(ds, "RecommendedDisplayFrameRate", None))
            if fps is None:
                fps = safe_int(getattr(ds, "CineRate", None))
            if fps is None and hasattr(ds, "FrameTime"):
                try:
                    fps = round(1000.0 / float(ds.FrameTime), 1)
                except (ValueError, ZeroDivisionError):
                    pass
            if fps is not None:
                fps_list.append(fps)

            rows_ds = getattr(ds, "Rows", None)
            cols = getattr(ds, "Columns", None)
            if rows_ds and cols:
                resolutions.append(f"{rows_ds}×{cols}")

            file_size_mb.append(os.path.getsize(fpath) / (1024 * 1024))
        except Exception as e:
            errors += 1
            print(f"  [WARN] Failed to read {os.path.basename(fpath)}: {e}")

    stats = {
        "sampled": len(sample),
        "errors": errors,
        "n_frames": n_frames,
        "fps": fps_list,
        "resolutions": Counter(resolutions),
        "file_size_mb": file_size_mb,
    }

    if n_frames:
        stats["frames_min"] = min(n_frames)
        stats["frames_max"] = max(n_frames)
        stats["frames_mean"] = round(np.mean(n_frames), 1)
        stats["frames_std"] = round(np.std(n_frames), 1)
    if fps_list:
        stats["fps_min"] = min(fps_list)
        stats["fps_max"] = max(fps_list)
        stats["fps_mean"] = round(np.mean(fps_list), 1)
    if file_size_mb:
        stats["size_min_mb"] = round(min(file_size_mb), 2)
        stats["size_max_mb"] = round(max(file_size_mb), 2)
        stats["size_mean_mb"] = round(np.mean(file_size_mb), 2)
    return stats


# ============================================================
# 画图
# ============================================================

def plot_lvef_distribution(train_data, val_data, output_dir):
    """绘制 train/val LVEF 分布对比直方图"""
    os.makedirs(output_dir, exist_ok=True)

    def loads(path):
        return [r["lvef"] for r in load_csv(path) if r["lvef"] is not None]

    train_lvef = loads(train_data["csv_path"])
    val_lvef = loads(val_data["csv_path"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, lvef_list, name in zip(axes, [train_lvef, val_lvef], ["Train", "Validation"]):
        ax.hist(lvef_list, bins=30, edgecolor="white", alpha=0.8)
        ax.axvline(np.mean(lvef_list), color="red", linestyle="--", linewidth=1.5,
                   label=f"Mean={np.mean(lvef_list):.1f}")
        ax.axvline(50, color="gray", linestyle=":", linewidth=1, label="LVEF=50")
        ax.set_title(f"{name} LVEF Distribution (n={len(lvef_list)})")
        ax.set_xlabel("LVEF (%)")
        ax.set_ylabel("Count")
        ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "lvef_distribution.png"), dpi=150)
    plt.close(fig)
    print(f"  LVEF 分布图已保存到 {output_dir}/lvef_distribution.png")


def plot_biomarker_by_timepoint(train_data, val_data, output_dir):
    """绘制 biomarker_elevated 按时间点的患病率折线图"""
    os.makedirs(output_dir, exist_ok=True)
    tps = ["T1", "T2", "T3", "T4", "T5"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, data, name in zip(axes, [train_data, val_data], ["Train", "Validation"]):
        bio_tp = data["biomarker_by_timepoint"]
        rates = [bio_tp[tp]["rate"] if tp in bio_tp else None for tp in tps]
        counts = [bio_tp[tp]["n"] if tp in bio_tp else 0 for tp in tps]

        ax.plot(tps, rates, marker="o", linewidth=2, markersize=8)
        for i, (tp, r, c) in enumerate(zip(tps, rates, counts)):
            if r is not None:
                ax.annotate(f"n={c}", (tp, r), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8)
        ax.set_ylim(0, max(filter(None, rates)) * 1.3 if any(r is not None for r in rates) else 100)
        ax.set_title(f"{name} Biomarker Elevated Rate by Timepoint")
        ax.set_xlabel("Timepoint")
        ax.set_ylabel("Rate (%)")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "biomarker_by_timepoint.png"), dpi=150)
    plt.close(fig)
    print(f"  biomarker 时间趋势图已保存到 {output_dir}/biomarker_by_timepoint.png")


# ============================================================
# 报告输出
# ============================================================

def print_report(train_data, val_data, dicom_stats, output_dir):
    """打印/保存完整统计报告"""
    lines = []
    def w(s=""):
        lines.append(s)
        print(s)

    w("=" * 70)
    w("  EchoRisk Task1 数据集统计报告")
    w("=" * 70)

    for data in [train_data, val_data]:
        w()
        w(f"--- {data['split'].upper()} ---")
        w(f"  文件:              {data['csv_path']}")
        w(f"  记录数:            {data['n_records']}")
        w(f"  患者数:            {data['n_patients']}")
        w(f"  每患者记录数分布:  {data['records_per_patient']}")
        w(f"  各时间点记录数:    {data['records_by_timepoint']}")
        w()

        if not np.isnan(data["lvef_mean"]):
            w("  LVEF (%):")
            w(f"    N:              {data['lvef_n']}")
            w(f"    Mean ± Std:     {data['lvef_mean']:.2f} ± {data['lvef_std']:.2f}")
            w(f"    Range:          [{data['lvef_min']}, {data['lvef_max']}]")
            w(f"    Q25 / Median / Q75:  {data['lvef_q25']:.1f} / {data['lvef_median']:.1f} / {data['lvef_q75']:.1f}")
            w(f"    分层:  <40: {data['lvef_below_40']} | 40-50: {data['lvef_40_50']} | 50-70: {data['lvef_50_70']} | ≥70: {data['lvef_above_70']}")
        else:
            w("  LVEF: N/A")

        w()
        w("  biomarker_elevated:")
        w(f"    已知: {data['biomarker_elevated_known']}  |  缺失: {data['biomarker_elevated_missing']}")
        if data['biomarker_elevated_known']:
            w(f"    阳性: {data['biomarker_elevated_pos']}  |  阴性: {data['biomarker_elevated_neg']}")
            w(f"    患病率: {data['biomarker_elevated_rate']:.1f}%")
        w("    按时间点: {}".format(
            {tp: f"{v['rate']:.1f}% (n={v['n']})" for tp, v in data.get("biomarker_by_timepoint", {}).items()}
        ))

        w()
        w("  缺失统计:")
        w(f"    A4C 缺失: {data['missing_a4c']} ({data['missing_a4c_pct']:.1f}%)")
        w(f"    A2C 缺失: {data['missing_a2c']} ({data['missing_a2c_pct']:.1f}%)")
        w(f"    任一缺失: {data['missing_either']}")
        w(f"    双切面均缺失: {data['missing_both']}")

        w()
        w("  切面分布:")
        w(f"    双切面 (A4C+A2C):       {data['view_both']} ({data['view_both']/data['n_records']*100:.1f}%)")
        w(f"    仅 A4C:                {data['view_only_a4c']} ({data['view_only_a4c']/data['n_records']*100:.1f}%)")
        w(f"    仅 A2C:                {data['view_only_a2c']} ({data['view_only_a2c']/data['n_records']*100:.1f}%)")

        if data["top_timepoint_sequences"]:
            w()
            w("  前 5 时间点序列:")
            for seq, cnt in data["top_timepoint_sequences"][:5]:
                w(f"    {seq}: {cnt} 患者")

    # ---- DICOM 元数据 ----
    w()
    w("--- DICOM 元数据 (随机抽样) ---")
    if "error" in dicom_stats:
        w(f"  {dicom_stats['error']}")
    else:
        w(f"  抽样数:   {dicom_stats['sampled']}  |  读取失败: {dicom_stats['errors']}")
        if "frames_min" in dicom_stats:
            w(f"  帧数:     {dicom_stats['frames_min']} ~ {dicom_stats['frames_max']} "
              f"(mean={dicom_stats['frames_mean']}, std={dicom_stats['frames_std']})")
        if "fps_min" in dicom_stats:
            w(f"  帧率:     {dicom_stats['fps_min']} ~ {dicom_stats['fps_max']} "
              f"(mean={dicom_stats['fps_mean']} fps)")
        if dicom_stats.get("resolutions"):
            w(f"  分辨率:")
            for res, cnt in dicom_stats["resolutions"].most_common(5):
                w(f"    {res}: {cnt} 个")
        if "size_min_mb" in dicom_stats:
            w(f"  文件大小: {dicom_stats['size_min_mb']} ~ {dicom_stats['size_max_mb']} MB "
              f"(mean={dicom_stats['size_mean_mb']} MB)")

    # ---- 总体 ----
    w()
    w("--- 总体 ---")
    w(f"  训练集: {train_data['n_records']} 条记录, {train_data['n_patients']} 名患者")
    w(f"  验证集: {val_data['n_records']} 条记录, {val_data['n_patients']} 名患者")
    w(f"  合计:   {train_data['n_records'] + val_data['n_records']} 条记录, "
      f"{train_data['n_patients'] + val_data['n_patients']} 名患者")
    w("=" * 70)

    # ---- 保存到文件 ----
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "dataset_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n报告已保存到 {report_path}")


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="EchoRisk Task1 数据集统计分析")
    parser.add_argument("--dicom_sample", type=int, default=10,
                        help="随机抽样 DICOM 文件数 (默认 10, 设为 0 跳过)")
    parser.add_argument("--output", type=str, default=None,
                        help="输出目录 (默认 output/analysis)")
    args = parser.parse_args()

    output_dir = args.output or os.path.join(OUTPUT_ROOT, "analysis")

    print("正在统计训练集...")
    train_data = analyze_split(TASK1_TRAIN_CSV, TASK1_TRAIN_DICOM, "train")

    print("正在统计验证集...")
    val_data = analyze_split(TASK1_VAL_CSV, TASK1_VAL_DICOM, "val")

    if args.dicom_sample > 0:
        print(f"正在随机抽样 {args.dicom_sample} 个 DICOM 文件...")
        dicom_stats = scan_dicom_headers(TASK1_TRAIN_CSV, TASK1_TRAIN_DICOM, args.dicom_sample)
    else:
        dicom_stats = {"error": "跳过 (--dicom_sample 0)"}

    print("正在绘制图表...")
    plot_lvef_distribution(train_data, val_data, output_dir)
    plot_biomarker_by_timepoint(train_data, val_data, output_dir)

    print_report(train_data, val_data, dicom_stats, output_dir)


if __name__ == "__main__":
    main()
