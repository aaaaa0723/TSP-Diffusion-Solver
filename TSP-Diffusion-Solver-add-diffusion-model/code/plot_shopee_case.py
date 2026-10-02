"""Redraw saved tours with full Taichung boundary and an explicitly labelled zoom."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate import load_boundary_rings
from road_geometry import road_geometry


def plot_case(directory):
    directory = Path(directory)
    record = json.loads((directory / "plot_records.json").read_text(encoding="utf-8"))[0]
    points = np.asarray(record["coords"])
    rings = [np.asarray(ring) for ring in load_boundary_rings()]
    boundary = np.concatenate(rings)
    plt.rcParams["font.family"] = ["Microsoft JhengHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), gridspec_kw={"height_ratios": [1, 1.5]})
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    model_name = "舊模型＋直線距離解碼器" if report.get("feature_mode", "").startswith("legacy") else "模型＋道路解碼器"
    for col, (key, label, length) in enumerate([
            ("reference", "OR-Tools（30 秒）", record["reference_length"]),
            ("route", model_name, record["route_length"]) ]):
        line = road_geometry(points, record[key], directory / "road_geometry")
        for row in range(2):
            ax = axes[row, col]
            for ring in rings:
                ax.fill(ring[:, 0], ring[:, 1], color="#f1f5f9", zorder=0)
                ax.plot(ring[:, 0], ring[:, 1], color="#64748b", linewidth=.8)
            ax.plot(line[:, 0], line[:, 1], color="#2563eb", linewidth=.65, alpha=.85, label="OSRM 行車路線")
            ax.scatter(points[:, 0], points[:, 1], s=5 if row == 0 else 9, color="#dc2626", label="蝦皮門市", zorder=3)
            extent = boundary if row == 0 else points
            ax.set_xlim(extent[:, 0].min() - .025, extent[:, 0].max() + .025)
            ax.set_ylim(extent[:, 1].min() - .025, extent[:, 1].max() + .025)
            ax.set_aspect(1 / np.cos(np.deg2rad(24.2)))
            ax.set_xlabel("經度")
            ax.set_ylabel("緯度")
            ax.set_title(f"{label}｜{length:.2f} 公里\n" + ("台中市完整行政範圍" if row == 0 else "門市分布區域放大（非完整市界）"))
            if row == 0:
                ax.text(121.05, 24.27, "和平區／山區", color="#64748b", fontsize=11)
                ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(f"台中市蝦皮 {len(points)} 間門市｜OSRM 行車路線比較｜Gap {record['gap_percent']:+.2f}%", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(directory / "route_comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    plot_case(parser.parse_args().directory)
