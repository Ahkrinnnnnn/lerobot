#!/usr/bin/env python3
"""读取 Parquet（如 LeRobot 数据集分片），计算相邻行时间戳间隔并可视化。

默认读取当前目录下的 file-000.parquet；可用 --parquet 指定路径。

依赖: pip install pyarrow matplotlib

示例:
  python visualize_parquet_intervals.py
  python visualize_parquet_intervals.py /path/to/data.parquet --save plot.png
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

try:
    import pyarrow.parquet as pq
except ImportError as e:
    raise SystemExit(
        "需要 pyarrow。安装: pip install pyarrow matplotlib\n"
    ) from e

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    raise SystemExit(
        "需要 matplotlib。安装: pip install matplotlib\n"
    ) from e


def column_to_seconds(column) -> list[float]:
    """将一列时间戳转为秒（float）。"""
    arr = column.to_pylist()
    if not arr:
        return []
    x0 = arr[0]
    if x0 is not None and hasattr(x0, "timestamp"):
        return [float(v.timestamp()) if v is not None else float("nan") for v in arr]
    out: list[float] = []
    for v in arr:
        if v is None:
            out.append(float("nan"))
            continue
        f = float(v)
        if abs(f) > 1e12:
            f /= 1e9
        elif abs(f) > 1e9:
            f /= 1e6
        out.append(f)
    return out


def guess_timestamp_column(names: list[str]) -> str | None:
    """按常见列名猜测时间戳列。"""
    candidates = [
        "timestamp",
        "system_timestamp",
        "time",
        "t",
    ]
    lower = {n.lower(): n for n in names}
    for c in candidates:
        if c in lower:
            return lower[c]
    for n in names:
        nl = n.lower()
        if "timestamp" in nl or nl.endswith("_ts"):
            return n
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "parquet",
        nargs="?",
        default="file-000.parquet",
        type=Path,
        help="Parquet 文件路径（默认: file-000.parquet）",
    )
    parser.add_argument(
        "--column",
        default=None,
        help="时间戳列名；省略则自动猜测",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        metavar="PATH",
        help="将图保存到 PATH（不弹窗）",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="保存或不显示窗口时与 --save 联用",
    )
    args = parser.parse_args()

    path = args.parquet.resolve()
    if not path.is_file():
        raise SystemExit(f"文件不存在: {path}")

    table = pq.read_table(path)
    col = args.column
    if col is None:
        col = guess_timestamp_column(table.column_names)
        if col is None:
            raise SystemExit(
                f"无法猜测时间戳列。请用 --column 指定。可用列: {table.column_names}"
            )
        print(f"使用列: {col!r}（可用 --column 覆盖）")
    elif col not in table.column_names:
        raise SystemExit(f"列 {col!r} 不存在。可用列: {table.column_names}")

    t = column_to_seconds(table.column(col))
    if any(x != x for x in t):
        raise SystemExit("时间戳列在转换后出现 NaN。")
    if len(t) < 2:
        raise SystemExit("至少需要 2 行才能计算间隔。")

    dt = [t[i + 1] - t[i] for i in range(len(t) - 1)]
    median_dt = statistics.median(dt)
    mean_dt = statistics.mean(dt)
    std_dt = statistics.pstdev(dt) if len(dt) > 1 else 0.0
    inferred_hz = 1.0 / median_dt if median_dt > 0 else float("nan")

    print(f"文件: {path}")
    print(f"行数: {len(t)}, 时间列: {col!r}")
    print(f"间隔 Δt — 中位数: {median_dt:.6f} s  →  约 {inferred_hz:.3f} Hz")
    print(f"间隔 Δt — 均值: {mean_dt:.6f} s, 标准差: {std_dt:.6f} s")
    print(f"间隔 Δt — 最小/最大: {min(dt):.6f} / {max(dt):.6f} s")

    idx = list(range(len(dt)))
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [2, 1]})

    ax0 = axes[0]
    ax0.scatter(idx, dt, s=10, c="tab:blue", alpha=0.65, label=r"$\Delta t$ (s)")
    ax0.axhline(median_dt, color="tab:orange", linestyle="--", linewidth=1.2, label=f"median ({median_dt:.4f} s)")
    ax0.set_xlabel("间隔序号（第 k 与 k+1 行之间）")
    ax0.set_ylabel("Δt (s)")
    ax0.set_title(f"{path.name}: 相邻行时间间隔")
    ax0.legend(loc="upper right")
    ax0.grid(True, alpha=0.3)

    ax1 = axes[1]
    ax1.hist(dt, bins=min(80, max(10, len(dt) // 50)), color="tab:green", edgecolor="white", alpha=0.85)
    ax1.axvline(median_dt, color="tab:orange", linestyle="--", linewidth=1.2, label="median")
    ax1.set_xlabel("Δt (s)")
    ax1.set_ylabel("频数")
    ax1.set_title("间隔分布（直方图）")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    if args.save is not None:
        fig.savefig(args.save, dpi=150)
        print(f"图已保存: {args.save.resolve()}")
    if not args.no_show or args.save is None:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
