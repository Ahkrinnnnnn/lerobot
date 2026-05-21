#!/usr/bin/env python3
"""Read a parquet file and check whether timestamps match a target FPS (default 30 Hz).

Shows a dot plot: x = frame index (start of each interval), y = Δt between that frame and the next.

Requires: pyarrow, matplotlib (``pip install pyarrow matplotlib``).
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

try:
    import pyarrow.parquet as pq
except ImportError as e:
    raise SystemExit(
        "This script needs pyarrow. Install with: pip install pyarrow\n"
    ) from e

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    plt = None  # type: ignore[misc, assignment]
    _matplotlib_import_error = e
else:
    _matplotlib_import_error = None


def column_to_seconds(column) -> list[float]:
    """Convert a pyarrow column to seconds (float)."""
    arr = column.to_pylist()
    if not arr:
        return []
    x0 = arr[0]
    # datetime from parquet
    if x0 is not None and hasattr(x0, "timestamp"):
        return [float(v.timestamp()) if v is not None else float("nan") for v in arr]
    out: list[float] = []
    for v in arr:
        if v is None:
            out.append(float("nan"))
            continue
        f = float(v)
        # Heuristic: epoch ns / us
        if abs(f) > 1e12:
            f /= 1e9
        elif abs(f) > 1e9:
            f /= 1e6
        out.append(f)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "parquet",
        nargs="?",
        default="file-000.parquet",
        type=Path,
        help="Path to parquet file (default: file-000.parquet in cwd)",
    )
    parser.add_argument(
        "--column",
        default="timestamp",
        help="Timestamp column name (default: timestamp)",
    )
    parser.add_argument(
        "--target-hz",
        type=float,
        default=30.0,
        help="Expected frame rate in Hz (default: 30)",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=0.05,
        help="Relative tolerance on interval vs 1/target_hz (default: 0.05 = 5%%)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not open the Δt vs frame_index figure",
    )
    parser.add_argument(
        "--save-plot",
        type=Path,
        default=None,
        metavar="PATH",
        help="Save the figure to PATH (e.g. delta_t.png)",
    )
    args = parser.parse_args()

    path = args.parquet.resolve()
    if not path.is_file():
        raise SystemExit(f"File not found: {path}")

    table = pq.read_table(path)
    if args.column not in table.column_names:
        raise SystemExit(f"Column {args.column!r} not found. Columns: {table.column_names}")

    t = column_to_seconds(table.column(args.column))
    if any(x != x for x in t):  # NaN check
        raise SystemExit("Timestamp column contains NaN after conversion.")

    if len(t) < 2:
        raise SystemExit("Need at least 2 rows to estimate interval.")

    dt = [t[i + 1] - t[i] for i in range(len(t) - 1)]
    expected = 1.0 / args.target_hz
    median_dt = statistics.median(dt)
    mean_dt = statistics.mean(dt)
    std_dt = statistics.pstdev(dt) if len(dt) > 1 else 0.0
    inferred_hz = 1.0 / median_dt if median_dt > 0 else float("nan")

    tol = args.rtol * abs(expected)
    frac_ok = sum(1 for d in dt if abs(d - expected) <= tol) / len(dt)

    print(f"File: {path}")
    print(f"Rows: {len(t)}, column: {args.column!r}")
    print(f"Target: {args.target_hz} Hz (interval ≈ {expected:.6f} s)")
    print()
    print(f"Δt median: {median_dt:.6f} s  →  inferred rate ≈ {inferred_hz:.3f} Hz")
    print(f"Δt mean:   {mean_dt:.6f} s, std: {std_dt:.6f} s")
    print(f"Δt min/max: {min(dt):.6f} / {max(dt):.6f} s")
    print(
        f"Fraction of consecutive pairs within ~{args.rtol*100:.1f}% of {expected:.6f} s: "
        f"{frac_ok*100:.2f}%"
    )

    hz_ok = abs(inferred_hz - args.target_hz) / args.target_hz <= args.rtol
    if frac_ok > 0.99 and hz_ok:
        print(f"\nConclusion: timestamps are consistent with ~{args.target_hz:.0f} Hz.")
    else:
        print("\nConclusion: timestamps do not consistently match the target rate (see stats above).")

    if not args.no_plot or args.save_plot is not None:
        if plt is None:
            raise SystemExit(
                "Plotting needs matplotlib. Install with: pip install matplotlib\n"
                "Or pass --no-plot to skip the figure.\n"
            ) from _matplotlib_import_error

        # Δt between frame k and k+1 is plotted at x=k (frame index of interval start)
        frame_index = list(range(len(dt)))
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.scatter(frame_index, dt, s=8, c="tab:blue", alpha=0.7, label=r"$\Delta t$")
        ax.axhline(expected, color="tab:orange", linestyle="--", linewidth=1, label=f"1/{args.target_hz:g} s (target)")
        lower = expected * (1.0 - args.rtol)
        upper = expected * (1.0 + args.rtol)
        ax.axhline(lower, color="tab:green", linestyle=":", linewidth=1, label=f"-{args.rtol*100:.1f}% bound")
        ax.axhline(upper, color="tab:red", linestyle=":", linewidth=1, label=f"+{args.rtol*100:.1f}% bound")
        ax.set_xlabel("frame_index (start of interval)")
        ax.set_ylabel("Δt (s)")
        ax.set_title(f"{path.name}: timestamp interval per frame pair")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        if args.save_plot is not None:
            fig.savefig(args.save_plot, dpi=150)
            print(f"\nFigure saved to {args.save_plot.resolve()}")
        if not args.no_plot:
            plt.show()
        else:
            plt.close(fig)


if __name__ == "__main__":
    main()
