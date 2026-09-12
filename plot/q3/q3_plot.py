#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
2026 高教社杯 A 题“药材的烘干问题”——问题 3 论文绘图脚本

一次运行生成 3 张核心论文图：

图1：全域最大水分浓度随时间变化及烘干终止事件
     (a) 全过程；(b) 临界时刻局部放大
图2：不同时刻药材径向水分浓度分布
图3：不同径向位置达到临界含水率 C=0.15 kg/kg 的时间

推荐放置位置：
    26MathModel/
    ├── file/附件1.xlsx
    ├── q3/q3_solver.py
    ├── output/q3/result3.xlsx
    └── plot/
        ├── q1/...
        ├── q2/q2_plot.py
        └── q3/q3_plot.py   <- 本脚本

从项目根目录运行：
    python plot/q3/q3_plot.py

默认输出：
    plot/output/q3/
        q3_fig1_drying_event.png
        q3_fig1_drying_event.pdf
        q3_fig2_radial_profiles.png
        q3_fig2_radial_profiles.pdf
        q3_fig3_threshold_time.png
        q3_fig3_threshold_time.pdf

说明：
1. 为避免 result3.xlsx 四位小数造成 C=0.15 阈值时刻的量化误差，
   本脚本直接复用 q3/q3_solver.py 的高精度 BDF 稠密解，不从 Excel 反推临界时刻。
2. 默认使用 N=1280，与问题 3 正式细网格计算保持一致。
3. 预览时可使用 --grid-intervals 320 或 640；论文最终图建议使用 1280。
4. 所有图同时保存 600 dpi PNG 与 PDF，便于直接插入 Word 论文。
5. 图1包含两个子图，(a)、(b) 编号统一置于子图下方，与问题1/2绘图风格一致。
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator
from scipy.optimize import brentq


# ============================================================
# 0. 路径与默认配置
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
# q3_plot.py 固定放在 26MathModel/plot/q3/ 下：
# SCRIPT_DIR.parents[0] = .../26MathModel/plot
# SCRIPT_DIR.parents[1] = .../26MathModel
PROJECT_ROOT = SCRIPT_DIR.parents[1]

DEFAULT_ATTACHMENT1 = "file/附件1.xlsx"
DEFAULT_SOLVER = "q3/q3_solver.py"
DEFAULT_OUTPUT_DIR = "plot/output/q3"

CRITICAL_MOISTURE = 0.15

# 图2选择的代表性时刻。最后还会自动追加精确烘干结束时刻。
PROFILE_HOURS = [6.0, 18.0, 30.0, 42.0, 54.0]


# ============================================================
# 1. 路径与绘图风格
# ============================================================


def resolve_input_path(path_str: str) -> Path:
    """依次尝试当前工作目录、项目根目录、脚本目录。"""
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p

    candidates = [
        Path.cwd() / p,
        PROJECT_ROOT / p,
        SCRIPT_DIR / p,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return (PROJECT_ROOT / p).resolve()


def resolve_output_path(path_str: str) -> Path:
    """相对输出路径统一以项目根目录为基准。"""
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def configure_matplotlib() -> None:
    """与 plot/q1、plot/q2 保持接近的中文论文绘图风格。"""
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    candidates = [
        "PingFang SC",          # macOS
        "Microsoft YaHei",     # Windows
        "SimHei",              # Windows / Linux
        "Noto Sans CJK SC",    # Linux
        "Source Han Sans SC",  # 思源黑体
        "Arial Unicode MS",
        "DejaVu Sans",
    ]

    for font_name in candidates:
        if font_name in available_fonts:
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            break

    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.0,
            "figure.dpi": 120,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "stix",
        }
    )


def style_axis(ax: plt.Axes) -> None:
    """统一坐标轴样式。"""
    ax.grid(True, linestyle="--", linewidth=0.55, alpha=0.35)
    ax.tick_params(direction="in", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def add_panel_labels(axes: np.ndarray | list[plt.Axes]) -> None:
    """把 (a)、(b)、... 放在各子图下方。"""
    axes_flat = np.asarray(axes, dtype=object).ravel()
    for i, ax in enumerate(axes_flat):
        label = f"({chr(ord('a') + i)})"
        ax.text(
            0.5,
            -0.245,
            label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=11.0,
            clip_on=False,
        )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    """同时保存 600 dpi PNG 与 PDF。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"

    fig.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")

    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")


# ============================================================
# 2. 动态加载并复用 q3_solver.py
# ============================================================


def load_solver_module(path: Path) -> ModuleType:
    """从项目中的 q3/q3_solver.py 动态加载求解模块。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到问题3求解器：{path}")

    module_name = "q3_solver_for_plot"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载问题3求解器：{path}")

    module = importlib.util.module_from_spec(spec)
    # dataclass 在装饰阶段会通过 sys.modules 查找所属模块，因此先注册。
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def solve_high_precision(
    solver: ModuleType,
    attachment1_path: Path,
    grid_intervals: int,
    stretch_power: float,
    max_step_s: float,
    h_heat: float,
    h_mass: float,
    plateau_window_s: float,
    plateau_temperature: float | None,
    plateau_moisture: float | None,
):
    """按 q3_solver.py 的正式模型重新求一次高精度细网格解。"""
    boundary = solver.read_attachment1(
        path=attachment1_path,
        plateau_window_s=plateau_window_s,
        plateau_temperature_c=plateau_temperature,
        plateau_moisture=plateau_moisture,
    )

    config = solver.SolverConfig(
        heat_transfer_coefficient=h_heat,
        mass_transfer_coefficient=h_mass,
        grid_intervals=grid_intervals,
        stretch_power=stretch_power,
        max_hours=96.0,
        max_step_s=max_step_s,
    )

    model = solver.DryingModel(config, boundary)

    print("\n开始求解问题3高精度绘图数据……")
    print(
        f"  N={config.grid_intervals}，节点数={model.n_nodes}，"
        f"p={config.stretch_power:g}，max_step={config.max_step_s:g} s"
    )
    print(
        "  4 h 后边界："
        f"T∞={boundary.plateau_temperature_c:.9f} °C，"
        f"C∞={boundary.plateau_moisture:.9f} kg/kg"
    )

    result = solver.solve_until_dry(model)
    solver.validate_solution(result)

    print(
        f"  连续事件时间：{result.drying_time_s:.6f} s "
        f"= {result.drying_time_s / 3600.0:.9f} h"
    )
    print(
        f"  BDF 内部时间节点：{len(result.ode_solution.t)}，"
        f"函数计算次数：{result.ode_solution.nfev}，"
        f"求解耗时：{result.wall_time_s:.2f} s"
    )
    return result


# ============================================================
# 3. 高精度场数据辅助函数
# ============================================================


def evaluate_moisture_on_mesh(result, times_s: np.ndarray) -> np.ndarray:
    """返回 shape=(n_time, n_node) 的高精度含水率矩阵。"""
    times_s = np.asarray(times_s, dtype=float)
    if times_s.ndim != 1:
        raise ValueError("times_s 必须是一维数组。")
    if np.any(times_s < -1e-12) or np.any(times_s > result.drying_time_s + 1e-8):
        raise ValueError("绘图时间超出主 BDF 稠密解范围。")

    states = result.ode_solution.sol(times_s)
    n = result.model.n_nodes
    moisture = np.asarray(states[n:, :].T, dtype=float)
    return moisture


def interpolate_profiles_to_radii(
    mesh_r_m: np.ndarray,
    moisture_on_mesh: np.ndarray,
    target_r_m: np.ndarray,
) -> np.ndarray:
    """
    将多个时刻的网格含水率插值到目标半径。

    输入：
        moisture_on_mesh: shape=(n_time, n_mesh)
    输出：
        shape=(n_time, n_target)
    """
    target_r_m = np.asarray(target_r_m, dtype=float)
    if target_r_m[0] < -1e-15 or target_r_m[-1] > mesh_r_m[-1] + 1e-15:
        raise ValueError("目标半径超出药材范围。")

    return np.vstack(
        [
            np.interp(target_r_m, mesh_r_m, row)
            for row in moisture_on_mesh
        ]
    )


def moisture_at_radius(result, t_s: float, r_m: float) -> float:
    """利用 BDF 稠密解计算任意时刻、任意半径处的高精度含水率。"""
    state = np.asarray(result.ode_solution.sol(float(t_s)), dtype=float)
    n = result.model.n_nodes
    moisture = state[n:]
    return float(np.interp(r_m, result.model.mesh.r_m, moisture))


# ============================================================
# 4. 图1：全域最大含水率 + 连续终止事件
# ============================================================


def plot_drying_event(result, output_dir: Path) -> None:
    critical = float(result.model.config.critical_moisture)
    drying_h = result.drying_time_s / 3600.0
    n = result.model.n_nodes

    # 全过程使用等时间采样；额外保证精确事件时刻在数组末端。
    times_s = np.linspace(0.0, result.drying_time_s, 1800)
    states = result.ode_solution.sol(times_s)
    c_max = np.max(states[n:, :], axis=0)
    t_h = times_s / 3600.0

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.85))
    fig.subplots_adjust(left=0.075, right=0.97, bottom=0.24, top=0.89, wspace=0.28)

    # (a) 全过程
    axes[0].plot(t_h, c_max, linewidth=1.65, label=r"$C_{\max}(t)$")
    axes[0].axhline(
        critical,
        linestyle="--",
        linewidth=1.15,
        color="black",
        label=r"临界值 $C_{\mathrm{cr}}=0.15$",
    )
    axes[0].axvline(
        drying_h,
        linestyle=":",
        linewidth=1.15,
        color="0.35",
    )
    axes[0].scatter([drying_h], [critical], s=28, zorder=4)
    axes[0].set_title("全烘干过程")
    axes[0].set_xlabel(r"时间 $t/\mathrm{h}$")
    axes[0].set_ylabel(
        r"全域最大水分浓度 $C_{\max}/(\mathrm{kg}\cdot\mathrm{kg}^{-1})$"
    )
    axes[0].set_xlim(0.0, drying_h * 1.015)
    axes[0].set_ylim(bottom=0.0)
    axes[0].xaxis.set_major_locator(MaxNLocator(7))
    axes[0].yaxis.set_major_locator(MaxNLocator(7))
    axes[0].legend(loc="upper right", frameon=True)
    style_axis(axes[0])

    # (b) 临界区局部放大
    zoom_start_h = max(0.0, drying_h - 5.0)
    zoom_mask = t_h >= zoom_start_h
    t_zoom = t_h[zoom_mask]
    c_zoom = c_max[zoom_mask]

    axes[1].plot(t_zoom, c_zoom, linewidth=1.75, label=r"$C_{\max}(t)$")
    axes[1].axhline(
        critical,
        linestyle="--",
        linewidth=1.15,
        color="black",
        label=r"$C_{\mathrm{cr}}=0.15$",
    )
    axes[1].axvline(
        drying_h,
        linestyle=":",
        linewidth=1.15,
        color="0.35",
    )
    axes[1].scatter([drying_h], [critical], s=32, zorder=4)

    y_min = min(float(np.min(c_zoom)), critical)
    y_max = max(float(np.max(c_zoom)), critical)
    padding = max((y_max - y_min) * 0.16, 0.0015)
    axes[1].set_ylim(max(0.0, y_min - padding), y_max + padding)
    axes[1].set_xlim(zoom_start_h, drying_h + 0.12)

    axes[1].annotate(
        f"烘干结束\n$t={drying_h:.4f}$ h",
        xy=(drying_h, critical),
        xytext=(0.58, 0.72),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "linewidth": 0.9},
        ha="left",
        va="center",
        fontsize=9.2,
    )
    axes[1].set_title("临界时刻局部放大")
    axes[1].set_xlabel(r"时间 $t/\mathrm{h}$")
    axes[1].set_ylabel(
        r"全域最大水分浓度 $C_{\max}/(\mathrm{kg}\cdot\mathrm{kg}^{-1})$"
    )
    axes[1].xaxis.set_major_locator(MaxNLocator(6))
    axes[1].yaxis.set_major_locator(MaxNLocator(6))
    axes[1].legend(loc="upper right", frameon=True)
    style_axis(axes[1])

    add_panel_labels(axes)

    print("\n保存图1：全域最大含水率及烘干终止事件")
    save_figure(fig, output_dir, "q3_fig1_drying_event")
    plt.close(fig)

    print("========== 图1关键结果 ==========")
    print(f"烘干阈值：{critical:.6f} kg/kg")
    print(f"连续烘干结束时刻：{drying_h:.9f} h")
    print(f"事件时刻 Cmax：{np.max(result.event_state[n:]):.10f} kg/kg")


# ============================================================
# 5. 图2：典型时刻径向水分浓度分布
# ============================================================


def plot_radial_profiles(result, output_dir: Path) -> None:
    critical = float(result.model.config.critical_moisture)
    drying_h = result.drying_time_s / 3600.0

    profile_hours = [h for h in PROFILE_HOURS if h < drying_h - 1e-8]
    profile_hours.append(drying_h)
    profile_hours = np.asarray(profile_hours, dtype=float)
    times_s = profile_hours * 3600.0

    moisture_mesh = evaluate_moisture_on_mesh(result, times_s)

    r_cm = np.linspace(0.0, result.model.config.radius_m * 100.0, 201)
    r_m = r_cm / 100.0
    profiles = interpolate_profiles_to_radii(
        result.model.mesh.r_m,
        moisture_mesh,
        r_m,
    )

    fig, ax = plt.subplots(figsize=(7.2, 5.05))
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.14, top=0.92)

    for i, (hour, profile) in enumerate(zip(profile_hours, profiles)):
        if i == len(profile_hours) - 1:
            label = f"烘干结束 {hour:.4f} h"
            ax.plot(r_cm, profile, linewidth=2.0, label=label)
        else:
            ax.plot(r_cm, profile, linewidth=1.55, label=f"{hour:g} h")

    ax.axhline(
        critical,
        linestyle="--",
        linewidth=1.05,
        color="black",
        label=r"临界值 $C_{\mathrm{cr}}=0.15$",
    )

    ax.set_xlabel(r"到药材中心距离 $r/\mathrm{cm}$")
    ax.set_ylabel(r"水分浓度 $C/(\mathrm{kg}\cdot\mathrm{kg}^{-1})$")
    ax.set_xlim(0.0, 2.0)
    ax.set_xticks(np.arange(0.0, 2.01, 0.5))
    ax.set_ylim(bottom=0.0)
    ax.yaxis.set_major_locator(MaxNLocator(7))
    ax.legend(loc="upper right", ncol=2, frameon=True)
    style_axis(ax)

    print("\n保存图2：典型时刻径向水分浓度分布")
    save_figure(fig, output_dir, "q3_fig2_radial_profiles")
    plt.close(fig)

    print("========== 图2结束时径向含水率 ==========")
    # 输出论文最常用的 0/0.5/1/1.5/2 cm 五个位置。
    report_r_cm = np.arange(0.0, 2.01, 0.5)
    end_profile = np.interp(report_r_cm, r_cm, profiles[-1])
    for rr, cc in zip(report_r_cm, end_profile):
        print(f"r={rr:.1f} cm: C={cc:.8f} kg/kg")


# ============================================================
# 6. 图3：不同半径位置达到 C=0.15 的时间
# ============================================================


def threshold_times_by_radius(
    result,
    target_r_cm: np.ndarray,
    coarse_step_s: float = 300.0,
) -> np.ndarray:
    """
    计算每个径向位置首次满足 C(r,t) <= Ccrit 的时间。

    做法：
    1. 先用较粗时间采样寻找“首次过阈值”的时间括区间；
    2. 再利用 BDF 稠密解 + Brent 根求解，将该位置的过阈值时刻精确定位。

    这样既避免使用 result3.xlsx 四位小数导致误判，又保证找的是“首次”过阈值。
    """
    critical = float(result.model.config.critical_moisture)
    dry_s = float(result.drying_time_s)
    mesh_r = result.model.mesh.r_m
    target_r_m = np.asarray(target_r_cm, dtype=float) / 100.0

    # 粗采样必须包含 0 和精确事件终点。
    coarse_times = np.arange(0.0, dry_s, coarse_step_s, dtype=float)
    if coarse_times.size == 0 or coarse_times[0] != 0.0:
        coarse_times = np.insert(coarse_times, 0, 0.0)
    if coarse_times[-1] < dry_s - 1e-10:
        coarse_times = np.append(coarse_times, dry_s)
    else:
        coarse_times[-1] = dry_s

    moisture_mesh = evaluate_moisture_on_mesh(result, coarse_times)
    moisture_target = interpolate_profiles_to_radii(
        mesh_r,
        moisture_mesh,
        target_r_m,
    )  # (n_time, n_radius)

    crossing_s = np.empty(target_r_m.size, dtype=float)

    for j, r_m in enumerate(target_r_m):
        series = moisture_target[:, j] - critical
        hit = np.flatnonzero(series <= 0.0)
        if hit.size == 0:
            raise RuntimeError(
                f"r={target_r_cm[j]:.6f} cm 在烘干结束前未检测到阈值交点。"
            )

        k = int(hit[0])
        if k == 0:
            crossing_s[j] = coarse_times[0]
            continue

        t_left = float(coarse_times[k - 1])
        t_right = float(coarse_times[k])
        f_left = float(series[k - 1])
        f_right = float(series[k])

        if abs(f_right) < 1e-12:
            crossing_s[j] = t_right
            continue
        if abs(f_left) < 1e-12:
            crossing_s[j] = t_left
            continue

        if f_left * f_right > 0.0:
            raise RuntimeError(
                f"r={target_r_cm[j]:.6f} cm 的阈值括区间异常："
                f"f_left={f_left:.3e}, f_right={f_right:.3e}。"
            )

        def root_function(t_s: float) -> float:
            return moisture_at_radius(result, t_s, r_m) - critical

        crossing_s[j] = brentq(
            root_function,
            t_left,
            t_right,
            xtol=1e-7,
            rtol=1e-12,
            maxiter=100,
        )

    return crossing_s


def plot_threshold_time(
    result,
    output_dir: Path,
    radius_step_cm: float,
) -> None:
    if radius_step_cm <= 0.0 or radius_step_cm > 0.5:
        raise ValueError("--threshold-radius-step 建议取 (0, 0.5] cm。")

    radius_max_cm = result.model.config.radius_m * 100.0
    n_intervals = int(round(radius_max_cm / radius_step_cm))
    r_cm = np.linspace(0.0, radius_max_cm, n_intervals + 1)

    crossing_s = threshold_times_by_radius(result, r_cm, coarse_step_s=300.0)
    crossing_h = crossing_s / 3600.0
    drying_h = result.drying_time_s / 3600.0

    fig, ax = plt.subplots(figsize=(7.2, 5.05))
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.14, top=0.92)

    ax.plot(r_cm, crossing_h, linewidth=1.8)
    ax.scatter([0.0], [crossing_h[0]], s=30, zorder=4)
    ax.axhline(
        drying_h,
        linestyle=":",
        linewidth=1.0,
        color="0.35",
    )
    ax.annotate(
        f"最后达标位置：圆心\n$t={crossing_h[0]:.4f}$ h",
        xy=(0.0, crossing_h[0]),
        xytext=(0.27, 0.86),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "linewidth": 0.9},
        ha="left",
        va="top",
        fontsize=9.2,
    )

    ax.set_xlabel(r"到药材中心距离 $r/\mathrm{cm}$")
    ax.set_ylabel(r"首次达到 $C=0.15$ 的时间 $t_{0.15}/\mathrm{h}$")
    ax.set_xlim(0.0, radius_max_cm)
    ax.set_xticks(np.arange(0.0, radius_max_cm + 0.01, 0.5))
    ax.set_ylim(bottom=0.0)
    ax.yaxis.set_major_locator(MaxNLocator(7))
    style_axis(ax)

    print("\n保存图3：不同径向位置达到临界含水率的时间")
    save_figure(fig, output_dir, "q3_fig3_threshold_time")
    plt.close(fig)

    print("========== 图3典型位置达标时间 ==========")
    report_r_cm = np.arange(0.0, radius_max_cm + 0.01, 0.5)
    report_t_h = np.interp(report_r_cm, r_cm, crossing_h)
    for rr, tt in zip(report_r_cm, report_t_h):
        print(f"r={rr:.1f} cm: t_0.15={tt:.6f} h")

    # 数值逻辑核验：中心应为最后达到阈值的位置。
    index_last = int(np.argmax(crossing_h))
    print(
        f"最晚达标位置：r={r_cm[index_last]:.6f} cm，"
        f"t={crossing_h[index_last]:.9f} h"
    )


# ============================================================
# 7. 主程序
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "问题3论文绘图：生成烘干终止事件、典型径向剖面、"
            "各径向位置阈值达标时间 3 张图。"
        )
    )
    parser.add_argument(
        "--attachment1",
        default=DEFAULT_ATTACHMENT1,
        help=f"附件1路径，默认：{DEFAULT_ATTACHMENT1}",
    )
    parser.add_argument(
        "--solver",
        default=DEFAULT_SOLVER,
        help=f"问题3求解器路径，默认：{DEFAULT_SOLVER}",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"图片输出目录，默认：{DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--grid-intervals",
        type=int,
        default=1280,
        help="绘图时重新求解所用径向区间数；论文最终图推荐 1280。",
    )
    parser.add_argument(
        "--stretch-power",
        type=float,
        default=2.0,
        help="表面加密幂指数，默认 2。",
    )
    parser.add_argument(
        "--max-step",
        type=float,
        default=60.0,
        help="BDF 最大时间步/s，默认 60。",
    )
    parser.add_argument(
        "--plateau-window-seconds",
        type=float,
        default=3600.0,
        help="附件末尾多少秒的均值作为 4 h 后平台，默认 3600。",
    )
    parser.add_argument(
        "--plateau-temperature",
        type=float,
        default=None,
        help="4 h 后烘房温度/°C；默认沿用 q3_solver.py 的末1h均值。",
    )
    parser.add_argument(
        "--plateau-moisture",
        type=float,
        default=None,
        help="4 h 后烘房水分浓度；默认沿用 q3_solver.py 的末1h均值。",
    )
    parser.add_argument(
        "--h-heat",
        type=float,
        default=25.0,
        help="对流换热系数 W/(m²·K)，默认 25。",
    )
    parser.add_argument(
        "--h-mass",
        type=float,
        default=8.0e-7,
        help="对流传质系数 m/s，默认 8e-7。",
    )
    parser.add_argument(
        "--threshold-radius-step",
        type=float,
        default=0.02,
        help="图3径向采样间隔/cm，默认 0.02（共约101个位置）。",
    )
    args = parser.parse_args()

    if args.grid_intervals < 40:
        raise ValueError("--grid-intervals 不应小于 40。")
    if args.max_step <= 0.0:
        raise ValueError("--max-step 必须为正数。")

    configure_matplotlib()

    attachment1_path = resolve_input_path(args.attachment1)
    solver_path = resolve_input_path(args.solver)
    output_dir = resolve_output_path(args.output_dir)

    print("========== 问题3论文绘图 ==========")
    print(f"项目根目录：{PROJECT_ROOT}")
    print(f"附件1      ：{attachment1_path}")
    print(f"q3求解器   ：{solver_path}")
    print(f"图片目录   ：{output_dir}")

    solver = load_solver_module(solver_path)
    result = solve_high_precision(
        solver=solver,
        attachment1_path=attachment1_path,
        grid_intervals=args.grid_intervals,
        stretch_power=args.stretch_power,
        max_step_s=args.max_step,
        h_heat=args.h_heat,
        h_mass=args.h_mass,
        plateau_window_s=args.plateau_window_seconds,
        plateau_temperature=args.plateau_temperature,
        plateau_moisture=args.plateau_moisture,
    )

    plot_drying_event(result, output_dir)
    plot_radial_profiles(result, output_dir)
    plot_threshold_time(
        result,
        output_dir,
        radius_step_cm=args.threshold_radius_step,
    )

    print("\n========== 绘图完成 ==========")
    print("图1：全域最大水分浓度及烘干终止事件")
    print("图2：不同时刻径向水分浓度分布")
    print("图3：不同径向位置首次达到 C=0.15 的时间")
    print(f"输出目录：{output_dir}")


if __name__ == "__main__":
    main()
