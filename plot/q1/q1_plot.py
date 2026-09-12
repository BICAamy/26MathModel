#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
2026 高教社杯 A题：药材的烘干问题 —— 问题1绘图脚本

一次运行自动生成 3 组核心论文图，每张图包含 2 个子图：

图2：温度场 / 水分浓度场二维时空分布图
图3：典型时刻温度 / 水分浓度径向分布曲线
图4：FVM+RK45 与 FDM+FTCS 的逐时刻全场最大绝对误差

推荐放置位置：
    26MathModel/
    ├── file/附件1.xlsx
    ├── output/q1/result1.xlsx
    └── q1/
        ├── q1_solver_fdm.py
        └── q1_plot.py

从项目根目录运行：
    python q1/q1_plot.py

默认输出：
    output/q1/figures/
        q1_fig2_spatiotemporal_fields.png
        q1_fig2_spatiotemporal_fields.pdf
        q1_fig3_radial_profiles.png
        q1_fig3_radial_profiles.pdf
        q1_fig4_validation_errors.png
        q1_fig4_validation_errors.pdf

说明：
1. 图2、图3直接读取主算法 result1.xlsx。
2. 图4优先读取 output/q1/result1_fdm_refined.xlsx；若不存在，则自动调用
   q1_solver_fdm.py，以 dr=0.025 cm、dt=0.05 s 重新计算并缓存。
   这一组参数与 q1.docx 中“更细网格验证”的误差统计一致。
3. 所有图同时输出 600 dpi PNG 和 PDF，适合插入 Word 论文。
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator
from openpyxl import load_workbook
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Songti SC']
plt.rcParams['axes.unicode_minus'] = False   # 负号正常显示

# ============================================================
# 0. 默认配置
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name.lower() == "q1" else SCRIPT_DIR

# 图3只选 4 个代表时刻，避免 7 条曲线过于拥挤。
PROFILE_TIMES = [100, 600, 1200, 1800]

# 与论文中第二算法“更细空间网格 + 更小时间步长”的验证设置一致。
DEFAULT_FDM_DR_CM = 0.025
DEFAULT_FDM_DT = 0.05


# ============================================================
# 1. 路径与绘图风格
# ============================================================

def resolve_input_path(path_str: str) -> Path:
    """解析输入文件：依次尝试当前工作目录、项目根目录、脚本目录。"""
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

    # 返回最符合项目结构的路径，后续报错时更容易定位。
    return (PROJECT_ROOT / p).resolve()


def resolve_output_path(path_str: str) -> Path:
    """输出路径统一以项目根目录为基准。"""
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def configure_matplotlib() -> None:
    """配置适合中文数模论文的 Matplotlib 风格。"""
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    candidates = [
        "PingFang SC",          # macOS
        "Microsoft YaHei",     # Windows
        "SimHei",              # Windows / Linux 常见中文字体
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
            "legend.fontsize": 9.5,
            "figure.dpi": 120,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "stix",
        }
    )


def style_axis(ax: plt.Axes) -> None:
    """统一坐标轴风格。"""
    ax.grid(True, linestyle="--", linewidth=0.55, alpha=0.35)
    ax.tick_params(direction="in", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


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
# 2. 读取 result1.xlsx
# ============================================================

def _read_field_sheet(ws) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    读取题目模板形式工作表：
        A列：时间/s
        第1行 B...：到中心距离/cm
        数据区域：场变量
    """
    rows = list(ws.iter_rows(values_only=True))
    if len(rows) < 2:
        raise ValueError(f"工作表 {ws.title!r} 没有有效数据。")

    r_cm = np.asarray(
        [float(v) for v in rows[0][1:] if v is not None],
        dtype=float,
    )
    if r_cm.size == 0:
        raise ValueError(f"工作表 {ws.title!r} 第1行没有读取到空间网格。")

    times = []
    data = []
    n_r = len(r_cm)

    for row in rows[1:]:
        if row[0] is None:
            continue
        vals = row[1: 1 + n_r]
        if len(vals) < n_r or any(v is None for v in vals):
            continue

        times.append(float(row[0]))
        data.append([float(v) for v in vals])

    if not data:
        raise ValueError(f"工作表 {ws.title!r} 没有读取到有效数值矩阵。")

    return (
        np.asarray(times, dtype=float),
        r_cm,
        np.asarray(data, dtype=float),
    )


def read_result1(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """读取包含“温度”“水分浓度”两个工作表的结果文件。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到结果文件：{path}")

    wb = load_workbook(path, data_only=True, read_only=True)
    try:
        required = {"温度", "水分浓度"}
        if not required.issubset(set(wb.sheetnames)):
            raise ValueError(
                f"{path.name} 必须包含“温度”和“水分浓度”两个工作表，"
                f"当前工作表为：{wb.sheetnames}"
            )

        t_T, r_T, T = _read_field_sheet(wb["温度"])
        t_C, r_C, C = _read_field_sheet(wb["水分浓度"])
    finally:
        wb.close()

    if not np.array_equal(t_T, t_C):
        raise ValueError("温度表和水分浓度表的时间网格不一致。")
    if not np.allclose(r_T, r_C, atol=1e-12):
        raise ValueError("温度表和水分浓度表的空间网格不一致。")
    if T.shape != C.shape:
        raise ValueError("温度场和水分场矩阵维度不一致。")

    return t_T, r_T, T, C


# ============================================================
# 3. 第二算法 FDM 结果：优先读缓存，不存在则自动计算
# ============================================================

def import_module_from_file(module_path: Path):
    if not module_path.exists():
        raise FileNotFoundError(
            f"找不到第二算法脚本：{module_path}\n"
            "请确认 q1_solver_fdm.py 与 q1_plot.py 位于同一项目中，"
            "或通过 --fdm-script 指定路径。"
        )

    spec = importlib.util.spec_from_file_location("q1_solver_fdm_for_plot", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法动态导入：{module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_fdm_result(
    fdm_result_path: Path,
    fdm_script_path: Path,
    attachment1_path: Path,
    dr_cm: float,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """获取 refined FDM 结果；若无缓存则调用原 q1_solver_fdm.py 计算。"""
    if fdm_result_path.exists():
        print(f"读取已有 FDM 验证结果：{fdm_result_path}")
        return read_result1(fdm_result_path)

    if not attachment1_path.exists():
        raise FileNotFoundError(f"找不到附件1：{attachment1_path}")

    print("未找到 refined FDM 缓存，开始调用 q1_solver_fdm.py 计算……")
    print(f"  dr = {dr_cm:g} cm")
    print(f"  dt = {dt:g} s")

    mod = import_module_from_file(fdm_script_path)

    needed = ["solve_q1_fdm", "resample_to_required_grid"]
    for name in needed:
        if not hasattr(mod, name):
            raise AttributeError(f"{fdm_script_path.name} 中缺少函数 {name}()。")

    times, r_solver_cm, T_solver, C_solver = mod.solve_q1_fdm(
        attachment1_path,
        dr_cm=dr_cm,
        dt=dt,
    )

    r_out_cm, T_out = mod.resample_to_required_grid(r_solver_cm, T_solver)
    _, C_out = mod.resample_to_required_grid(r_solver_cm, C_solver)

    times = np.asarray(times, dtype=float)
    r_out_cm = np.asarray(r_out_cm, dtype=float)
    T_out = np.asarray(T_out, dtype=float)
    C_out = np.asarray(C_out, dtype=float)

    # 若原脚本提供写文件函数，则顺便缓存，下一次绘图无需重新计算。
    if hasattr(mod, "write_result1"):
        fdm_result_path.parent.mkdir(parents=True, exist_ok=True)
        mod.write_result1(fdm_result_path, times.astype(int), r_out_cm, T_out, C_out)
        print(f"已缓存 refined FDM 结果：{fdm_result_path}")

    return times, r_out_cm, T_out, C_out


# ============================================================
# 4. 图2：二维时空分布图
# ============================================================

def plot_spatiotemporal_fields(
    times: np.ndarray,
    r_cm: np.ndarray,
    T: np.ndarray,
    C: np.ndarray,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.65), constrained_layout=True)

    # 温度场
    levels_T = np.linspace(np.nanmin(T), np.nanmax(T), 60)
    cf_T = axes[0].contourf(
        times,
        r_cm,
        T.T,
        levels=levels_T,
        cmap="magma",
        extend="both",
    )
    cbar_T = fig.colorbar(cf_T, ax=axes[0], pad=0.02)
    cbar_T.set_label(r"温度 $T/{}^\circ\mathrm{C}$")

    axes[0].set_title("(a) 药材内部温度场")
    axes[0].set_xlabel(r"时间 $t/\mathrm{s}$")
    axes[0].set_ylabel(r"到药材中心距离 $r/\mathrm{cm}$")
    axes[0].set_xlim(times[0], times[-1])
    axes[0].set_ylim(r_cm[0], r_cm[-1])
    axes[0].set_xticks(np.arange(0, 1801, 300))
    axes[0].set_yticks(np.arange(0, 2.01, 0.5))
    axes[0].tick_params(direction="in")

    # 水分场
    levels_C = np.linspace(np.nanmin(C), np.nanmax(C), 60)
    cf_C = axes[1].contourf(
        times,
        r_cm,
        C.T,
        levels=levels_C,
        cmap="viridis",
        extend="both",
    )
    cbar_C = fig.colorbar(cf_C, ax=axes[1], pad=0.02)
    cbar_C.set_label(r"水分浓度 $C/(\mathrm{kg}\cdot\mathrm{kg}^{-1})$")

    axes[1].set_title("(b) 药材内部水分浓度场")
    axes[1].set_xlabel(r"时间 $t/\mathrm{s}$")
    axes[1].set_ylabel(r"到药材中心距离 $r/\mathrm{cm}$")
    axes[1].set_xlim(times[0], times[-1])
    axes[1].set_ylim(r_cm[0], r_cm[-1])
    axes[1].set_xticks(np.arange(0, 1801, 300))
    axes[1].set_yticks(np.arange(0, 2.01, 0.5))
    axes[1].tick_params(direction="in")

    print("保存图2：温度场 / 水分浓度场二维时空分布图")
    save_figure(fig, output_dir, "q1_fig2_spatiotemporal_fields")
    plt.close(fig)


# ============================================================
# 5. 图3：典型时刻径向分布曲线
# ============================================================

def find_time_index(times: np.ndarray, target: float) -> int:
    idx = int(np.argmin(np.abs(times - target)))
    if not np.isclose(times[idx], target, atol=1e-9):
        raise ValueError(f"结果中不存在目标时刻 t={target:g} s。")
    return idx


def plot_radial_profiles(
    times: np.ndarray,
    r_cm: np.ndarray,
    T: np.ndarray,
    C: np.ndarray,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.65), constrained_layout=True)

    linestyles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D"]

    for i, t in enumerate(PROFILE_TIMES):
        idx = find_time_index(times, t)
        axes[0].plot(
            r_cm,
            T[idx],
            linestyle=linestyles[i % len(linestyles)],
            marker=markers[i % len(markers)],
            markevery=2,
            markersize=3.8,
            linewidth=1.45,
            label=f"{t} s",
        )
        axes[1].plot(
            r_cm,
            C[idx],
            linestyle=linestyles[i % len(linestyles)],
            marker=markers[i % len(markers)],
            markevery=2,
            markersize=3.8,
            linewidth=1.45,
            label=f"{t} s",
        )

    axes[0].set_title("(a) 典型时刻温度径向分布")
    axes[0].set_xlabel(r"到药材中心距离 $r/\mathrm{cm}$")
    axes[0].set_ylabel(r"温度 $T/{}^\circ\mathrm{C}$")
    axes[0].set_xlim(r_cm[0], r_cm[-1])
    axes[0].set_xticks(np.arange(0, 2.01, 0.25))
    axes[0].legend(frameon=False, ncol=2)
    style_axis(axes[0])

    axes[1].set_title("(b) 典型时刻水分浓度径向分布")
    axes[1].set_xlabel(r"到药材中心距离 $r/\mathrm{cm}$")
    axes[1].set_ylabel(r"水分浓度 $C/(\mathrm{kg}\cdot\mathrm{kg}^{-1})$")
    axes[1].set_xlim(r_cm[0], r_cm[-1])
    axes[1].set_xticks(np.arange(0, 2.01, 0.25))
    axes[1].legend(frameon=False, ncol=2)
    style_axis(axes[1])

    print("保存图3：典型时刻温度 / 水分浓度径向分布")
    save_figure(fig, output_dir, "q1_fig3_radial_profiles")
    plt.close(fig)


# ============================================================
# 6. 图4：两种算法逐时刻全场最大绝对误差
# ============================================================

def check_same_grid(
    t_ref: np.ndarray,
    r_ref: np.ndarray,
    t_cmp: np.ndarray,
    r_cmp: np.ndarray,
) -> None:
    if len(t_ref) != len(t_cmp) or not np.allclose(t_ref, t_cmp, atol=1e-12):
        raise ValueError("FVM 与 FDM 的时间网格不一致，无法逐点比较。")
    if len(r_ref) != len(r_cmp) or not np.allclose(r_ref, r_cmp, atol=1e-12):
        raise ValueError("FVM 与 FDM 的空间网格不一致，无法逐点比较。")


def add_error_annotation(ax: plt.Axes, global_max: float, global_mean: float, unit: str) -> None:
    text = (
        f"全场最大绝对误差 = {global_max:.6g} {unit}\n"
        f"全场平均绝对误差 = {global_mean:.6g} {unit}"
    )
    ax.text(
        0.03,
        0.96,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9.0,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="0.75", alpha=0.9),
    )


def plot_validation_errors(
    times: np.ndarray,
    r_cm: np.ndarray,
    T_fvm: np.ndarray,
    C_fvm: np.ndarray,
    t_fdm: np.ndarray,
    r_fdm: np.ndarray,
    T_fdm: np.ndarray,
    C_fdm: np.ndarray,
    output_dir: Path,
) -> None:
    check_same_grid(times, r_cm, t_fdm, r_fdm)

    abs_T = np.abs(T_fdm - T_fvm)
    abs_C = np.abs(C_fdm - C_fvm)

    # 每一个时刻，在全部 21 个径向位置上取最大绝对误差。
    err_T_t = np.max(abs_T, axis=1)
    err_C_t = np.max(abs_C, axis=1)

    max_T = float(np.max(abs_T))
    mean_T = float(np.mean(abs_T))
    max_C = float(np.max(abs_C))
    mean_C = float(np.mean(abs_C))

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.65), constrained_layout=True)

    line_T, = axes[0].plot(times, err_T_t, linewidth=1.45)
    iT = int(np.argmax(err_T_t))
    axes[0].scatter(
        times[iT], err_T_t[iT],
        s=24, zorder=4, color=line_T.get_color(),
        label=f"最大值：t={times[iT]:g} s",
    )
    axes[0].set_title("(a) 温度场逐时刻最大绝对误差")
    axes[0].set_xlabel(r"时间 $t/\mathrm{s}$")
    axes[0].set_ylabel(r"$\max_r |T_{\mathrm{FDM}}-T_{\mathrm{FVM}}|/{}^\circ\mathrm{C}$")
    axes[0].set_xlim(times[0], times[-1])
    axes[0].set_xticks(np.arange(0, 1801, 300))
    axes[0].yaxis.set_major_locator(MaxNLocator(6))
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    axes[0].legend(frameon=False, loc="lower right")
    add_error_annotation(axes[0], max_T, mean_T, "°C")
    style_axis(axes[0])

    line_C, = axes[1].plot(times, err_C_t, linewidth=1.45)
    iC = int(np.argmax(err_C_t))
    axes[1].scatter(
        times[iC], err_C_t[iC],
        s=24, zorder=4, color=line_C.get_color(),
        label=f"最大值：t={times[iC]:g} s",
    )
    axes[1].set_title("(b) 水分场逐时刻最大绝对误差")
    axes[1].set_xlabel(r"时间 $t/\mathrm{s}$")
    axes[1].set_ylabel(r"$\max_r |C_{\mathrm{FDM}}-C_{\mathrm{FVM}}|/(\mathrm{kg}\cdot\mathrm{kg}^{-1})$")
    axes[1].set_xlim(times[0], times[-1])
    axes[1].set_xticks(np.arange(0, 1801, 300))
    axes[1].yaxis.set_major_locator(MaxNLocator(6))
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    axes[1].legend(frameon=False, loc="lower right")
    add_error_annotation(axes[1], max_C, mean_C, "kg/kg")
    style_axis(axes[1])

    print("保存图4：FVM+RK45 与 FDM+FTCS 的误差验证")
    save_figure(fig, output_dir, "q1_fig4_validation_errors")
    plt.close(fig)

    print("\n========== 两套算法全场误差统计 ==========")
    print(f"温度：最大绝对误差 = {max_T:.8g} °C")
    print(f"温度：平均绝对误差 = {mean_T:.8g} °C")
    print(f"水分：最大绝对误差 = {max_C:.8g} kg/kg")
    print(f"水分：平均绝对误差 = {mean_C:.8g} kg/kg")


# ============================================================
# 7. 主程序
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="问题1论文绘图：一次生成 3 张图，每张 2 个子图。"
    )

    parser.add_argument(
        "--result",
        default="output/q1/result1.xlsx",
        help="主算法 FVM+RK45 的 result1.xlsx",
    )
    parser.add_argument(
        "--attachment",
        default="file/附件1.xlsx",
        help="附件1.xlsx，仅在需要自动计算 FDM 时使用",
    )
    parser.add_argument(
        "--fdm-script",
        default="q1/q1_solver_fdm.py",
        help="第二算法 q1_solver_fdm.py 路径",
    )
    parser.add_argument(
        "--fdm-result",
        default="output/q1/result1_fdm_refined.xlsx",
        help="refined FDM 结果缓存；存在则直接读取，不存在则自动生成",
    )
    parser.add_argument(
        "--output-dir",
        default="output/plot/q1",
        help="图片输出目录",
    )
    parser.add_argument(
        "--fdm-dr-cm",
        type=float,
        default=DEFAULT_FDM_DR_CM,
        help=f"FDM 验证空间步长/cm，默认 {DEFAULT_FDM_DR_CM}",
    )
    parser.add_argument(
        "--fdm-dt",
        type=float,
        default=DEFAULT_FDM_DT,
        help=f"FDM 验证时间步长/s，默认 {DEFAULT_FDM_DT}",
    )

    args = parser.parse_args()

    configure_matplotlib()

    result_path = resolve_input_path(args.result)
    attachment_path = resolve_input_path(args.attachment)
    fdm_script_path = resolve_input_path(args.fdm_script)
    fdm_result_path = resolve_output_path(args.fdm_result)
    output_dir = resolve_output_path(args.output_dir)

    print("========== 问题1论文绘图 ==========")
    print(f"主算法结果：{result_path}")
    print(f"图片目录  ：{output_dir}")

    # 1) 主算法结果
    times, r_cm, T_fvm, C_fvm = read_result1(result_path)
    print(f"读取主算法结果完成：T/C shape = {T_fvm.shape}")
    print(f"时间范围：{times[0]:g} ~ {times[-1]:g} s")
    print(f"空间范围：{r_cm[0]:g} ~ {r_cm[-1]:g} cm")

    # 基本一致性检查：问题1应为 0~1800 s、0~2 cm。
    if times[0] > 0 or times[-1] < 1800:
        raise ValueError("主算法结果必须覆盖 0~1800 s。")
    if r_cm[0] > 0 or r_cm[-1] < 2.0:
        raise ValueError("主算法结果必须覆盖 r=0~2 cm。")

    # 2) 图2：时空场
    plot_spatiotemporal_fields(times, r_cm, T_fvm, C_fvm, output_dir)

    # 3) 图3：典型时刻径向剖面
    plot_radial_profiles(times, r_cm, T_fvm, C_fvm, output_dir)

    # 4) 获取第二算法结果并画图4
    t_fdm, r_fdm, T_fdm, C_fdm = get_fdm_result(
        fdm_result_path=fdm_result_path,
        fdm_script_path=fdm_script_path,
        attachment1_path=attachment_path,
        dr_cm=args.fdm_dr_cm,
        dt=args.fdm_dt,
    )

    plot_validation_errors(
        times=times,
        r_cm=r_cm,
        T_fvm=T_fvm,
        C_fvm=C_fvm,
        t_fdm=t_fdm,
        r_fdm=r_fdm,
        T_fdm=T_fdm,
        C_fdm=C_fdm,
        output_dir=output_dir,
    )

    print("\n全部完成。共生成 3 张论文图，每张图含 2 个子图。")


if __name__ == "__main__":
    main()
