#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
2026 高教社杯 A 题：药材的烘干问题 —— 问题 4

考虑径向收缩的热—质耦合移动边界模型：
    1. 用 xi = r / R(t) 将移动区域 0 <= r <= R(t) 映射到 0 <= xi <= 1；
    2. 在 xi 上采用圆柱径向有限体积法（FVM）；
    3. 用 scipy.integrate.solve_ivp 的 BDF 方法进行刚性时间积分；
    4. 通过 max(C) - 0.15 = 0 的终止事件确定连续烘干结束时间；
    5. 将固定域结果反插值到题目要求的实际距离，并生成 result4.xlsx。

建议从项目根目录运行：

    python q4/q4_solver.py

默认文件：

    file/附件1.xlsx
    file/附件2.xlsx
    file/result4.xlsx       # 若不存在，则自动新建工作簿

默认输出：

    output/result4.xlsx

也可以显式指定路径：

    python q4_solver.py \
        --attachment1 "附件1.xlsx" \
        --attachment2 "附件2.xlsx" \
        --template "result4.xlsx" \
        --output "output/result4.xlsx"

依赖：

    numpy
    scipy
    openpyxl

重要单位：

    时间统一使用 s；半径和实际距离进入 PDE 前统一使用 m；
    温度状态量使用摄氏度，但 D(C,T) 中必须使用 T + 273.15 K。
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from scipy.integrate import OdeSolution, solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.sparse import csr_matrix, lil_matrix


# ============================================================================
# 1. 参数配置
# ============================================================================


@dataclass(frozen=True)
class SolverConfig:
    """问题 4 的物理参数、数值参数和输出参数。"""

    # 初始状态
    initial_temperature_c: float = 28.0
    initial_moisture: float = 2.55

    # 表面对流参数：沿用题目附录 2 给出的数值
    heat_transfer_coefficient: float = 25.0       # W/(m^2 K)
    mass_transfer_coefficient: float = 8.0e-7     # m/s

    # 4 h 后的稳定烘房工况
    steady_temperature_c: float = 50.0            # degC
    steady_moisture: float = 0.05                 # kg/kg

    # 烘干终止条件
    target_moisture: float = 0.15                  # kg/kg

    # 固定域网格：N 个区间、N+1 个节点，节点包含中心与表面
    n_cells: int = 640

    # 时间积分
    max_hours: float = 100.0
    max_step_s: float = 60.0
    rtol: float = 1.0e-6
    temperature_atol: float = 1.0e-7
    moisture_atol: float = 1.0e-9

    # result4.xlsx 输出间隔
    output_step_s: int = 60
    output_distance_step_cm: float = 0.1

    # 论文表 6 的时间和距离间隔
    paper_time_step_h: float = 6.0
    paper_distance_step_cm: float = 0.5


@dataclass(frozen=True)
class InputData:
    """附件数据及其插值函数。"""

    environment_time_s: np.ndarray
    environment_temperature_c: np.ndarray
    environment_moisture: np.ndarray
    radius_time_s: np.ndarray
    radius_m: np.ndarray
    temperature_interpolator: PchipInterpolator
    moisture_interpolator: PchipInterpolator
    radius_interpolator: PchipInterpolator


@dataclass(frozen=True)
class SpatialGrid:
    """固定材料坐标 xi 上的有限体积网格。"""

    xi: np.ndarray
    delta_xi: float
    internal_face_xi: np.ndarray
    control_volume_weights: np.ndarray


@dataclass
class DryingSolution:
    """连续求解结果及烘干结束状态。"""

    ode_solution: OdeSolution
    dry_time_s: float
    dry_state: np.ndarray
    solver_steps: int
    function_evaluations: int
    message: str


# ============================================================================
# 2. 读取和检查附件
# ============================================================================


def _read_numeric_columns(path: Path, n_columns: int) -> np.ndarray:
    """读取工作簿活动工作表的前 n_columns 列，自动跳过表头和空行。"""

    if not path.exists():
        raise FileNotFoundError(f"找不到输入文件：{path}")

    workbook = load_workbook(path, read_only=True, data_only=True)
    worksheet = workbook.active

    records: list[list[float]] = []
    for row_index, row in enumerate(
        worksheet.iter_rows(min_row=2, max_col=n_columns, values_only=True),
        start=2,
    ):
        values = row[:n_columns]
        if all(value is None for value in values):
            continue
        if any(value is None for value in values):
            raise ValueError(f"{path} 第 {row_index} 行存在缺失数据：{values}")

        try:
            numeric_values = [float(value) for value in values]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{path} 第 {row_index} 行不是纯数值数据：{values}"
            ) from exc

        if not np.all(np.isfinite(numeric_values)):
            raise ValueError(f"{path} 第 {row_index} 行含 NaN 或无穷大：{values}")
        records.append(numeric_values)

    workbook.close()

    if len(records) < 2:
        raise ValueError(f"{path} 中有效数据不足，至少需要 2 行。")

    data = np.asarray(records, dtype=float)
    time_s = data[:, 0]
    if np.any(np.diff(time_s) <= 0.0):
        raise ValueError(f"{path} 的时间列必须严格递增。")
    if time_s[0] < 0.0:
        raise ValueError(f"{path} 的时间不能为负数。")

    return data


def read_input_data(attachment1: Path, attachment2: Path) -> InputData:
    """
    读取附件 1 和附件 2，并建立保形分段三次 Hermite 插值（PCHIP）。

    附件 1：时间/s、烘房温度/degC、烘房水分浓度/(kg/kg)
    附件 2：时间/s、药材半径/cm
    """

    environment = _read_numeric_columns(attachment1, n_columns=3)
    radius_data = _read_numeric_columns(attachment2, n_columns=2)

    env_time = environment[:, 0]
    env_temperature = environment[:, 1]
    env_moisture = environment[:, 2]

    radius_time = radius_data[:, 0]
    radius_m = radius_data[:, 1] * 1.0e-2       # cm -> m

    if np.any(radius_m <= 0.0):
        raise ValueError("附件 2 中的半径必须全部大于 0。")

    radius_increase = np.max(np.diff(radius_m))
    if radius_increase > 1.0e-12:
        warnings.warn(
            "附件 2 中存在半径增大的区间；PCHIP 将忠实保留这些实测变化。",
            RuntimeWarning,
        )

    return InputData(
        environment_time_s=env_time,
        environment_temperature_c=env_temperature,
        environment_moisture=env_moisture,
        radius_time_s=radius_time,
        radius_m=radius_m,
        temperature_interpolator=PchipInterpolator(
            env_time, env_temperature, extrapolate=False
        ),
        moisture_interpolator=PchipInterpolator(
            env_time, env_moisture, extrapolate=False
        ),
        radius_interpolator=PchipInterpolator(
            radius_time, radius_m, extrapolate=False
        ),
    )


def make_environment_function(
    data: InputData,
    config: SolverConfig,
) -> Callable[[float], tuple[float, float]]:
    """
    建立烘房边界函数。

    附件 1 范围内使用 PCHIP；超过附件 1 的最后时刻后，采用恒温干燥
    工况 T_inf=50 degC、C_inf=0.05 kg/kg（可由命令行修改）。
    """

    first_time = float(data.environment_time_s[0])
    last_time = float(data.environment_time_s[-1])

    def environment_at(time_s: float) -> tuple[float, float]:
        if time_s <= first_time:
            return (
                float(data.environment_temperature_c[0]),
                float(data.environment_moisture[0]),
            )
        if time_s <= last_time:
            return (
                float(data.temperature_interpolator(time_s)),
                float(data.moisture_interpolator(time_s)),
            )
        return config.steady_temperature_c, config.steady_moisture

    return environment_at


def make_radius_function(data: InputData) -> Callable[[float], float]:
    """
    建立半径函数 R(t)。

    附件 2 范围内采用 PCHIP；超过最后一个实测时刻后采用末值保持，
    禁止无约束地继续线性外推半径。
    """

    first_time = float(data.radius_time_s[0])
    last_time = float(data.radius_time_s[-1])
    first_radius = float(data.radius_m[0])
    last_radius = float(data.radius_m[-1])

    def radius_at(time_s: float) -> float:
        if time_s <= first_time:
            return first_radius
        if time_s <= last_time:
            return float(data.radius_interpolator(time_s))
        return last_radius

    return radius_at


# ============================================================================
# 3. 物性公式与固定域有限体积离散
# ============================================================================


def material_properties(
    temperature_c: np.ndarray,
    moisture: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """按题目附录 4 计算 rho(C)、cp(C)、k(C) 和 D(C,T)。"""

    # BDF 的牛顿迭代可能短暂试探到极小负值。这里只对物性计算做安全保护，
    # 不改变实际积分状态；正常物理解始终远大于该下限。
    moisture_for_properties = np.maximum(moisture, 1.0e-10)
    temperature_k = np.maximum(temperature_c + 273.15, 1.0)

    density = 760.0 + 90.0 * moisture_for_properties
    heat_capacity = (
        1850.0
        + 2150.0
        * moisture_for_properties
        / (moisture_for_properties + 1.0)
    )
    thermal_conductivity = (
        0.12
        + 0.20
        * moisture_for_properties
        / (moisture_for_properties + 1.0)
    )
    moisture_diffusivity = (
        4.2e-4
        * np.exp(-0.30 / moisture_for_properties)
        * np.exp(-3850.0 / temperature_k)
    )

    return density, heat_capacity, thermal_conductivity, moisture_diffusivity


def harmonic_mean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """正物性在控制体界面上的调和平均。"""

    denominator = np.maximum(left + right, np.finfo(float).tiny)
    return 2.0 * left * right / denominator


def build_spatial_grid(n_cells: int) -> SpatialGrid:
    """建立包含 xi=0 和 xi=1 的节点型圆柱径向有限体积网格。"""

    if n_cells < 10:
        raise ValueError("n_cells 至少应为 10；正式计算建议使用 640。")

    xi = np.linspace(0.0, 1.0, n_cells + 1)
    delta_xi = 1.0 / n_cells

    west_face = np.maximum(0.0, xi - 0.5 * delta_xi)
    east_face = np.minimum(1.0, xi + 0.5 * delta_xi)

    # 乘以 xi 后积分得到的无量纲圆柱控制体积权重：integral xi dxi
    weights = 0.5 * (east_face**2 - west_face**2)
    internal_face_xi = (np.arange(n_cells, dtype=float) + 0.5) * delta_xi

    if np.any(weights <= 0.0):
        raise RuntimeError("有限体积网格生成失败：存在非正控制体积。")

    return SpatialGrid(
        xi=xi,
        delta_xi=delta_xi,
        internal_face_xi=internal_face_xi,
        control_volume_weights=weights,
    )


def build_jacobian_sparsity(n_nodes: int) -> csr_matrix:
    """
    构造 BDF 数值 Jacobian 的块三对角稀疏结构。

    每个节点的温度和水分导数只依赖本节点及相邻节点的 T、C。
    """

    state_size = 2 * n_nodes
    pattern = lil_matrix((state_size, state_size), dtype=np.int8)

    for equation_block in range(2):
        for node in range(n_nodes):
            row = equation_block * n_nodes + node
            left = max(0, node - 1)
            right = min(n_nodes - 1, node + 1)
            for variable_block in range(2):
                col_offset = variable_block * n_nodes
                for neighbor in range(left, right + 1):
                    pattern[row, col_offset + neighbor] = 1

    return pattern.tocsr()


def make_rhs(
    grid: SpatialGrid,
    config: SolverConfig,
    environment_at: Callable[[float], tuple[float, float]],
    radius_at: Callable[[float], float],
) -> Callable[[float, np.ndarray], np.ndarray]:
    """构造固定材料坐标下的热—质耦合 FVM 半离散方程。"""

    n_nodes = grid.xi.size
    n_internal_faces = n_nodes - 1
    delta_xi = grid.delta_xi
    face_xi = grid.internal_face_xi
    weights = grid.control_volume_weights

    h = config.heat_transfer_coefficient
    h_m = config.mass_transfer_coefficient

    def rhs(time_s: float, state: np.ndarray) -> np.ndarray:
        temperature_c = state[:n_nodes]
        moisture = state[n_nodes:]

        density, heat_capacity, conductivity, diffusivity = material_properties(
            temperature_c, moisture
        )
        radius_m = radius_at(time_s)
        environment_temperature_c, environment_moisture = environment_at(time_s)

        if not math.isfinite(radius_m) or radius_m <= 0.0:
            raise FloatingPointError(f"t={time_s} s 时半径无效：{radius_m}")

        conductivity_face = harmonic_mean(
            conductivity[:n_internal_faces], conductivity[1:]
        )
        diffusivity_face = harmonic_mean(
            diffusivity[:n_internal_faces], diffusivity[1:]
        )

        # face_flux[0]：中心边界；face_flux[-1]：药材表面边界。
        # 内部通量的定义为 xi * Gamma * partial(phi)/partial(xi)。
        heat_flux = np.empty(n_nodes + 1, dtype=float)
        moisture_flux = np.empty(n_nodes + 1, dtype=float)

        # xi=0 的中心对称边界
        heat_flux[0] = 0.0
        moisture_flux[0] = 0.0

        heat_flux[1:-1] = (
            face_xi
            * conductivity_face
            * np.diff(temperature_c)
            / delta_xi
        )
        moisture_flux[1:-1] = (
            face_xi
            * diffusivity_face
            * np.diff(moisture)
            / delta_xi
        )

        # xi=1 的 Robin 边界。
        # 由 -k/R * T_xi = h(T_s-T_inf) 得 k*T_xi = R*h(T_inf-T_s)。
        heat_flux[-1] = (
            radius_m * h * (environment_temperature_c - temperature_c[-1])
        )
        moisture_flux[-1] = (
            radius_m * h_m * (environment_moisture - moisture[-1])
        )

        radius_squared = radius_m * radius_m
        temperature_rate = (
            (heat_flux[1:] - heat_flux[:-1])
            / (radius_squared * weights * density * heat_capacity)
        )
        moisture_rate = (
            (moisture_flux[1:] - moisture_flux[:-1])
            / (radius_squared * weights)
        )

        return np.concatenate((temperature_rate, moisture_rate))

    return rhs


# ============================================================================
# 4. BDF 时间积分与烘干终止事件
# ============================================================================


def solve_drying_process(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    grid: SpatialGrid,
    config: SolverConfig,
) -> DryingSolution:
    """求解到全域最大水分浓度第一次降至目标值。"""

    n_nodes = grid.xi.size
    initial_state = np.concatenate(
        (
            np.full(n_nodes, config.initial_temperature_c, dtype=float),
            np.full(n_nodes, config.initial_moisture, dtype=float),
        )
    )

    atol = np.concatenate(
        (
            np.full(n_nodes, config.temperature_atol, dtype=float),
            np.full(n_nodes, config.moisture_atol, dtype=float),
        )
    )
    jacobian_sparsity = build_jacobian_sparsity(n_nodes)

    def dry_event(_time_s: float, state: np.ndarray) -> float:
        moisture = state[n_nodes:]
        return float(np.max(moisture) - config.target_moisture)

    dry_event.terminal = True
    dry_event.direction = -1.0

    result = solve_ivp(
        fun=rhs,
        t_span=(0.0, config.max_hours * 3600.0),
        y0=initial_state,
        method="BDF",
        rtol=config.rtol,
        atol=atol,
        max_step=config.max_step_s,
        jac_sparsity=jacobian_sparsity,
        events=dry_event,
        dense_output=True,
    )

    if not result.success:
        raise RuntimeError(f"BDF 求解失败：{result.message}")
    if result.sol is None:
        raise RuntimeError("BDF 求解未返回稠密输出，无法生成每 60 s 的结果。")
    if len(result.t_events) == 0 or result.t_events[0].size == 0:
        final_moisture = result.y[n_nodes:, -1]
        raise RuntimeError(
            "在最大计算时长内没有达到烘干条件。"
            f" t={config.max_hours:.3f} h 时 max(C)="
            f"{np.max(final_moisture):.6f} kg/kg；"
            "请增大 --max-hours 或检查参数。"
        )

    dry_time_s = float(result.t_events[0][0])
    dry_state = np.asarray(result.y_events[0][0], dtype=float)

    return DryingSolution(
        ode_solution=result.sol,
        dry_time_s=dry_time_s,
        dry_state=dry_state,
        solver_steps=result.t.size,
        function_evaluations=result.nfev,
        message=result.message,
    )


def continue_to_reporting_time(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    grid: SpatialGrid,
    config: SolverConfig,
    dry_time_s: float,
    dry_state: np.ndarray,
    reporting_end_s: float,
) -> OdeSolution | None:
    """从连续终止时刻继续积分到后一个 60 s 报告网格点。"""

    if reporting_end_s <= dry_time_s + 1.0e-9:
        return None

    n_nodes = grid.xi.size
    atol = np.concatenate(
        (
            np.full(n_nodes, config.temperature_atol, dtype=float),
            np.full(n_nodes, config.moisture_atol, dtype=float),
        )
    )

    continuation = solve_ivp(
        fun=rhs,
        t_span=(dry_time_s, reporting_end_s),
        y0=dry_state,
        method="BDF",
        rtol=config.rtol,
        atol=atol,
        max_step=min(config.max_step_s, reporting_end_s - dry_time_s),
        jac_sparsity=build_jacobian_sparsity(n_nodes),
        dense_output=True,
    )

    if not continuation.success or continuation.sol is None:
        raise RuntimeError(f"终止时刻后的短积分失败：{continuation.message}")

    return continuation.sol


# ============================================================================
# 5. 输出采样、实际坐标恢复与 Excel 写入
# ============================================================================


def reporting_end_time(dry_time_s: float, output_step_s: int) -> int:
    """返回第一个不小于连续结束时间的输出网格时刻。"""

    if output_step_s <= 0:
        raise ValueError("output_step_s 必须为正整数。")
    quotient = dry_time_s / output_step_s
    return int(math.ceil(quotient - 1.0e-12) * output_step_s)


def sample_moisture_history(
    solution: DryingSolution,
    continuation: OdeSolution | None,
    grid: SpatialGrid,
    config: SolverConfig,
    reporting_end_s: int,
) -> tuple[np.ndarray, np.ndarray]:
    """在 60,120,...,reporting_end_s 时刻采样节点水分场。"""

    times_s = np.arange(
        config.output_step_s,
        reporting_end_s + config.output_step_s,
        config.output_step_s,
        dtype=float,
    )
    n_nodes = grid.xi.size
    moisture_history = np.empty((times_s.size, n_nodes), dtype=float)

    before_or_at_event = times_s <= solution.dry_time_s + 1.0e-9
    if np.any(before_or_at_event):
        states = solution.ode_solution(times_s[before_or_at_event])
        moisture_history[before_or_at_event, :] = states[n_nodes:, :].T

    after_event = ~before_or_at_event
    if np.any(after_event):
        if continuation is None:
            raise RuntimeError("缺少终止事件之后的短积分结果。")
        states = continuation(times_s[after_event])
        moisture_history[after_event, :] = states[n_nodes:, :].T

    if not np.all(np.isfinite(moisture_history)):
        raise FloatingPointError("输出水分场中出现 NaN 或无穷大。")

    return times_s, moisture_history


def build_fixed_distance_columns(
    initial_radius_m: float,
    distance_step_cm: float,
) -> np.ndarray:
    """
    建立 0,0.1,...,1.9 cm 等固定实际距离列。

    初始表面使用单独的“药材表面”列，因此固定距离列严格小于初始半径。
    """

    if distance_step_cm <= 0.0:
        raise ValueError("实际距离输出间隔必须大于 0。")

    initial_radius_cm = 100.0 * initial_radius_m
    n_distances = int(
        math.floor((initial_radius_cm - 1.0e-10) / distance_step_cm)
    ) + 1
    distances = np.arange(n_distances, dtype=float) * distance_step_cm
    return np.round(distances, 10)


def interpolate_at_actual_distances(
    nodal_moisture: np.ndarray,
    grid: SpatialGrid,
    radius_m: float,
    distances_cm: np.ndarray,
) -> list[float | None]:
    """将 C(xi,t) 反插值到固定实际距离；药材外部位置返回 None。"""

    values: list[float | None] = []
    for distance_cm in distances_cm:
        distance_m = distance_cm * 1.0e-2

        # 等于表面的固定列也留空，表面值统一写到最后一列，避免重复。
        if distance_m >= radius_m - 1.0e-12:
            values.append(None)
            continue

        xi_query = distance_m / radius_m
        value = float(np.interp(xi_query, grid.xi, nodal_moisture))
        values.append(value)

    return values


def create_result_workbook(
    template_path: Path | None,
    output_path: Path,
    times_s: np.ndarray,
    moisture_history: np.ndarray,
    distances_cm: np.ndarray,
    grid: SpatialGrid,
    radius_at: Callable[[float], float],
) -> None:
    """生成符合题目 result4.xlsx 模板含义的完整结果表。"""

    if template_path is not None and template_path.exists():
        workbook = load_workbook(template_path)
        worksheet = workbook.active

        # 模板只给出了示意行列。保留工作表名称，但清除示意内容和旧表对象。
        for table_name in list(worksheet.tables.keys()):
            del worksheet.tables[table_name]
        for merged_range in list(worksheet.merged_cells.ranges):
            worksheet.unmerge_cells(str(merged_range))
        if worksheet.max_row > 0:
            worksheet.delete_rows(1, worksheet.max_row)
    else:
        workbook = Workbook()
        worksheet = workbook.active
        worksheet.title = "Sheet1"

    header: list[object] = ["时间\\到药材中心的距离"]
    header.extend(float(value) for value in distances_cm)
    header.append("药材表面")
    worksheet.append(header)

    for row_index, (time_s, nodal_moisture) in enumerate(
        zip(times_s, moisture_history, strict=True),
        start=2,
    ):
        radius_m = radius_at(float(time_s))
        actual_distance_values = interpolate_at_actual_distances(
            nodal_moisture=nodal_moisture,
            grid=grid,
            radius_m=radius_m,
            distances_cm=distances_cm,
        )

        row: list[object] = [int(round(float(time_s)))]
        row.extend(
            None if value is None else round(float(value), 4)
            for value in actual_distance_values
        )
        row.append(round(float(nodal_moisture[-1]), 4))
        worksheet.append(row)

    # 简洁、数据型格式：只用灰色表头、清晰数字格式和冻结窗格。
    header_fill = PatternFill(fill_type="solid", fgColor="D9E2F3")
    thin_gray = Side(style="thin", color="D9D9D9")
    header_border = Border(
        left=thin_gray,
        right=thin_gray,
        top=thin_gray,
        bottom=thin_gray,
    )

    header_font = Font(name="Arial", size=10, bold=True, color="000000")
    header_alignment = Alignment(horizontal="center", vertical="center")
    body_font = Font(name="Arial", size=10)
    time_alignment = Alignment(horizontal="center", vertical="center")
    number_alignment = Alignment(horizontal="right", vertical="center")

    for cell in worksheet[1]:
        cell.font = header_font
        cell.fill = header_fill
        cell.border = header_border
        cell.alignment = header_alignment

    worksheet.row_dimensions[1].height = 25
    worksheet.column_dimensions["A"].width = 24

    last_column = 2 + len(distances_cm)
    for column_index in range(2, last_column + 1):
        worksheet.column_dimensions[get_column_letter(column_index)].width = 10
    worksheet.column_dimensions[get_column_letter(last_column)].width = 13

    for row in worksheet.iter_rows(
        min_row=2,
        max_row=worksheet.max_row,
        min_col=1,
        max_col=worksheet.max_column,
    ):
        row[0].number_format = "0"
        row[0].alignment = time_alignment
        row[0].font = body_font
        for cell in row[1:]:
            cell.number_format = "0.0000"
            cell.alignment = number_alignment
            cell.font = body_font

    worksheet.freeze_panes = "B2"
    worksheet.auto_filter.ref = (
        f"A1:{get_column_letter(worksheet.max_column)}{worksheet.max_row}"
    )
    worksheet.sheet_view.showGridLines = True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output_path)
    workbook.close()


def validate_result_workbook(
    output_path: Path,
    expected_times_s: np.ndarray,
    expected_distances_cm: np.ndarray,
    target_moisture: float,
) -> None:
    """重新打开结果文件，检查表头、时间步长、维度和最后时刻判据。"""

    workbook = load_workbook(output_path, read_only=True, data_only=True)
    worksheet = workbook.active

    expected_rows = 1 + expected_times_s.size
    expected_columns = 2 + expected_distances_cm.size

    # ReadOnlyWorksheet 的 worksheet.cell(row=...) 属于随机访问，会反复扫描 XML。
    # 这里一次性顺序读取，保证校验复杂度为 O(行数 × 列数)。
    all_rows = list(worksheet.iter_rows(values_only=True))
    if len(all_rows) != expected_rows:
        raise RuntimeError(
            f"结果表行数错误：实际 {len(all_rows)}，预期 {expected_rows}。"
        )
    if not all_rows or len(all_rows[0]) != expected_columns:
        raise RuntimeError(
            "结果表列数错误："
            f"实际 {0 if not all_rows else len(all_rows[0])}，"
            f"预期 {expected_columns}。"
        )

    actual_times = np.asarray(
        [row[0] for row in all_rows[1:]],
        dtype=float,
    )
    if not np.array_equal(actual_times, expected_times_s):
        raise RuntimeError("结果表 A 列时间与规定的 60 s 网格不一致。")

    header_distances = np.asarray(
        all_rows[0][1 : 1 + expected_distances_cm.size],
        dtype=float,
    )
    if not np.allclose(header_distances, expected_distances_cm, atol=1.0e-12):
        raise RuntimeError("结果表固定实际距离表头不正确。")

    last_row_values = all_rows[-1][1:expected_columns]
    last_numeric_values = [
        float(value) for value in last_row_values if isinstance(value, (int, float))
    ]
    if not last_numeric_values:
        raise RuntimeError("结果表最后一行没有水分浓度数值。")

    # Excel 只保留 4 位小数，因此允许半个最末位的舍入误差。
    if max(last_numeric_values) > target_moisture + 5.0e-5:
        raise RuntimeError(
            "结果表最后一个 60 s 时刻仍未满足全域含水率要求："
            f"max(C)={max(last_numeric_values):.6f}。"
        )

    workbook.close()


# ============================================================================
# 6. 论文表 6 数据和终端摘要
# ============================================================================


def _paper_times(dry_time_s: float, step_hours: float) -> np.ndarray:
    """生成 6 h、12 h、... 和精确烘干结束时间。"""

    step_s = step_hours * 3600.0
    regular = np.arange(step_s, dry_time_s - 1.0e-9, step_s, dtype=float)
    return np.concatenate((regular, np.asarray([dry_time_s], dtype=float)))


def print_paper_table(
    solution: DryingSolution,
    grid: SpatialGrid,
    config: SolverConfig,
    radius_at: Callable[[float], float],
) -> None:
    """在终端打印论文表 6 可直接采用的数据。"""

    times_s = _paper_times(solution.dry_time_s, config.paper_time_step_h)
    n_nodes = grid.xi.size

    radii_m = np.asarray([radius_at(float(time_s)) for time_s in times_s])
    minimum_radius_cm = 100.0 * float(np.min(radii_m))
    n_fixed = int(
        math.floor(
            (minimum_radius_cm - 1.0e-10) / config.paper_distance_step_cm
        )
    ) + 1
    distances_cm = (
        np.arange(n_fixed, dtype=float) * config.paper_distance_step_cm
    )

    labels = [f"r={distance:g} cm" for distance in distances_cm]
    labels.append("药材表面")

    print("\n论文表 6 建议数据（kg/kg）")
    print("时间/h".ljust(14) + "".join(label.rjust(14) for label in labels))

    for time_s, radius_m in zip(times_s, radii_m, strict=True):
        if abs(time_s - solution.dry_time_s) <= 1.0e-6:
            state = solution.dry_state
            time_label = f"{time_s / 3600.0:.4f}*"
        else:
            state = solution.ode_solution(float(time_s))
            time_label = f"{time_s / 3600.0:.0f}"

        moisture = np.asarray(state[n_nodes:], dtype=float)
        fixed_values = interpolate_at_actual_distances(
            nodal_moisture=moisture,
            grid=grid,
            radius_m=float(radius_m),
            distances_cm=distances_cm,
        )
        numeric_values = [
            float("nan") if value is None else float(value)
            for value in fixed_values
        ]
        numeric_values.append(float(moisture[-1]))

        print(
            time_label.ljust(14)
            + "".join(
                ("" if not math.isfinite(value) else f"{value:.4f}").rjust(14)
                for value in numeric_values
            )
        )

    print("* 表示连续事件定位得到的烘干结束时间。")


def print_summary(
    solution: DryingSolution,
    reporting_end_s: int,
    grid: SpatialGrid,
    config: SolverConfig,
    radius_at: Callable[[float], float],
    output_path: Path,
) -> None:
    """打印主要求解信息和结束状态。"""

    n_nodes = grid.xi.size
    dry_moisture = solution.dry_state[n_nodes:]
    max_index = int(np.argmax(dry_moisture))
    max_xi = float(grid.xi[max_index])
    radius_cm = 100.0 * radius_at(solution.dry_time_s)

    print("\n" + "=" * 72)
    print("问题 4 求解完成")
    print("=" * 72)
    print(f"网格区间数 N          : {config.n_cells}")
    print(f"BDF 接受的时间节点数   : {solution.solver_steps}")
    print(f"右端函数调用次数        : {solution.function_evaluations}")
    print(f"连续烘干结束时间        : {solution.dry_time_s:.4f} s")
    print(f"连续烘干时长            : {solution.dry_time_s / 3600.0:.6f} h")
    print(f"首个 60 s 报告结束时刻 : {reporting_end_s:d} s")
    print(f"对应报告时长            : {reporting_end_s / 3600.0:.6f} h")
    print(f"结束时药材半径          : {radius_cm:.6f} cm")
    print(f"结束时 max(C)           : {np.max(dry_moisture):.8f} kg/kg")
    print(f"max(C) 所在 xi          : {max_xi:.6f}")
    print(f"结果文件                : {output_path.resolve()}")


# ============================================================================
# 7. 命令行入口
# ============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="问题 4：考虑尺寸收缩的热—质耦合 FVM+BDF 求解器。"
    )

    parser.add_argument(
        "--attachment1",
        type=Path,
        default=Path("file/附件1.xlsx"),
        help="附件 1 路径，默认 file/附件1.xlsx。",
    )
    parser.add_argument(
        "--attachment2",
        type=Path,
        default=Path("file/附件2.xlsx"),
        help="附件 2 路径，默认 file/附件2.xlsx。",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("file/附件3/result4.xlsx"),
        help="result4.xlsx 模板路径；若不存在则自动新建。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/q4/result4.xlsx"),
        help="结果文件路径，默认 output/result4.xlsx。",
    )

    parser.add_argument("--n-cells", type=int, default=640)
    parser.add_argument("--max-hours", type=float, default=100.0)
    parser.add_argument("--max-step-s", type=float, default=60.0)
    parser.add_argument("--output-step-s", type=int, default=60)
    parser.add_argument("--rtol", type=float, default=1.0e-6)

    parser.add_argument("--h", type=float, default=25.0)
    parser.add_argument("--hm", type=float, default=8.0e-7)
    parser.add_argument("--target-moisture", type=float, default=0.15)
    parser.add_argument("--steady-temperature-c", type=float, default=50.0)
    parser.add_argument("--steady-moisture", type=float, default=0.05)

    return parser


def validate_config(config: SolverConfig) -> None:
    """检查命令行生成的参数是否合法。"""

    if config.max_hours <= 0.0:
        raise ValueError("max_hours 必须大于 0。")
    if config.max_step_s <= 0.0:
        raise ValueError("max_step_s 必须大于 0。")
    if config.output_step_s <= 0:
        raise ValueError("output_step_s 必须大于 0。")
    if config.rtol <= 0.0:
        raise ValueError("rtol 必须大于 0。")
    if config.heat_transfer_coefficient <= 0.0:
        raise ValueError("h 必须大于 0。")
    if config.mass_transfer_coefficient <= 0.0:
        raise ValueError("h_m 必须大于 0。")
    if config.target_moisture <= config.steady_moisture:
        # target 必须高于外界值，才能以扩散方式有限时间接近并穿越目标。
        raise ValueError("target_moisture 必须大于 steady_moisture。")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config = SolverConfig(
        heat_transfer_coefficient=args.h,
        mass_transfer_coefficient=args.hm,
        steady_temperature_c=args.steady_temperature_c,
        steady_moisture=args.steady_moisture,
        target_moisture=args.target_moisture,
        n_cells=args.n_cells,
        max_hours=args.max_hours,
        max_step_s=args.max_step_s,
        output_step_s=args.output_step_s,
        rtol=args.rtol,
    )
    validate_config(config)

    data = read_input_data(args.attachment1, args.attachment2)
    environment_at = make_environment_function(data, config)
    radius_at = make_radius_function(data)
    grid = build_spatial_grid(config.n_cells)
    rhs = make_rhs(grid, config, environment_at, radius_at)

    print("正在求解问题 4，请稍候……")
    solution = solve_drying_process(rhs, grid, config)

    report_end_s = reporting_end_time(
        solution.dry_time_s, config.output_step_s
    )
    continuation = continue_to_reporting_time(
        rhs=rhs,
        grid=grid,
        config=config,
        dry_time_s=solution.dry_time_s,
        dry_state=solution.dry_state,
        reporting_end_s=float(report_end_s),
    )
    times_s, moisture_history = sample_moisture_history(
        solution=solution,
        continuation=continuation,
        grid=grid,
        config=config,
        reporting_end_s=report_end_s,
    )

    distances_cm = build_fixed_distance_columns(
        initial_radius_m=radius_at(0.0),
        distance_step_cm=config.output_distance_step_cm,
    )
    create_result_workbook(
        template_path=args.template,
        output_path=args.output,
        times_s=times_s,
        moisture_history=moisture_history,
        distances_cm=distances_cm,
        grid=grid,
        radius_at=radius_at,
    )
    validate_result_workbook(
        output_path=args.output,
        expected_times_s=times_s,
        expected_distances_cm=distances_cm,
        target_moisture=config.target_moisture,
    )

    print_summary(
        solution=solution,
        reporting_end_s=report_end_s,
        grid=grid,
        config=config,
        radius_at=radius_at,
        output_path=args.output,
    )
    print_paper_table(solution, grid, config, radius_at)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError, FloatingPointError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1) from exc
