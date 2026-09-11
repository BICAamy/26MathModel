#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
2026 高教社杯 A 题：药材的烘干问题——问题 3

算法：
    表面加密的一维圆柱径向有限体积法（FVM）
    + BDF 隐式时间积分
    + 全域最大含水率事件检测

默认从项目根目录运行：
    python q3/q3_solver.py

默认输入：
    file/附件1.xlsx
    file/result3.xlsx（若不存在则自动新建）

默认输出：
    output/result3.xlsx

依赖：
    numpy
    scipy
    openpyxl

推荐的最终提交命令：
    python q3/q3_solver.py --grid-intervals 1280

若要严格把 4 h 后的烘房条件固定为 50 °C 和 0.05 kg/kg：
    python q3/q3_solver.py \
        --plateau-temperature 50 \
        --plateau-moisture 0.05 \
        --grid-intervals 1280
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix, lil_matrix


# ============================================================
# 1. 数据结构
# ============================================================


@dataclass(frozen=True)
class SolverConfig:
    """问题 3 的物理参数和数值参数。"""

    radius_m: float = 0.02
    initial_temperature_c: float = 28.0
    initial_moisture: float = 2.55
    critical_moisture: float = 0.15

    # 附录 3 没有重新给出表面对流系数，因此沿用附录 2。
    heat_transfer_coefficient: float = 25.0
    mass_transfer_coefficient: float = 8.0e-7

    grid_intervals: int = 640
    stretch_power: float = 2.0
    max_hours: float = 96.0
    max_step_s: float = 60.0
    rtol: float = 2.0e-7
    atol_temperature: float = 1.0e-6
    atol_moisture: float = 1.0e-9


@dataclass(frozen=True)
class AmbientBoundary:
    """附件 1 给出的烘房时变边界。"""

    time_s: np.ndarray
    temperature_c: np.ndarray
    moisture: np.ndarray
    plateau_temperature_c: float
    plateau_moisture: float

    def value(self, t_s: float) -> tuple[float, float]:
        """
        0～附件末时刻：分段线性插值；
        超出附件时域：保持恒温恒湿平台值，禁止线性外推。
        """
        if t_s <= self.time_s[-1]:
            t_air = float(np.interp(t_s, self.time_s, self.temperature_c))
            c_air = float(np.interp(t_s, self.time_s, self.moisture))
            return t_air, c_air
        return self.plateau_temperature_c, self.plateau_moisture


@dataclass(frozen=True)
class RadialMesh:
    """包含圆心和表面节点的非均匀有限体积网格。"""

    r_m: np.ndarray
    face_r_m: np.ndarray
    node_distance_m: np.ndarray
    volume_per_length_m2: np.ndarray
    face_area_per_length_m: np.ndarray
    outer_area_per_length_m: float

    @property
    def n_nodes(self) -> int:
        return self.r_m.size


@dataclass
class DryingSolution:
    """一次网格计算的结果。"""

    model: "DryingModel"
    ode_solution: object
    drying_time_s: float
    event_state: np.ndarray
    wall_time_s: float


# ============================================================
# 2. 读取附件 1 并构造外部边界
# ============================================================


def read_attachment1(
    path: Path,
    plateau_window_s: float = 3600.0,
    plateau_temperature_c: float | None = None,
    plateau_moisture: float | None = None,
) -> AmbientBoundary:
    """读取附件 1 的前三列：时间/s、温度/°C、水分浓度/(kg/kg)。"""
    if not path.exists():
        raise FileNotFoundError(
            f"未找到附件 1：{path}\n"
            "请用 --attachment1 指定实际路径。"
        )

    workbook = load_workbook(path, data_only=True, read_only=True)
    worksheet = workbook.active

    rows: list[tuple[float, float, float]] = []
    for row_number, values in enumerate(
        worksheet.iter_rows(min_row=2, max_col=3, values_only=True), start=2
    ):
        if all(value is None for value in values):
            continue
        if any(value is None for value in values):
            raise ValueError(f"附件 1 第 {row_number} 行存在空值：{values}")
        try:
            rows.append(tuple(float(value) for value in values))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"附件 1 第 {row_number} 行不是有效数值：{values}"
            ) from exc

    workbook.close()

    if len(rows) < 2:
        raise ValueError("附件 1 至少需要两个有效数据点。")

    data = np.asarray(rows, dtype=float)
    order = np.argsort(data[:, 0])
    data = data[order]
    time_s = data[:, 0]
    temperature_c = data[:, 1]
    moisture = data[:, 2]

    if not np.all(np.diff(time_s) > 0.0):
        raise ValueError("附件 1 的时间必须严格递增，不能存在重复时间。")
    if time_s[0] > 0.0:
        raise ValueError("附件 1 缺少 t=0 附近的初始边界数据。")
    if np.any(temperature_c <= -273.15):
        raise ValueError("附件 1 中存在低于绝对零度的温度。")
    if np.any(moisture < 0.0):
        raise ValueError("附件 1 中存在负水分浓度。")

    tail_start = max(float(time_s[0]), float(time_s[-1] - plateau_window_s))
    tail_mask = time_s >= tail_start
    if not np.any(tail_mask):
        tail_mask[-1] = True

    if plateau_temperature_c is None:
        plateau_temperature_c = float(np.mean(temperature_c[tail_mask]))
    if plateau_moisture is None:
        plateau_moisture = float(np.mean(moisture[tail_mask]))

    return AmbientBoundary(
        time_s=time_s,
        temperature_c=temperature_c,
        moisture=moisture,
        plateau_temperature_c=float(plateau_temperature_c),
        plateau_moisture=float(plateau_moisture),
    )


# ============================================================
# 3. 非均匀径向有限体积网格
# ============================================================


def build_radial_mesh(
    radius_m: float,
    grid_intervals: int,
    stretch_power: float,
) -> RadialMesh:
    """
    构造表面加密网格：

        r_i = R * [1 - (1 - i/N)^p]

    p>1 时网格在药材表面 r=R 附近更密。
    """
    if radius_m <= 0.0:
        raise ValueError("药材半径必须为正数。")
    if grid_intervals < 20:
        raise ValueError("grid_intervals 至少应为 20。")
    if stretch_power < 1.0:
        raise ValueError("stretch_power 应不小于 1；推荐使用 2。")

    xi = np.linspace(0.0, 1.0, grid_intervals + 1)
    r_m = radius_m * (1.0 - (1.0 - xi) ** stretch_power)
    node_distance_m = np.diff(r_m)

    if np.any(node_distance_m <= 0.0):
        raise ValueError("径向网格节点没有严格递增。")

    # 相邻节点的中点作为有限体积界面。
    face_r_m = 0.5 * (r_m[:-1] + r_m[1:])
    west_boundary_m = np.concatenate(([0.0], face_r_m))
    east_boundary_m = np.concatenate((face_r_m, [radius_m]))

    # 取单位圆柱长度，长度会在通量和体积中相消。
    volume_per_length_m2 = np.pi * (
        east_boundary_m**2 - west_boundary_m**2
    )
    face_area_per_length_m = 2.0 * np.pi * face_r_m
    outer_area_per_length_m = 2.0 * np.pi * radius_m

    return RadialMesh(
        r_m=r_m,
        face_r_m=face_r_m,
        node_distance_m=node_distance_m,
        volume_per_length_m2=volume_per_length_m2,
        face_area_per_length_m=face_area_per_length_m,
        outer_area_per_length_m=outer_area_per_length_m,
    )


def harmonic_mean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """扩散型方程界面系数的调和平均。"""
    denominator = left + right
    return np.divide(
        2.0 * left * right,
        denominator,
        out=np.zeros_like(denominator),
        where=np.abs(denominator) > 1.0e-300,
    )


# ============================================================
# 4. 变物性热—质耦合模型
# ============================================================


class DryingModel:
    """问题 3 的半离散常微分方程组。"""

    def __init__(self, config: SolverConfig, boundary: AmbientBoundary):
        self.config = config
        self.boundary = boundary
        self.mesh = build_radial_mesh(
            radius_m=config.radius_m,
            grid_intervals=config.grid_intervals,
            stretch_power=config.stretch_power,
        )
        self.jacobian_sparsity = self._build_jacobian_sparsity()

    @property
    def n_nodes(self) -> int:
        return self.mesh.n_nodes

    @property
    def state_size(self) -> int:
        return 2 * self.n_nodes

    def initial_state(self) -> np.ndarray:
        temperature = np.full(
            self.n_nodes, self.config.initial_temperature_c, dtype=float
        )
        moisture = np.full(
            self.n_nodes, self.config.initial_moisture, dtype=float
        )
        return np.concatenate((temperature, moisture))

    @staticmethod
    def material_properties(
        temperature_c: np.ndarray,
        moisture: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """按附录 3 计算 rho、cp、k、D。"""
        # 只在经验公式中使用保护值，避免牛顿迭代的中间预测值除以零。
        # 正常求解在 C=0.15 时已经终止，不会真正接近该保护下限。
        safe_moisture = np.maximum(moisture, 1.0e-10)
        temperature_k = temperature_c + 273.15
        if np.any(temperature_k <= 0.0):
            raise FloatingPointError("计算中出现非物理的绝对温度。")

        density = 650.0 + 128.0 * safe_moisture
        heat_capacity = 1450.0 + 2736.0 * (
            safe_moisture / (safe_moisture + 1.0)
        )
        thermal_conductivity = 0.21 + 0.38 * (
            safe_moisture / (safe_moisture + 1.0)
        )
        moisture_diffusivity = (
            2.4e-3
            * np.exp(-0.45 / safe_moisture)
            * np.exp(-3850.0 / temperature_k)
        )
        return density, heat_capacity, thermal_conductivity, moisture_diffusivity

    def rhs(self, t_s: float, state: np.ndarray) -> np.ndarray:
        """有限体积半离散后的 d(state)/dt。"""
        n = self.n_nodes
        temperature_c = state[:n]
        moisture = state[n:]

        density, heat_capacity, conductivity, diffusivity = (
            self.material_properties(temperature_c, moisture)
        )

        k_face = harmonic_mean(conductivity[:-1], conductivity[1:])
        d_face = harmonic_mean(diffusivity[:-1], diffusivity[1:])

        temperature_gradient = (
            temperature_c[1:] - temperature_c[:-1]
        ) / self.mesh.node_distance_m
        moisture_gradient = (
            moisture[1:] - moisture[:-1]
        ) / self.mesh.node_distance_m

        heat_face_flux = (
            self.mesh.face_area_per_length_m * k_face * temperature_gradient
        )
        moisture_face_flux = (
            self.mesh.face_area_per_length_m * d_face * moisture_gradient
        )

        t_air_c, c_air = self.boundary.value(t_s)
        d_temperature_dt = np.empty(n, dtype=float)
        d_moisture_dt = np.empty(n, dtype=float)

        # 圆心控制体：r=0 处界面面积为零，自动满足对称零通量条件。
        d_temperature_dt[0] = heat_face_flux[0] / (
            density[0]
            * heat_capacity[0]
            * self.mesh.volume_per_length_m2[0]
        )
        d_moisture_dt[0] = (
            moisture_face_flux[0] / self.mesh.volume_per_length_m2[0]
        )

        # 内部控制体：东侧通量减西侧通量。
        d_temperature_dt[1:-1] = (
            heat_face_flux[1:] - heat_face_flux[:-1]
        ) / (
            density[1:-1]
            * heat_capacity[1:-1]
            * self.mesh.volume_per_length_m2[1:-1]
        )
        d_moisture_dt[1:-1] = (
            moisture_face_flux[1:] - moisture_face_flux[:-1]
        ) / self.mesh.volume_per_length_m2[1:-1]

        # 药材表面的 Robin 对流边界。
        surface_heat_flux = (
            self.mesh.outer_area_per_length_m
            * self.config.heat_transfer_coefficient
            * (t_air_c - temperature_c[-1])
        )
        surface_moisture_flux = (
            self.mesh.outer_area_per_length_m
            * self.config.mass_transfer_coefficient
            * (c_air - moisture[-1])
        )

        d_temperature_dt[-1] = (
            surface_heat_flux - heat_face_flux[-1]
        ) / (
            density[-1]
            * heat_capacity[-1]
            * self.mesh.volume_per_length_m2[-1]
        )
        d_moisture_dt[-1] = (
            surface_moisture_flux - moisture_face_flux[-1]
        ) / self.mesh.volume_per_length_m2[-1]

        derivative = np.concatenate((d_temperature_dt, d_moisture_dt))
        if not np.all(np.isfinite(derivative)):
            raise FloatingPointError("常微分方程右端出现 NaN 或无穷值。")
        return derivative

    def _build_jacobian_sparsity(self) -> csr_matrix:
        """构造块三对角 Jacobian 稀疏结构，加速 BDF。"""
        n = self.n_nodes
        sparsity = lil_matrix((2 * n, 2 * n), dtype=np.int8)

        # 每个温度/水分方程只依赖相邻节点的温度和水分。
        for equation_block in (0, 1):
            for i in range(n):
                for j in range(max(0, i - 1), min(n, i + 2)):
                    sparsity[equation_block * n + i, j] = 1
                    sparsity[equation_block * n + i, n + j] = 1
        return sparsity.tocsr()


# ============================================================
# 5. BDF 积分与烘干结束事件
# ============================================================


def solve_until_dry(model: DryingModel) -> DryingSolution:
    """积分到全域最大含水率第一次达到临界值。"""
    config = model.config
    n = model.n_nodes

    def dry_event(_t_s: float, state: np.ndarray) -> float:
        return float(np.max(state[n:]) - config.critical_moisture)

    dry_event.direction = -1
    dry_event.terminal = True

    absolute_tolerance = np.concatenate(
        (
            np.full(n, config.atol_temperature),
            np.full(n, config.atol_moisture),
        )
    )

    start = time.perf_counter()
    solution = solve_ivp(
        fun=model.rhs,
        t_span=(0.0, config.max_hours * 3600.0),
        y0=model.initial_state(),
        method="BDF",
        rtol=config.rtol,
        atol=absolute_tolerance,
        max_step=config.max_step_s,
        events=dry_event,
        dense_output=True,
        jac_sparsity=model.jacobian_sparsity,
    )
    wall_time = time.perf_counter() - start

    if not solution.success:
        raise RuntimeError(f"BDF 求解失败：{solution.message}")
    if not solution.t_events or solution.t_events[0].size == 0:
        final_max = float(np.max(solution.y[n:, -1]))
        raise RuntimeError(
            f"计算到 {config.max_hours:.2f} h 仍未达到烘干条件；"
            f"此时最大含水率为 {final_max:.8f}。"
            "请增大 --max-hours 后重试。"
        )

    drying_time_s = float(solution.t_events[0][0])
    event_state = np.asarray(solution.y_events[0][0], dtype=float)
    return DryingSolution(
        model=model,
        ode_solution=solution,
        drying_time_s=drying_time_s,
        event_state=event_state,
        wall_time_s=wall_time,
    )


def evaluate_solution(
    result: DryingSolution,
    requested_times_s: np.ndarray,
) -> np.ndarray:
    """
    在指定时刻计算完整状态。

    主积分在精确事件时刻终止。若 result3 的最后一分钟稍晚于事件时刻，
    则从事件状态继续积分不足 60 s，避免对稠密解进行区间外外推。
    """
    times = np.asarray(requested_times_s, dtype=float)
    if times.ndim != 1 or times.size == 0:
        raise ValueError("requested_times_s 必须是一维非空数组。")
    if np.any(np.diff(times) < 0.0):
        raise ValueError("requested_times_s 必须按升序排列。")
    if times[0] < 0.0:
        raise ValueError("输出时间不能为负。")

    model = result.model
    config = model.config
    output = np.empty((model.state_size, times.size), dtype=float)

    before_or_at_event = times <= result.drying_time_s + 1.0e-10
    if np.any(before_or_at_event):
        output[:, before_or_at_event] = result.ode_solution.sol(
            times[before_or_at_event]
        )

    after_event = ~before_or_at_event
    if np.any(after_event):
        after_times = times[after_event]
        n = model.n_nodes
        absolute_tolerance = np.concatenate(
            (
                np.full(n, config.atol_temperature),
                np.full(n, config.atol_moisture),
            )
        )
        continuation = solve_ivp(
            fun=model.rhs,
            t_span=(result.drying_time_s, float(after_times[-1])),
            y0=result.event_state,
            method="BDF",
            t_eval=after_times,
            rtol=config.rtol,
            atol=absolute_tolerance,
            max_step=config.max_step_s,
            jac_sparsity=model.jacobian_sparsity,
        )
        if not continuation.success:
            raise RuntimeError(f"结束时刻后的短时积分失败：{continuation.message}")
        output[:, after_event] = continuation.y

    return output


def interpolate_moisture_profiles(
    result: DryingSolution,
    times_s: np.ndarray,
    output_radius_m: np.ndarray,
) -> np.ndarray:
    """把细计算网格上的含水率插值到指定输出半径。"""
    states = evaluate_solution(result, times_s)
    n = result.model.n_nodes
    moisture_on_mesh = states[n:, :].T
    mesh_radius = result.model.mesh.r_m

    if output_radius_m[0] < 0.0 or output_radius_m[-1] > mesh_radius[-1]:
        raise ValueError("输出半径超出了药材内部区域。")

    profiles = np.vstack(
        [
            np.interp(output_radius_m, mesh_radius, moisture_row)
            for moisture_row in moisture_on_mesh
        ]
    )
    if not np.all(np.isfinite(profiles)):
        raise FloatingPointError("插值后的含水率存在 NaN 或无穷值。")
    return profiles


# ============================================================
# 6. 写入 result3.xlsx
# ============================================================


def clear_worksheet_values(worksheet) -> None:
    """清除模板中的占位值，保留工作表本身。"""
    for merged_range in list(worksheet.merged_cells.ranges):
        worksheet.unmerge_cells(str(merged_range))
    for row in worksheet.iter_rows():
        for cell in row:
            cell.value = None


def write_result3(
    template_path: Path,
    output_path: Path,
    times_s: np.ndarray,
    radius_cm: np.ndarray,
    moisture: np.ndarray,
) -> None:
    """按照附件 3 的结构写出问题 3 完整结果。"""
    if moisture.shape != (times_s.size, radius_cm.size):
        raise ValueError("水分浓度矩阵尺寸与时间/空间坐标不一致。")

    if template_path.exists():
        workbook = load_workbook(template_path)
        worksheet = workbook.active
        clear_worksheet_values(worksheet)
    else:
        workbook = Workbook()
        worksheet = workbook.active
        worksheet.title = "Sheet1"

    worksheet.cell(1, 1, "时间\\到药材中心的距离")
    for column, r_cm in enumerate(radius_cm, start=2):
        cell = worksheet.cell(1, column, float(round(r_cm, 10)))
        cell.number_format = "0.0"

    for row, t_s in enumerate(times_s, start=2):
        time_cell = worksheet.cell(row, 1, int(round(float(t_s))))
        time_cell.number_format = "0"
        for column, value in enumerate(moisture[row - 2], start=2):
            cell = worksheet.cell(row, column, round(float(value), 4))
            cell.number_format = "0.0000"

    header_fill = PatternFill("solid", fgColor="D9EAF7")
    header_font = Font(name="Arial", size=10, bold=True, color="000000")
    body_font = Font(name="Arial", size=10, color="000000")
    center = Alignment(horizontal="center", vertical="center")

    max_row = times_s.size + 1
    max_column = radius_cm.size + 1
    for row in worksheet.iter_rows(
        min_row=1, max_row=max_row, min_col=1, max_col=max_column
    ):
        for cell in row:
            cell.font = body_font
            cell.alignment = center

    for cell in worksheet[1][:max_column]:
        cell.fill = header_fill
        cell.font = header_font

    worksheet.column_dimensions["A"].width = 24
    for column in range(2, max_column + 1):
        worksheet.column_dimensions[
            worksheet.cell(1, column).column_letter
        ].width = 11
    worksheet.freeze_panes = "B2"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output_path)
    workbook.close()


# ============================================================
# 7. 结果检查和论文表 5 输出
# ============================================================


def validate_solution(result: DryingSolution) -> None:
    """检查阈值、最慢干燥位置和基本物理范围。"""
    n = result.model.n_nodes
    temperature = result.event_state[:n]
    moisture = result.event_state[n:]
    critical = result.model.config.critical_moisture

    if not np.all(np.isfinite(result.event_state)):
        raise FloatingPointError("事件状态含有 NaN 或无穷值。")
    if np.min(moisture) < -1.0e-8:
        raise ValueError("计算出现明显负含水率，请收紧误差或检查模型。")

    maximum = float(np.max(moisture))
    maximum_index = int(np.argmax(moisture))
    threshold_error = abs(maximum - critical)
    if threshold_error > 5.0e-7:
        raise ValueError(
            f"事件定位误差过大：max(C)={maximum:.10f}，阈值={critical:.10f}。"
        )

    monotonic_tolerance = 2.0e-7
    if np.any(np.diff(moisture) > monotonic_tolerance):
        print("警告：结束时含水率并非严格从圆心向表面递减，请进行网格检查。")

    r_max_cm = result.model.mesh.r_m[maximum_index] * 100.0
    print(f"结束时最大含水率：{maximum:.10f} kg/kg")
    print(f"最大含水率位置：r = {r_max_cm:.8f} cm")
    print(
        f"结束时温度范围：{np.min(temperature):.6f}～"
        f"{np.max(temperature):.6f} °C"
    )


def print_table5(result: DryingSolution) -> None:
    """按题目表 5 的时间和位置在终端打印水分浓度。"""
    drying_hours = result.drying_time_s / 3600.0
    regular_hours = np.arange(6.0, drying_hours - 1.0e-12, 6.0)
    table_hours = np.concatenate((regular_hours, [drying_hours]))
    table_times_s = table_hours * 3600.0
    radius_cm = np.arange(0.0, result.model.config.radius_m * 100.0 + 0.25, 0.5)
    radius_m = radius_cm / 100.0
    profiles = interpolate_moisture_profiles(result, table_times_s, radius_m)

    print("\n表 5  药材烘干过程的水分浓度（kg/kg）")
    header = "时间/h" + "".join(f"{value:>12.1f} cm" for value in radius_cm)
    print(header)
    for index, (hour, row) in enumerate(zip(table_hours, profiles)):
        label = f"{hour:.4f}" if index == len(table_hours) - 1 else f"{hour:.1f}"
        values = "".join(f"{value:>15.4f}" for value in row)
        print(f"{label:>7}{values}")


def check_result_workbook(
    path: Path,
    expected_rows: int,
    expected_columns: int,
    critical_moisture: float,
) -> None:
    """重新打开输出文件，检查尺寸、数值和最后一行。"""
    workbook = load_workbook(path, data_only=True, read_only=True)
    worksheet = workbook.active

    if worksheet.max_row != expected_rows:
        raise ValueError(
            f"result3 行数错误：实际 {worksheet.max_row}，预期 {expected_rows}。"
        )
    if worksheet.max_column != expected_columns:
        raise ValueError(
            f"result3 列数错误：实际 {worksheet.max_column}，"
            f"预期 {expected_columns}。"
        )

    last_values = [
        worksheet.cell(worksheet.max_row, column).value
        for column in range(2, worksheet.max_column + 1)
    ]
    if any(value is None for value in last_values):
        raise ValueError("result3 最后一行存在空值。")
    if max(float(value) for value in last_values) > critical_moisture + 5.0e-5:
        raise ValueError("result3 最后一行尚未满足烘干阈值。")

    workbook.close()


# ============================================================
# 8. 命令行入口
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="问题 3：非均匀 FVM + BDF 求解药材烘干时间。"
    )
    parser.add_argument(
        "--attachment1",
        type=Path,
        default=Path("file/附件1.xlsx"),
        help="附件 1 路径。",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("file/附件3/result3.xlsx"),
        help="result3 模板路径；不存在时自动新建。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/q3/result3.xlsx"),
        help="结果文件路径。",
    )
    parser.add_argument(
        "--grid-intervals",
        type=int,
        default=1280,
        help="细网格区间数；正式提交推荐 1280。",
    )
    parser.add_argument(
        "--stretch-power",
        type=float,
        default=2.0,
        help="表面加密幂指数，推荐 2。",
    )
    parser.add_argument(
        "--max-hours",
        type=float,
        default=96.0,
        help="最大积分时长/h。",
    )
    parser.add_argument(
        "--max-step",
        type=float,
        default=60.0,
        help="BDF 最大时间步/s。",
    )
    parser.add_argument(
        "--plateau-window-seconds",
        type=float,
        default=3600.0,
        help="用附件末尾多少秒的均值估计恒温恒湿平台。",
    )
    parser.add_argument(
        "--plateau-temperature",
        type=float,
        default=None,
        help="4 h 后烘房温度/°C；默认取附件最后一小时均值。",
    )
    parser.add_argument(
        "--plateau-moisture",
        type=float,
        default=None,
        help="4 h 后烘房水分浓度；默认取附件最后一小时均值。",
    )
    parser.add_argument(
        "--h-heat",
        type=float,
        default=25.0,
        help="对流换热系数 W/(m²·K)。",
    )
    parser.add_argument(
        "--h-mass",
        type=float,
        default=8.0e-7,
        help="对流传质系数 m/s。",
    )
    parser.add_argument(
        "--skip-grid-check",
        action="store_true",
        help="跳过 N/2 粗网格计算，只运行细网格。",
    )
    return parser.parse_args()


def make_config(args: argparse.Namespace, grid_intervals: int) -> SolverConfig:
    return SolverConfig(
        heat_transfer_coefficient=args.h_heat,
        mass_transfer_coefficient=args.h_mass,
        grid_intervals=grid_intervals,
        stretch_power=args.stretch_power,
        max_hours=args.max_hours,
        max_step_s=args.max_step,
    )


def run_one_grid(
    boundary: AmbientBoundary,
    config: SolverConfig,
    label: str,
) -> DryingSolution:
    model = DryingModel(config, boundary)
    mesh = model.mesh
    print(
        f"\n[{label}] N={config.grid_intervals}，节点数={model.n_nodes}，"
        f"最大网格宽度={np.max(mesh.node_distance_m) * 100:.8f} cm，"
        f"最小网格宽度={np.min(mesh.node_distance_m) * 100:.8f} cm"
    )
    result = solve_until_dry(model)
    print(
        f"[{label}] 烘干时间={result.drying_time_s / 3600.0:.9f} h，"
        f"BDF 内部步数={len(result.ode_solution.t)}，"
        f"函数计算次数={result.ode_solution.nfev}，"
        f"耗时={result.wall_time_s:.2f} s"
    )
    return result


def main() -> None:
    args = parse_args()
    if args.grid_intervals < 40:
        raise ValueError("网格过粗；grid_intervals 不应小于 40。")
    if args.max_hours <= 0.0 or args.max_step <= 0.0:
        raise ValueError("max_hours 和 max_step 必须为正数。")

    boundary = read_attachment1(
        path=args.attachment1,
        plateau_window_s=args.plateau_window_seconds,
        plateau_temperature_c=args.plateau_temperature,
        plateau_moisture=args.plateau_moisture,
    )
    print(f"附件 1 时间范围：0～{boundary.time_s[-1]:.0f} s")
    print(
        "附件时域结束后的边界："
        f"T∞={boundary.plateau_temperature_c:.9f} °C，"
        f"C∞={boundary.plateau_moisture:.9f} kg/kg"
    )

    coarse_result: DryingSolution | None = None
    if not args.skip_grid_check:
        coarse_intervals = args.grid_intervals // 2
        coarse_config = make_config(args, coarse_intervals)
        coarse_result = run_one_grid(boundary, coarse_config, "粗网格")

    fine_config = make_config(args, args.grid_intervals)
    fine_result = run_one_grid(boundary, fine_config, "细网格")
    validate_solution(fine_result)

    if coarse_result is not None:
        coarse_h = coarse_result.drying_time_s / 3600.0
        fine_h = fine_result.drying_time_s / 3600.0
        # 非均匀有限体积格式的空间误差按二阶估计。
        richardson_h = fine_h + (fine_h - coarse_h) / (2.0**2 - 1.0)
        difference_minutes = abs(fine_h - coarse_h) * 60.0
        print(f"粗细网格结束时间相差：{difference_minutes:.6f} min")
        print(f"Richardson 外推烘干时间：{richardson_h:.9f} h")
        print(f"论文四位小数参考值：{richardson_h:.4f} h")

    # result3：从 60 s 开始，每 60 s 一行，保存到第一个晚于事件的完整分钟。
    output_end_s = int(math.ceil(fine_result.drying_time_s / 60.0) * 60)
    output_times_s = np.arange(60.0, output_end_s + 0.1, 60.0)
    radius_cm = np.round(
        np.arange(0.0, fine_config.radius_m * 100.0 + 0.05, 0.1), 10
    )
    radius_m = radius_cm / 100.0
    moisture_output = interpolate_moisture_profiles(
        fine_result, output_times_s, radius_m
    )

    if np.max(moisture_output[-1]) >= fine_config.critical_moisture:
        raise ValueError("最后一个整分钟的全域含水率尚未严格低于阈值。")

    write_result3(
        template_path=args.template,
        output_path=args.output,
        times_s=output_times_s,
        radius_cm=radius_cm,
        moisture=moisture_output,
    )
    check_result_workbook(
        path=args.output,
        expected_rows=output_times_s.size + 1,
        expected_columns=radius_cm.size + 1,
        critical_moisture=fine_config.critical_moisture,
    )

    print_table5(fine_result)
    print(f"\n连续事件时间：{fine_result.drying_time_s:.6f} s")
    print(f"连续事件时间：{fine_result.drying_time_s / 3600.0:.9f} h")
    print(f"result3 最后时刻：{output_end_s} s = {output_end_s / 3600.0:.9f} h")
    print(f"result3 数据尺寸：{output_times_s.size} 行 × {radius_cm.size} 个位置")
    print(f"结果文件：{args.output.resolve()}")


if __name__ == "__main__":
    main()
