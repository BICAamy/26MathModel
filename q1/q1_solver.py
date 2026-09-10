#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
2026 高教社杯 A题：药材的烘干问题 —— 问题1
一维圆柱径向有限体积法(FVM) + RK45 时间积分

默认从项目根目录运行：
    python q1/q1_solver.py

默认输入：
    file/附件1.xlsx

默认输出：
    output/result1.xlsx

依赖：
    numpy
    scipy
    openpyxl
"""

from pathlib import Path
import argparse
import numpy as np
from scipy.integrate import solve_ivp
from openpyxl import load_workbook, Workbook
from openpyxl.styles import Font, PatternFill, Alignment


# ============================================================
# 1. 读取附件1
# ============================================================

def read_attachment1(path: Path):
    """
    读取附件1.xlsx。

    预期前3列：
        A列：时间/s
        B列：烘房温度/°C
        C列：烘房水分浓度/(kg/kg)

    返回：
        time_data, temp_air_data, moist_air_data
    """
    wb = load_workbook(path, data_only=True)
    ws = wb[wb.sheetnames[0]]

    rows = list(ws.iter_rows(values_only=True))

    if len(rows) < 2:
        raise ValueError("附件1没有有效数据。")

    header = rows[0][:3]
    print("检测到附件1表头：", header)

    data = []
    for row in rows[1:]:
        if row[0] is None:
            continue
        try:
            t = float(row[0])
            T = float(row[1])
            C = float(row[2])
            data.append((t, T, C))
        except (TypeError, ValueError):
            continue

    if not data:
        raise ValueError("附件1中没有读取到有效数值数据。")

    arr = np.asarray(data, dtype=float)

    return arr[:, 0], arr[:, 1], arr[:, 2]


# ============================================================
# 2. 问题1数值求解
# ============================================================

def solve_q1(input_path: Path):
    # ---------------- 题目给定参数 ----------------
    rho = 820.0             # kg/m^3
    cp = 2600.0             # J/(kg·K)
    k = 0.36                # W/(m·K)
    h = 25.0                # W/(m^2·K)
    hm = 8.0e-7             # m/s

    # 圆柱半径
    R = 0.02                # m = 2 cm

    # 空间步长：题目要求每 0.1 cm 输出
    dr = 0.001              # m = 0.1 cm

    # 一共 21 个节点：0, 0.1, ..., 2.0 cm
    N = int(round(R / dr))
    r = np.arange(N + 1) * dr

    # 时间：题目要求 0~1800 s 每隔 1 s 输出
    t_eval = np.arange(0.0, 1800.0 + 1.0, 1.0)

    # ---------------- 读取附件1 ----------------
    time_data, Tair_data, Cair_data = read_attachment1(input_path)

    # 只截取问题1需要的 0~1800 s
    mask = (time_data >= 0.0) & (time_data <= 1800.0)
    time_q1 = time_data[mask]
    Tair_q1 = Tair_data[mask]
    Cair_q1 = Cair_data[mask]

    if len(time_q1) < 2:
        raise ValueError("附件1中 0~1800 s 的数据点不足。")

    if time_q1[0] > 0.0 or time_q1[-1] < 1800.0:
        raise ValueError(
            f"附件1必须覆盖 0~1800 s，当前覆盖范围为 "
            f"{time_q1[0]}~{time_q1[-1]} s"
        )

    # 外界烘房温度：分段线性插值
    def T_air(t):
        return float(np.interp(t, time_q1, Tair_q1))

    # 外界烘房水分浓度：分段线性插值
    def C_air(t):
        return float(np.interp(t, time_q1, Cair_q1))

    # 水分有效扩散系数
    def D_of_C(C):
        C_safe = np.maximum(np.asarray(C, dtype=float), 1e-12)
        return 7.0e-9 * np.exp(-0.89 / C_safe)

    # ========================================================
    # 有限体积法几何量
    # ========================================================

    # 取单位轴向长度 L=1 m。
    # 最终面积/体积比中 L 会自动约掉。
    V = np.empty(N + 1)

    for i in range(N + 1):
        r_in = max(0.0, r[i] - dr / 2.0)
        r_out = min(R, r[i] + dr / 2.0)
        V[i] = np.pi * (r_out**2 - r_in**2)

    # 相邻节点之间的圆柱侧面积
    A_face = 2.0 * np.pi * ((np.arange(N) + 0.5) * dr)

    # 最外表面积
    A_outer = 2.0 * np.pi * R

    # ========================================================
    # 温度场
    # ========================================================

    def rhs_temperature(t, T):
        """
        输入：
            t: 当前时间
            T: 21个径向节点当前温度

        输出：
            dTdt: 21个节点当前温度变化率
        """
        dTdt = np.zeros_like(T)
        Tinf = T_air(t)

        # ---------- 中心节点 r=0 ----------
        # 中心内侧面积为0，自然满足 dT/dr = 0
        q_right = k * A_face[0] * (T[1] - T[0]) / dr
        dTdt[0] = q_right / (rho * cp * V[0])

        # ---------- 内部节点 ----------
        for i in range(1, N):
            q_left = k * A_face[i - 1] * (T[i - 1] - T[i]) / dr
            q_right = k * A_face[i] * (T[i + 1] - T[i]) / dr

            dTdt[i] = (q_left + q_right) / (rho * cp * V[i])

        # ---------- 表面节点 r=R ----------
        q_left = k * A_face[N - 1] * (T[N - 1] - T[N]) / dr

        # 对流换热
        q_conv = h * A_outer * (Tinf - T[N])

        dTdt[N] = (q_left + q_conv) / (rho * cp * V[N])

        return dTdt

    # ========================================================
    # 水分浓度场
    # ========================================================

    def rhs_moisture(t, C):
        """
        输入：
            t: 当前时间
            C: 21个径向节点当前含水率

        输出：
            dCdt: 21个节点当前含水率变化率
        """
        dCdt = np.zeros_like(C)
        Cinf = C_air(t)

        # 每个节点当前的 D(C)
        D_node = D_of_C(C)

        # 相邻节点界面的扩散系数：调和平均
        D_face = (
            2.0 * D_node[:-1] * D_node[1:]
            / np.maximum(D_node[:-1] + D_node[1:], 1e-30)
        )

        # ---------- 中心节点 ----------
        j_right = (
            D_face[0]
            * A_face[0]
            * (C[1] - C[0])
            / dr
        )

        dCdt[0] = j_right / V[0]

        # ---------- 内部节点 ----------
        for i in range(1, N):
            j_left = (
                D_face[i - 1]
                * A_face[i - 1]
                * (C[i - 1] - C[i])
                / dr
            )

            j_right = (
                D_face[i]
                * A_face[i]
                * (C[i + 1] - C[i])
                / dr
            )

            dCdt[i] = (j_left + j_right) / V[i]

        # ---------- 表面节点 ----------
        j_left = (
            D_face[N - 1]
            * A_face[N - 1]
            * (C[N - 1] - C[N])
            / dr
        )

        # 对流传质
        j_conv = hm * A_outer * (Cinf - C[N])

        dCdt[N] = (j_left + j_conv) / V[N]

        return dCdt

    # ========================================================
    # 初始条件
    # ========================================================

    T0 = np.full(N + 1, 28.0)
    C0 = np.full(N + 1, 2.55)

    # ========================================================
    # 时间积分
    # ========================================================

    print("开始求解温度场……")

    sol_T = solve_ivp(
        rhs_temperature,
        (0.0, 1800.0),
        T0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )

    if not sol_T.success:
        raise RuntimeError("温度场求解失败：" + sol_T.message)

    print("温度场求解完成。")
    print("开始求解水分浓度场……")

    sol_C = solve_ivp(
        rhs_moisture,
        (0.0, 1800.0),
        C0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )

    if not sol_C.success:
        raise RuntimeError("水分场求解失败：" + sol_C.message)

    print("水分浓度场求解完成。")

    # 转成 shape=(1801,21)
    T = sol_T.y.T
    C = sol_C.y.T

    # 所有结果保留四位小数
    T = np.round(T, 4)
    C = np.round(C, 4)

    return t_eval.astype(int), r * 100.0, T, C


# ============================================================
# 3. 写出 result1.xlsx
# ============================================================

def write_result1(output_path: Path, times, r_cm, T, C):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wb = Workbook()

    # 删除默认Sheet并创建题目要求的两个工作表
    ws_T = wb.active
    ws_T.title = "温度"
    ws_C = wb.create_sheet("水分浓度")

    # 表头
    headers = ["时间/s"] + [round(float(x), 1) for x in r_cm]

    ws_T.append(headers)
    ws_C.append(headers)

    # 数据
    for i, t in enumerate(times):
        ws_T.append(
            [int(t)] + [float(v) for v in T[i]]
        )
        ws_C.append(
            [int(t)] + [float(v) for v in C[i]]
        )

    # 样式
    fill = PatternFill(
        fill_type="solid",
        fgColor="1F4E78"
    )
    white_font = Font(
        bold=True,
        color="FFFFFF"
    )
    center = Alignment(
        horizontal="center",
        vertical="center"
    )

    for ws in (ws_T, ws_C):
        # 表头
        for cell in ws[1]:
            cell.fill = fill
            cell.font = white_font
            cell.alignment = center

        # 冻结首行首列
        ws.freeze_panes = "B2"

        # 列宽
        ws.column_dimensions["A"].width = 11

        for col in range(2, 23):
            ws.column_dimensions[
                ws.cell(row=1, column=col).column_letter
            ].width = 9

        # 数值格式
        for row in ws.iter_rows(
            min_row=2,
            max_row=ws.max_row,
            min_col=2,
            max_col=22
        ):
            for cell in row:
                cell.number_format = "0.0000"
                cell.alignment = center

        for cell in ws["A"]:
            cell.alignment = center

    wb.save(output_path)


# ============================================================
# 4. 打印题目要求的表1、表2
# ============================================================

def print_required_tables(T, C):
    report_times = [
        100, 300, 600, 900,
        1200, 1500, 1800
    ]

    report_r_cm = [
        0.0, 0.5, 1.0, 1.5, 2.0
    ]

    # 因为空间步长是0.1cm
    idx = [
        int(round(x / 0.1))
        for x in report_r_cm
    ]

    print("\n==============================")
    print("表1：30分钟内药材的温度 / °C")
    print("==============================")

    print(
        "时间/s | "
        + " | ".join(
            f"{x:g}cm"
            for x in report_r_cm
        )
    )

    for t in report_times:
        print(
            f"{t:6d} | "
            + " | ".join(
                f"{T[t, j]:.4f}"
                for j in idx
            )
        )

    print("\n======================================")
    print("表2：30分钟内药材的水分浓度 / kg/kg")
    print("======================================")

    print(
        "时间/s | "
        + " | ".join(
            f"{x:g}cm"
            for x in report_r_cm
        )
    )

    for t in report_times:
        print(
            f"{t:6d} | "
            + " | ".join(
                f"{C[t, j]:.4f}"
                for j in idx
            )
        )


# ============================================================
# 5. 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="file/附件1.xlsx",
        help="附件1.xlsx路径"
    )

    parser.add_argument(
        "--output",
        default="output/q1/result1.xlsx",
        help="result1.xlsx输出路径"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(
            f"找不到输入文件：{input_path.resolve()}"
        )

    print("输入文件：", input_path.resolve())
    print("输出文件：", output_path.resolve())

    times, r_cm, T, C = solve_q1(input_path)

    print_required_tables(T, C)

    write_result1(
        output_path,
        times,
        r_cm,
        T,
        C
    )

    print("\n==============================")
    print("问题1求解完成")
    print("==============================")
    print("温度矩阵 shape =", T.shape)
    print("水分矩阵 shape =", C.shape)
    print("结果文件 =", output_path.resolve())


if __name__ == "__main__":
    main()
