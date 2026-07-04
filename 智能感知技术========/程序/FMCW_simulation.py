# -*- coding: utf-8 -*-
"""
FMCW 毫米波雷达测距、测速仿真 (Python 版)
原始 MATLAB 程序翻译 + 添加 3 个目标
目标1: 距离32m, 多普勒4.5kHz (原目标)
目标2: 距离60m, 多普勒2.0kHz (新增)
目标3: 距离85m, 多普勒8.0kHz (新增)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ======================== 中文字体设置 ========================
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# ======================== 系统参数 ========================
c = 3e8                # 光速
f0 = 24e9              # 载频 24GHz
B = 150e6              # 信号带宽 150MHz
T = 20e-6              # 信号时宽 20us
K = B / T              # 调频斜率
Num = 50               # 重复周期数（锯齿波个数）
fs = 500e6             # 采样频率 500MHz

# 时间轴（覆盖所有 Num 个周期）
t = np.arange(0, T * Num, 1 / fs)   # 总点数 = T*Num*fs = 500000

print(f"总采样点数: {len(t)}")
print(f"每个周期采样点数 T*fs = {int(T*fs)}")
print(f"调频斜率 K = {K:.3e} Hz/s")

# ======================== 矩形窗函数 (对应 rect.m) ========================
def rect(Tao, t_arr, fs_val):
    """
    矩形窗函数
    Tao: 窗宽度
    t_arr: 时间轴
    fs_val: 采样频率
    返回: 0 <= t < Tao 时为 1，其余为 0
    """
    W = np.ones(len(t_arr), dtype=complex)
    W[t_arr > Tao - 1 / fs_val] = 0
    W[t_arr < 0] = 0
    return W

# ======================== 目标参数（3个目标） ========================
# --- 目标1：原始目标 ---
Fd1 = 4.5e3                          # 多普勒频率 Hz
V1 = Fd1 / f0 * c / 2                # 速度 m/s
Dis0_1 = 32                          # 初始距离 m
Dis1 = Dis0_1 - V1 * t               # 距离随时间变化
DelayTime1 = (Dis0_1 - V1 * t) / c * 2  # 回波时延

# --- 目标2：新增（中距离，低速） ---
Fd2 = 2.0e3
V2 = Fd2 / f0 * c / 2
Dis0_2 = 60
Dis2 = Dis0_2 - V2 * t
DelayTime2 = (Dis0_2 - V2 * t) / c * 2

# --- 目标3：新增（远距离，高速） ---
Fd3 = 8.0e3
V3 = Fd3 / f0 * c / 2
Dis0_3 = 85
Dis3 = Dis0_3 - V3 * t
DelayTime3 = (Dis0_3 - V3 * t) / c * 2

print(f"\n目标1: 距离={Dis0_1}m, 速度={V1:.2f}m/s, 多普勒={Fd1/1e3:.1f}kHz")
print(f"目标2: 距离={Dis0_2}m, 速度={V2:.2f}m/s, 多普勒={Fd2/1e3:.1f}kHz")
print(f"目标3: 距离={Dis0_3}m, 速度={V3:.2f}m/s, 多普勒={Fd3/1e3:.1f}kHz")

# ======================= 演示：一个周期的发射信号和回波信号 =======================
# 发射窗
W_demo = rect(T, t, fs)
# 发射信号（单个周期演示）
St_demo = np.exp(1j * 2 * np.pi * (f0 * t + 0.5 * K * t**2)) * W_demo
# 接收窗（仅目标1演示）
Wr_demo = rect(T, t - DelayTime1, fs)
Sr_demo = np.exp(1j * 2 * np.pi * (f0 * (t - DelayTime1) + 0.5 * K * (t - DelayTime1)**2)) * Wr_demo

plt.figure(figsize=(10, 4))
plt.plot(t[:2000], np.real(Sr_demo[:2000]))
plt.xlabel('时间 (s)')
plt.ylabel('幅度')
plt.title('单个周期回波信号实部（前2000点）')
plt.grid(True)
plt.tight_layout()
plt.savefig('fig1_demo_received_signal.png', dpi=150)
plt.show()

# ======================== 目标建模：多周期发射与接收 ========================
St = np.zeros(len(t), dtype=complex)   # 总发射信号
Sr = np.zeros(len(t), dtype=complex)   # 总接收信号（3个目标叠加）

for Loop in range(1, Num + 1):
    shift = (Loop - 1) * T  # 当前周期的起始时间偏移
    t_shifted = t - shift   # 当前周期的本地时间

    # --- 发射信号 ---
    Wt = rect(T, t_shifted, fs)
    StTemp = np.exp(1j * 2 * np.pi * (f0 * t_shifted + 0.5 * K * t_shifted**2)) * Wt
    St = St + StTemp

    # --- 接收信号（3个目标叠加）---
    # 目标1
    Wr1 = rect(T, t_shifted - DelayTime1, fs)
    SrTemp1 = np.exp(1j * 2 * np.pi * (f0 * (t_shifted - DelayTime1) + 0.5 * K * (t_shifted - DelayTime1)**2)) * Wr1

    # 目标2
    Wr2 = rect(T, t_shifted - DelayTime2, fs)
    SrTemp2 = np.exp(1j * 2 * np.pi * (f0 * (t_shifted - DelayTime2) + 0.5 * K * (t_shifted - DelayTime2)**2)) * Wr2

    # 目标3
    Wr3 = rect(T, t_shifted - DelayTime3, fs)
    SrTemp3 = np.exp(1j * 2 * np.pi * (f0 * (t_shifted - DelayTime3) + 0.5 * K * (t_shifted - DelayTime3)**2)) * Wr3

    # 三目标回波叠加
    SrTemp = SrTemp1 + SrTemp2 + SrTemp3
    Sr = Sr + SrTemp

# 发射信号实部
plt.figure(figsize=(10, 4))
plt.plot(np.real(St[:5000]))
plt.xlabel('采样点')
plt.ylabel('幅度')
plt.title('发射信号实部（前5000点）')
plt.grid(True)
plt.tight_layout()
plt.savefig('fig2_transmitted_signal.png', dpi=150)
plt.show()

# ======================== 混频（去斜处理） ========================
# Smix = Sr .* conj(St)
Smix = Sr * np.conj(St)

# ======================== 信号处理 ========================
# 重构为快时间 x 慢时间矩阵：Num行 x (T*fs)列
N_fast = int(T * fs)  # 10000
Sd = Smix.reshape(Num, N_fast)  # (50, 10000)

# 去除无效点（最小距离对应的时延）
min_dis = min(Dis0_1, Dis0_2, Dis0_3)
remove_cols = int(np.floor(min_dis / c * 2 * fs))
print(f"\n去除前 {remove_cols} 个无效采样点（对应最小距离 {min_dis}m）")
Sd = Sd[:, remove_cols:]

# 显示第一个周期的差频信号
plt.figure(figsize=(10, 4))
plt.plot(np.real(Sd[0, :500]))
plt.xlabel('采样点')
plt.ylabel('幅度')
plt.title('第一个周期差频信号实部（前500点）')
plt.grid(True)
plt.tight_layout()
plt.savefig('fig3_beat_signal.png', dpi=150)
plt.show()

# ======================== 距离维 FFT ========================
SDis = np.fft.fftshift(np.fft.fft(Sd, n=N_fast, axis=1))

# 频率轴和距离轴
x_freq = np.arange(-fs / 2, fs / 2, fs / N_fast)       # 频率 Hz
d_range = x_freq / K * c / 2                             # 距离 m
y_chirp = np.arange(1, Num + 1)                          # 周期编号

# --- 距离维 3D mesh 图 ---
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
D_mesh, Y_mesh = np.meshgrid(d_range, y_chirp)
ax.plot_surface(D_mesh, Y_mesh, np.abs(SDis), cmap='jet', edgecolor='none')
ax.set_xlabel('距离 (m)')
ax.set_ylabel('周期编号')
ax.set_title('距离维 FFT 结果')
ax.view_init(elev=30, azim=-37.5)
plt.tight_layout()
plt.savefig('fig4_range_fft_mesh.png', dpi=150)
plt.show()

# --- 频率维切片（第一个周期） ---
plt.figure(figsize=(10, 4))
plt.plot(x_freq / 1e6, np.abs(SDis[0, :]))
plt.xlabel('频率 (MHz)')
plt.ylabel('幅度')
plt.title('第一个周期 — 频率维')
plt.grid(True)
plt.tight_layout()
plt.savefig('fig5_frequency_slice.png', dpi=150)
plt.show()

# --- 距离维切片（第一个周期） ---
plt.figure(figsize=(10, 4))
plt.plot(d_range, np.abs(SDis[0, :]))
plt.xlabel('距离 (m)')
plt.ylabel('幅度')
plt.title('第一个周期 — 距离维')
plt.grid(True)
plt.tight_layout()
plt.savefig('fig6_range_slice.png', dpi=150)
plt.show()

# ======================== 速度维 2D FFT（距离-多普勒图） ========================
N_doppler = 100   # 多普勒维 FFT 点数
N_range_fft = N_fast
Sfft = np.fft.fftshift(np.fft.fft2(Sd, s=(N_doppler, N_range_fft)))

# 距离轴（同上）
# 速度轴（多普勒频率）
y_doppler = np.arange(-1 / T / 2, 1 / T / 2, 1 / T / N_doppler)  # Doppler 频率 Hz
v_velocity = y_doppler * c / (2 * f0)                                # 速度 m/s

# --- 距离-多普勒 3D mesh 图 ---
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
D2_mesh, V2_mesh = np.meshgrid(d_range, y_doppler / 1e3)  # kHz
ax.plot_surface(D2_mesh, V2_mesh, np.abs(Sfft), cmap='jet', edgecolor='none')
ax.set_xlabel('距离 (m)')
ax.set_ylabel('多普勒频率 (kHz)')
ax.set_title('距离-多普勒 2D FFT 结果')
ax.view_init(elev=30, azim=-37.5)
plt.tight_layout()
plt.savefig('fig7_range_doppler_mesh.png', dpi=150)
plt.show()

# --- 在目标距离处切速度维 ---
# 自动找到 Sfft 中幅度最大的距离索引
max_row, max_col = np.unravel_index(np.argmax(np.abs(Sfft)), Sfft.shape)
print(f"\n峰值位置: 多普勒索引={max_row}, 距离索引={max_col}")
print(f"峰值对应距离≈{d_range[max_col]:.1f}m, 多普勒≈{y_doppler[max_row]/1e3:.2f}kHz")

# 在峰值距离处画速度维切片
plt.figure(figsize=(10, 4))
plt.plot(y_doppler / 1e3, np.abs(Sfft[:, max_col]))
plt.xlabel('多普勒频率 (kHz)')
plt.ylabel('幅度')
plt.title(f'距离 ≈ {d_range[max_col]:.1f} m 处的速度维切片')
plt.grid(True)
plt.tight_layout()
plt.savefig('fig8_velocity_slice.png', dpi=150)
plt.show()

# --- 在 32m 处切速度维（与原始 MATLAB 代码对应） ---
idx_32m = np.argmin(np.abs(d_range - 32))
plt.figure(figsize=(10, 4))
plt.plot(y_doppler / 1e3, np.abs(Sfft[:, idx_32m]))
plt.xlabel('多普勒频率 (kHz)')
plt.ylabel('幅度')
plt.title('距离 32m 处的速度维切片')
plt.grid(True)
plt.tight_layout()
plt.savefig('fig9_velocity_slice_32m.png', dpi=150)
plt.show()

# ======================== 汇总输出 ========================
print("\n" + "=" * 60)
print("仿真完成！生成的图片文件：")
print("  fig1_demo_received_signal.png  — 单个周期回波信号")
print("  fig2_transmitted_signal.png    — 发射信号")
print("  fig3_beat_signal.png           — 差频信号")
print("  fig4_range_fft_mesh.png        — 距离维 FFT 3D 图")
print("  fig5_frequency_slice.png       — 频率维切片")
print("  fig6_range_slice.png           — 距离维切片")
print("  fig7_range_doppler_mesh.png    — 距离-多普勒 3D 图")
print("  fig8_velocity_slice.png        — 峰值距离处速度维切片")
print("  fig9_velocity_slice_32m.png    — 32m 处速度维切片")
print("=" * 60)
print(f"\n三个目标参数：")
print(f"  目标1: 距离={Dis0_1}m, 速度={V1:.2f}m/s")
print(f"  目标2: 距离={Dis0_2}m, 速度={V2:.2f}m/s")
print(f"  目标3: 距离={Dis0_3}m, 速度={V3:.2f}m/s")
