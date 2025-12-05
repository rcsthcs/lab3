# ЛАБОРАТОРНАЯ РАБОТА 9
# Трёхслойные разностные схемы
# Уравнение колебаний (волновое): u_tt = a^2 * u_xx
# Граничные условия: u(0,t) = 0, u(1,t) = 0
# Начальные условия: u(x,0) = sin(πx), u_t(x,0) = 0

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def solve_wave_equation_explicit():
    """Решение волнового уравнения ЯВНОЙ трёхслойной схемой"""
    print("=" * 60)
    print("ЛАБОРАТОРНАЯ РАБОТА 9: Волновое уравнение (явная схема)")
    print("=" * 60)

    # Параметры
    L = 1.0  # Длина интервала [0, L]
    T = 2.0  # Время [0, T]
    a = 1.0  # Скорость волны
    N = 100  # Пространственные шаги
    M = 200  # Временные шаги

    h = L / N
    tau = T / M
    c_courant = a * tau / h  # Число Куранта

    print(f"\nПараметры сетки:")
    print(f"  Пространственный шаг h = {h:.6f}")
    print(f"  Временной шаг τ = {tau:.6f}")
    print(f"  Число Куранта c = a*τ/h = {c_courant:.6f}")
    print(f"  (Явная схема устойчива при |c| ≤ 1: {abs(c_courant) <= 1})")

    x = np.linspace(0, L, N + 1)
    t = np.linspace(0, T, M + 1)

    # Инициализация решения: u[i, j] = u(x_i, t_j)
    u = np.zeros((N + 1, M + 1))

    # Начальное условие: u(x, 0) = sin(πx)
    u[:, 0] = np.sin(np.pi * x)
    print(f"\nНачальное условие: u(x,0) = sin(πx)")

    # Второе начальное условие: u_t(x, 0) = 0
    # Используем разложение Тейлора: u(x, τ) ≈ u(x,0) + τ*u_t(x,0) + τ²/2*u_tt(x,0)
    # u_tt(x,0) = a²*u_xx(x,0)
    for i in range(1, N):
        u_xx = (u[i + 1, 0] - 2 * u[i, 0] + u[i - 1, 0]) / (h ** 2)
        u[i, 1] = u[i, 0] + (a * tau) ** 2 / 2 * u_xx

    # Граничные условия: u(0,t) = 0, u(L,t) = 0
    u[0, :] = 0
    u[N, :] = 0

    print(f"Граничные условия: u(0,t) = u(1,t) = 0")

    # Явная трёхслойная схема
    # u[i,j+1] = 2*u[i,j] - u[i,j-1] + c²*(u[i+1,j] - 2*u[i,j] + u[i-1,j])
    # где c = a*τ/h

    print("\nВычисление решения...")
    gamma_sq = (c_courant) ** 2

    for j in range(1, M):
        for i in range(1, N):
            u[i, j + 1] = (2 * u[i, j] - u[i, j - 1] +
                           gamma_sq * (u[i + 1, j] - 2 * u[i, j] + u[i - 1, j]))

        if (j + 1) % 50 == 0:
            print(f"  Шаг {j + 1}/{M} завершён, t = {t[j + 1]:.4f}")

    # Визуализация 1: 3D поверхность
    print("\nСоздание визуализации...")
    X, T_grid = np.meshgrid(x, t)

    fig = plt.figure(figsize=(14, 5))

    # 3D график
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X, T_grid, u.T, cmap='RdYlBu_r', edgecolor='none', alpha=0.9)
    ax1.set_xlabel('x')
    ax1.set_ylabel('t')
    ax1.set_zlabel('u(x,t)')
    ax1.set_title('Лаб.9: Волновое уравнение (3D)')
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)

    # Тепловая карта
    ax2 = fig.add_subplot(122)
    levels = np.linspace(u.min(), u.max(), 30)
    im = ax2.contourf(X, T_grid, u.T, levels=levels, cmap='RdYlBu_r')
    ax2.set_xlabel('x')
    ax2.set_ylabel('t')
    ax2.set_title('Лаб.9: Волновое уравнение (тепловая карта)')
    fig.colorbar(im, ax=ax2, label='u(x,t)')

    plt.tight_layout()
    plt.savefig('lab_9_wave_explicit.png', dpi=150, bbox_inches='tight')
    print("  График сохранён: lab_9_wave_explicit.png")
    plt.show()

    # Визуализация 2: Профили в разные моменты времени
    fig, ax = plt.subplots(figsize=(10, 6))
    time_indices = [0, M // 4, M // 2, 3 * M // 4, M]
    colors = plt.cm.viridis(np.linspace(0, 1, len(time_indices)))

    for idx, j in enumerate(time_indices):
        ax.plot(x, u[:, j], 'o-', label=f't = {t[j]:.3f}',
                color=colors[idx], linewidth=2, markersize=4)

    ax.set_xlabel('x')
    ax.set_ylabel('u(x,t)')
    ax.set_title('Лаб.9: Профили волны в разные моменты времени')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('lab_9_wave_profiles.png', dpi=150, bbox_inches='tight')
    print("  График сохранён: lab_9_wave_profiles.png")
    plt.show()

    # Визуализация 3: Анимация-подобное представление
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    snapshot_times = [0, M // 3, 2 * M // 3, M]
    for idx, j in enumerate(snapshot_times):
        ax = axes[idx]
        ax.fill_between(x, u[:, j], alpha=0.3, color='blue')
        ax.plot(x, u[:, j], 'b-', linewidth=2)
        ax.set_xlabel('x')
        ax.set_ylabel('u(x,t)')
        ax.set_title(f'Решение в момент t = {t[j]:.3f}')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-1.2, 1.2])

    plt.tight_layout()
    plt.savefig('lab_9_wave_snapshots.png', dpi=150, bbox_inches='tight')
    print("  График сохранён: lab_9_wave_snapshots.png")
    plt.show()

    print("\nРешение завершено успешно!")
    return u, x, t


if __name__ == "__main__":
    u, x, t = solve_wave_equation_explicit()