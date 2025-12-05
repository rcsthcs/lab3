# ЛАБОРАТОРНАЯ РАБОТА 8
# Неявная схема для уравнения теплопроводности. Метод прогонки.
# u_t = u_xx + x
# Граничные условия: u(0,t) = 0, u_x(1,t) = t
# Начальное условие: u(x,0) = sin(3πx/2)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def tridiagonal_solve(a, b, c, d):
    """Метод прогонки для решения СЛАУ с трёхдиагональной матрицей"""
    n = len(d)
    P = np.zeros(n)
    Q = np.zeros(n)

    # Прямой ход прогонки
    P[0] = -c[0] / b[0]
    Q[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] + a[i] * P[i - 1]
        if i < n - 1:
            P[i] = -c[i] / denom
        Q[i] = (d[i] - a[i] * Q[i - 1]) / denom

    # Обратный ход прогонки
    x = np.zeros(n)
    x[-1] = Q[-1]
    for i in range(n - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]

    return x


def solve_heat_implicit():
    """Решение уравнения теплопроводности НЕЯВНОЙ схемой"""
    print("=" * 60)
    print("ЛАБОРАТОРНАЯ РАБОТА 8: Неявная схема теплопроводности")
    print("=" * 60)

    # Параметры сетки
    L = 1.0  # Длина интервала [0, L]
    T = 0.5  # Время [0, T]
    N = 50  # Количество пространственных шагов
    J = 200  # Количество временных шагов

    h = L / N
    tau = T / J
    sigma = tau / (h ** 2)  # Число Куранта

    print(f"\nПараметры сетки:")
    print(f"  Пространственный шаг h = {h:.6f}")
    print(f"  Временной шаг τ = {tau:.6f}")
    print(f"  Число Куранта σ = τ/h² = {sigma:.6f}")
    print(f"  (Неявная схема устойчива при любом σ > 0)")

    x = np.linspace(0, L, N + 1)
    t = np.linspace(0, T, J + 1)

    # Инициализация решения
    u = np.zeros((N + 1, J + 1))

    # Начальное условие: u(x, 0) = sin(3πx/2)
    u[:, 0] = np.sin(3 * np.pi * x / 2)

    # Граничные условия: u(0, t) = 0
    u[0, :] = 0

    print("\nНачальные и граничные условия установлены.")
    print(f"  u(x, 0) = sin(3πx/2)")
    print(f"  u(0, t) = 0")
    print(f"  u_x(1, t) = t")

    # Основной цикл по времени (неявная схема)
    print("\nВычисление решения...")
    for j in range(J):
        # Коэффициенты СЛАУ (из неявной схемы)
        # (y_n^{j+1} - y_n^j) / τ = (y_{n-1}^{j+1} - 2y_n^{j+1} + y_{n+1}^{j+1}) / h^2 + x_n
        # Переупорядочиваем: σ·y_{n-1}^{j+1} - (1+2σ)·y_n^{j+1} + σ·y_{n+1}^{j+1} = -y_n^j - τ·x_n

        a = np.full(N - 1, sigma)  # Коэффициент при y_{i-1}
        b = np.full(N - 1, -(1 + 2 * sigma))  # Коэффициент при y_i
        c = np.full(N - 1, sigma)  # Коэффициент при y_{i+1}

        # Правая часть (для внутренних узлов)
        d = -u[1:N, j] - tau * x[1:N]

        # Модификация для левого граничного условия (y_0 = 0)
        d[0] -= sigma * u[0, j + 1]  # u[0, j+1] = 0

        # Модификация для правого граничного условия (y_N - y_{N-1} = -h*t)
        # Это условие Неймана третьего рода
        d[-1] -= sigma * (-h * t[j + 1])  # y_N = y_{N-1} - h*t[j+1]

        # Решаем СЛАУ методом прогонки
        u_interior = tridiagonal_solve(a, b, c, d)
        u[1:N, j + 1] = u_interior

        # Граничное условие справа: u_x(1,t) = t
        # Аппроксимация: (u_N - u_{N-1}) / h = t
        u[N, j + 1] = u[N - 1, j + 1] + h * t[j + 1]

        if (j + 1) % 50 == 0:
            print(f"  Шаг {j + 1}/{J} завершён, t = {t[j + 1]:.4f}")

    # Визуализация 1: 3D поверхность
    print("\nСоздание визуализации...")
    X, T_grid = np.meshgrid(x, t)

    fig = plt.figure(figsize=(14, 5))

    # 3D график
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X, T_grid, u.T, cmap='viridis', edgecolor='none', alpha=0.9)
    ax1.set_xlabel('x')
    ax1.set_ylabel('t')
    ax1.set_zlabel('u(x,t)')
    ax1.set_title('Лаб.8: Неявная схема теплопроводности (3D)')
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)

    # Тепловая карта
    ax2 = fig.add_subplot(122)
    im = ax2.contourf(X, T_grid, u.T, levels=20, cmap='RdYlBu_r')
    ax2.set_xlabel('x')
    ax2.set_ylabel('t')
    ax2.set_title('Лаб.8: Тепловая карта решения')
    fig.colorbar(im, ax=ax2, label='u(x,t)')

    plt.tight_layout()
    plt.savefig('lab_8_heat_implicit.png', dpi=150, bbox_inches='tight')
    print("  График сохранён: lab_8_heat_implicit.png")
    plt.show()

    # Визуализация 2: Профили решения на разных моментах времени
    fig, ax = plt.subplots(figsize=(10, 6))
    time_slices = [0, J // 4, J // 2, 3 * J // 4, J]
    colors = plt.cm.viridis(np.linspace(0, 1, len(time_slices)))

    for idx, j_idx in enumerate(time_slices):
        ax.plot(x, u[:, j_idx], 'o-', label=f't = {t[j_idx]:.3f}',
                color=colors[idx], linewidth=2, markersize=4)

    ax.set_xlabel('x')
    ax.set_ylabel('u(x,t)')
    ax.set_title('Лаб.8: Профили решения в разные моменты времени')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('lab_8_profiles.png', dpi=150, bbox_inches='tight')
    print("  График сохранён: lab_8_profiles.png")
    plt.show()

    print("\nРешение завершено успешно!")
    return u, x, t


if __name__ == "__main__":
    u, x, t = solve_heat_implicit()