
# Численное решение волнового уравнения
# u_tt = a^2 * u_xx + f(x,t)
# Используется явная трёхслойная схема

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def solve_lab_11():
    """Решение волнового уравнения (Лаб. 11)"""
    print("=" * 60)
    print("ЛАБОРАТОРНАЯ РАБОТА 11: Волновое уравнение")
    print("=" * 60)

    # Параметры
    L = 1.0
    T = 1.0
    a = 1.0
    N = 80
    J = 100

    h = L / N
    tau = T / J
    mu = a * tau / h  # Число Куранта

    print(f"\nПараметры сетки:")
    print(f"  h = {h:.6f}, τ = {tau:.6f}")
    print(f"  Число Куранта μ = a*τ/h = {mu:.6f}")
    print(f"  Устойчивость (|μ| ≤ 1): {abs(mu) <= 1}")

    x = np.linspace(0, L, N + 1)
    t = np.linspace(0, T, J + 1)

    u = np.zeros((N + 1, J + 1))

    # Начальное условие: гауссовский импульс
    u[:, 0] = np.exp(-100 * (x - 0.5) ** 2)

    # Второй слой по времени (Taylor разложение)
    for i in range(1, N):
        u_xx = (u[i + 1, 0] - 2 * u[i, 0] + u[i - 1, 0]) / h ** 2
        u[i, 1] = u[i, 0] + 0.5 * mu ** 2 * u_xx

    u[0, :] = 0
    u[N, :] = 0

    # Явная схема
    mu_sq = mu ** 2
    for j in range(1, J):
        for i in range(1, N):
            u[i, j + 1] = (2 * u[i, j] - u[i, j - 1] +
                           mu_sq * (u[i + 1, j] - 2 * u[i, j] + u[i - 1, j]))
        if (j + 1) % 25 == 0:
            print(f"  Шаг {j + 1}/{J}, t = {t[j + 1]:.4f}")

    # Визуализация
    X, T_mesh = np.meshgrid(x, t)

    fig = plt.figure(figsize=(14, 5))

    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X, T_mesh, u.T, cmap='viridis', edgecolor='none')
    ax1.set_xlabel('x')
    ax1.set_ylabel('t')
    ax1.set_zlabel('u(x,t)')
    ax1.set_title('Лаб.11: Волновое уравнение (3D)')
    fig.colorbar(surf, ax=ax1, shrink=0.5)

    ax2 = fig.add_subplot(122)
    im = ax2.contourf(X, T_mesh, u.T, levels=20, cmap='viridis')
    ax2.set_xlabel('x')
    ax2.set_ylabel('t')
    ax2.set_title('Лаб.11: Контурный график')
    fig.colorbar(im, ax=ax2)

    plt.tight_layout()
    plt.savefig('lab_11_wave.png', dpi=150, bbox_inches='tight')
    print("\nГрафик сохранён: lab_11_wave.png")
    plt.show()

    return u, x, t


if __name__ == "__main__":
    u, x, t = solve_lab_11()