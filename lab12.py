# ЛАБОРАТОРНАЯ РАБОТА 12
# Численное решение уравнения Пуассона в прямоугольнике
# u_xx + u_yy = -f(x,y)
# Граничные условия Дирихле: u(граница) = 0
# Метод: Итерация Зейделя (Либмана)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def solve_poisson_seidel():
    """Решение уравнения Пуассона методом Зейделя (Либмана)"""
    print("=" * 60)
    print("ЛАБОРАТОРНАЯ РАБОТА 12: Уравнение Пуассона")
    print("=" * 60)

    # Параметры
    lx = 1.0  # Размер области по x
    ly = 1.0  # Размер области по y
    Nx = 50  # Узлов по x
    Ny = 50  # Узлов по y

    hx = lx / Nx
    hy = ly / Ny

    print(f"\nПараметры сетки:")
    print(f"  Область [0, {lx}] x [0, {ly}]")
    print(f"  Узлов: {Nx + 1} x {Ny + 1}")
    print(f"  Шаг hx = {hx:.6f}, hy = {hy:.6f}")

    # Создание сетки
    x = np.linspace(0, lx, Nx + 1)
    y = np.linspace(0, ly, Ny + 1)
    X, Y = np.meshgrid(x, y)

    # Решение
    u = np.zeros((Nx + 1, Ny + 1))

    # Правая часть (источник): f(x,y) = -100 в центре, 0 везде
    f = np.zeros((Nx + 1, Ny + 1))
    # Источник в центре области
    center_i, center_j = Nx // 2, Ny // 2
    f[center_i, center_j] = 100.0

    # Граничные условия: u = 0 на всех границах
    u[0, :] = 0
    u[-1, :] = 0
    u[:, 0] = 0
    u[:, -1] = 0

    print("\nНачальные условия:")
    print(f"  Граничные: u = 0 везде")
    print(f"  Источник в центре: ({x[center_i]:.3f}, {y[center_j]:.3f})")

    # Итерационный метод Зейделя
    max_iter = 2000
    tolerance = 1e-5

    print("\nИтерационный процесс Зейделя...")
    residuals = []

    for iteration in range(max_iter):
        u_old = u.copy()
        residual_max = 0.0

        # Прямой проход (обновляем от (1,1) до (Nx-1, Ny-1))
        for i in range(1, Nx):
            for j in range(1, Ny):
                # Разностное уравнение Пуассона:
                # (u[i+1,j] - 2*u[i,j] + u[i-1,j])/hx² + (u[i,j+1] - 2*u[i,j] + u[i,j-1])/hy² = -f[i,j]
                # Переупорядочиваем для Зейделя (используем уже обновлённые значения)
                u[i, j] = (hy ** 2 * (u[i + 1, j] + u[i - 1, j]) +
                           hx ** 2 * (u[i, j + 1] + u[i, j - 1]) -
                           (hx ** 2 * hy ** 2) * f[i, j]) / (2 * (hx ** 2 + hy ** 2))

                residual_max = max(residual_max, abs(u[i, j] - u_old[i, j]))

        residuals.append(residual_max)

        if (iteration + 1) % 200 == 0:
            print(f"  Итерация {iteration + 1}, невязка = {residual_max:.2e}")

        if residual_max < tolerance:
            print(f"  Сошлось за {iteration + 1} итераций, невязка = {residual_max:.2e}")
            break

    # Визуализация 1: 3D поверхность решения
    print("\nСоздание визуализации...")
    fig = plt.figure(figsize=(14, 5))

    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X, Y, u, cmap='RdYlBu_r', edgecolor='none', alpha=0.9)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('u(x,y)')
    ax1.set_title('Лаб.12: Решение уравнения Пуассона (3D)')
    fig.colorbar(surf, ax=ax1, shrink=0.5)

    # Визуализация 2: Контурный график
    ax2 = fig.add_subplot(122)
    levels = np.linspace(u.min(), u.max(), 20)
    contourf = ax2.contourf(X, Y, u, levels=levels, cmap='RdYlBu_r')
    contour = ax2.contour(X, Y, u, levels=levels, colors='black', linewidths=0.5, alpha=0.3)
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Лаб.12: Контурный график')
    fig.colorbar(contourf, ax=ax2, label='u(x,y)')

    plt.tight_layout()
    plt.savefig('lab_12_poisson.png', dpi=150, bbox_inches='tight')
    print("  График сохранён: lab_12_poisson.png")
    plt.show()

    # Визуализация 3: Сходимость итераций
    fig, ax = plt.subplots(figsize=(10, 6))
    iterations_plot = range(1, len(residuals) + 1)
    ax.semilogy(iterations_plot, residuals, 'b-', linewidth=2)
    ax.axhline(y=tolerance, color='r', linestyle='--', label=f'Допуск = {tolerance:.2e}')
    ax.set_xlabel('Номер итерации')
    ax.set_ylabel('Максимальная невязка')
    ax.set_title('Лаб.12: Сходимость метода Зейделя')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig('lab_12_convergence.png', dpi=150, bbox_inches='tight')
    print("  График сохранён: lab_12_convergence.png")
    plt.show()

    print("\nРешение завершено успешно!")
    return u, x, y


if __name__ == "__main__":
    u, x, y = solve_poisson_seidel()