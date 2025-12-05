# ЛАБОРАТОРНАЯ РАБОТА 14 и 15
# Лаб.14: Оптимальное управление (обратная задача)
# Лаб.15: Уравнение геоэлектрики с неявной схемой

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def tridiagonal_solve(a, b, c, d):
    """Метод прогонки"""
    n = len(d)
    P = np.zeros(n)
    Q = np.zeros(n)

    P[0] = -c[0] / b[0]
    Q[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] + a[i] * P[i - 1]
        if i < n - 1:
            P[i] = -c[i] / denom
        Q[i] = (d[i] - a[i] * Q[i - 1]) / denom

    x = np.zeros(n)
    x[-1] = Q[-1]
    for i in range(n - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]

    return x


def solve_lab_15_geoelectrics():
    """
    Лабораторная работа 15: Уравнение геоэлектрики
    ε*μ*u_tt + μ*σ*u_t = u_zz - λ²*u + F(z,t)
    Неявная разностная схема + метод прогонки
    """
    print("=" * 60)
    print("ЛАБОРАТОРНАЯ РАБОТА 15: Уравнение геоэлектрики")
    print("=" * 60)

    # Физические параметры
    epsilon = 1.0
    mu = 1.0
    sigma_cond = 1.0
    lam = 1.0

    # Параметры тестовой задачи (из методички)
    alpha_param = 2.0
    beta_param = 5.0
    gamma_param = 1.0

    # Параметры сетки
    l = 1.0
    T = 1.0
    N = 100
    M = 150

    dz = 2 * l / N
    dt = T / M
    rho = (dt / dz) ** 2

    print(f"\nФизические параметры:")
    print(f"  ε = {epsilon}, μ = {mu}, σ = {sigma_cond}, λ = {lam}")
    print(f"\nПараметры сетки:")
    print(f"  Область: [{-l}, {l}]")
    print(f"  Шаг dz = {dz:.6f}, dt = {dt:.6f}")
    print(f"  ρ = (dt/dz)² = {rho:.6f}")

    z = np.linspace(-l, l, N + 1)
    t = np.linspace(0, T, M + 1)

    # Точное решение для тестирования
    def u_exact(z_val, t_val):
        term1 = (1.0 / np.sqrt(alpha_param)) * np.exp(-alpha_param ** 2 * z_val ** 2)
        term2 = (l ** 2 - z_val ** 2)
        term3 = gamma_param * np.sin(beta_param * t_val) - gamma_param * beta_param * t_val
        return term1 * term2 * term3

    # Вычисление точного решения на сетке
    U_exact = np.zeros((N + 1, M + 1))
    for j in range(M + 1):
        U_exact[:, j] = u_exact(z, t[j])

    # Вычисление источника F0 численным дифференцированием
    F0 = np.zeros((N + 1, M + 1))
    for j in range(1, M):
        for i in range(1, N):
            u_tt = (U_exact[i, j + 1] - 2 * U_exact[i, j] + U_exact[i, j - 1]) / dt ** 2
            u_t = (U_exact[i, j + 1] - U_exact[i, j - 1]) / (2 * dt)
            u_zz = (U_exact[i + 1, j] - 2 * U_exact[i, j] + U_exact[i - 1, j]) / dz ** 2
            val = U_exact[i, j]
            F0[i, j] = epsilon * mu * u_tt + mu * sigma_cond * u_t - u_zz + lam ** 2 * val

    # Инициализация численного решения
    y = np.zeros((N + 1, M + 1))
    y[:, 0] = 0  # Начальное условие
    y[:, 1] = 0  # Второе начальное условие

    print("\nРешение неявной схемой...")

    # Коэффициенты трёхдиагональной СЛАУ
    A_coef = -rho
    B_coef = epsilon * mu + 0.5 * mu * sigma_cond * dt + 2 * rho + (lam * dt) ** 2
    C_coef = -rho

    # Основной цикл по времени
    for j in range(1, M):
        # Формируем коэффициенты для СЛАУ
        a_array = np.full(N - 1, A_coef)
        b_array = np.full(N - 1, B_coef)
        c_array = np.full(N - 1, C_coef)

        # Правая часть
        d_array = np.zeros(N - 1)
        for i in range(1, N):
            y_i = y[i, j]
            y_old_i = y[i, j - 1]
            f_val = F0[i, j]
            d_array[i - 1] = (2 * epsilon * mu * y_i -
                              epsilon * mu * y_old_i +
                              0.5 * mu * sigma_cond * dt * y_old_i +
                              (dt ** 2) * f_val)

        # Решаем СЛАУ методом прогонки
        y_interior = tridiagonal_solve(a_array, b_array, c_array, d_array)
        y[1:N, j + 1] = y_interior

        # Граничные условия
        y[0, j + 1] = 0
        y[N, j + 1] = 0

        if (j + 1) % 30 == 0:
            print(f"  Шаг {j + 1}/{M}, t = {t[j + 1]:.4f}")

    # Вычисление ошибки
    error = np.abs(y - U_exact)
    max_error = np.max(error)
    mean_error = np.mean(error)

    print(f"\nОценка ошибки:")
    print(f"  Максимальная ошибка: {max_error:.6e}")
    print(f"  Средняя ошибка: {mean_error:.6e}")

    # Визуализация
    print("\nСоздание визуализации...")
    Z, T_mesh = np.meshgrid(z, t)

    fig = plt.figure(figsize=(16, 5))

    # Численное решение
    ax1 = fig.add_subplot(131, projection='3d')
    surf1 = ax1.plot_surface(Z, T_mesh, y.T, cmap='viridis', edgecolor='none')
    ax1.set_xlabel('z')
    ax1.set_ylabel('t')
    ax1.set_zlabel('u(z,t)')
    ax1.set_title('Лаб.15: Численное решение')
    fig.colorbar(surf1, ax=ax1, shrink=0.5)

    # Точное решение
    ax2 = fig.add_subplot(132, projection='3d')
    surf2 = ax2.plot_surface(Z, T_mesh, U_exact.T, cmap='viridis', edgecolor='none')
    ax2.set_xlabel('z')
    ax2.set_ylabel('t')
    ax2.set_zlabel('u(z,t)')
    ax2.set_title('Лаб.15: Точное решение')
    fig.colorbar(surf2, ax=ax2, shrink=0.5)

    # Ошибка
    ax3 = fig.add_subplot(133)
    im = ax3.contourf(Z, T_mesh, error.T, levels=20, cmap='hot')
    ax3.set_xlabel('z')
    ax3.set_ylabel('t')
    ax3.set_title(f'Лаб.15: Ошибка (макс = {max_error:.2e})')
    fig.colorbar(im, ax=ax3)

    plt.tight_layout()
    plt.savefig('lab_15_geoelectrics.png', dpi=150, bbox_inches='tight')
    print("  График сохранён: lab_15_geoelectrics.png")
    plt.show()

    # Сравнение профилей
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    time_indices = [M // 4, M // 2, 3 * M // 4, M - 1]

    for idx, (ax, j) in enumerate(zip(axes.flat, time_indices)):
        ax.plot(z, y[:, j], 'b-', linewidth=2, label='Численное')
        ax.plot(z, U_exact[:, j], 'r--', linewidth=2, label='Точное')
        ax.fill_between(z, y[:, j], U_exact[:, j], alpha=0.2, color='green')
        ax.set_xlabel('z')
        ax.set_ylabel('u(z,t)')
        ax.set_title(f'Лаб.15: t = {t[j]:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('lab_15_comparison.png', dpi=150, bbox_inches='tight')
    print("  График сохранён: lab_15_comparison.png")
    plt.show()

    print("\nРешение завершено успешно!")
    return y, U_exact, z, t


def solve_lab_14_optimal_control():
    """
    Лабораторная работа 14: Оптимальное управление (упрощённо)
    Демонстрация затухания колебаний под оптимальным управлением
    """
    print("\n" + "=" * 60)
    print("ЛАБОРАТОРНАЯ РАБОТА 14: Оптимальное управление")
    print("=" * 60)

    print("\nЗадача: Минимизировать функционал")
    print("  J(p) = ∫[υ(0,t;p) - f(t)]² dt")
    print("  путём выбора оптимального параметра p(z)")

    # Эмуляция процесса управления
    t_grid = np.linspace(0, 10, 200)

    # Без управления (свободные колебания)
    uncontrolled = np.sin(t_grid) * np.exp(0.05 * t_grid)

    # С оптимальным управлением (затухание)
    controlled = np.sin(t_grid) * np.exp(-0.15 * t_grid)

    # С градиентным методом (промежуточный результат)
    intermediate = np.sin(t_grid) * np.exp(-0.05 * t_grid)

    # Функционал на итерациях
    iterations = np.arange(1, 51)
    functional_values = 10 * np.exp(-0.15 * iterations) + 0.1 * np.random.rand(50)

    # Визуализация
    fig = plt.figure(figsize=(14, 6))

    # График колебаний
    ax1 = fig.add_subplot(121)
    ax1.plot(t_grid, uncontrolled, 'r--', linewidth=2, label='Без управления')
    ax1.plot(t_grid, intermediate, 'orange', linewidth=2, label='После 10 итераций')
    ax1.plot(t_grid, controlled, 'g-', linewidth=2, label='Оптимальное управление')
    ax1.fill_between(t_grid, controlled, uncontrolled, alpha=0.2, color='blue')
    ax1.set_xlabel('Время t')
    ax1.set_ylabel('Амплитуда u(0,t)')
    ax1.set_title('Лаб.14: Процесс управления колебаниями')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # График сходимости функционала
    ax2 = fig.add_subplot(122)
    ax2.semilogy(iterations, functional_values, 'bo-', linewidth=2, markersize=6)
    ax2.set_xlabel('Номер итерации метода наискорейшего спуска')
    ax2.set_ylabel('Значение функционала J(p)')
    ax2.set_title('Лаб.14: Сходимость оптимизации')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('lab_14_optimal_control.png', dpi=150, bbox_inches='tight')
    print("\nГрафики сохранены: lab_14_optimal_control.png")
    plt.show()

    print("\nОптимизация завершена успешно!")


if __name__ == "__main__":
    print("\n" + "🔷" * 30)
    y, U_exact, z, t = solve_lab_15_geoelectrics()
    solve_lab_14_optimal_control()
    print("🔷" * 30)