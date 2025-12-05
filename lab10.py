# lab_10_poisson_stability.py
import numpy as np
import matplotlib.pyplot as plt

def solve_poisson_dirichlet(N, max_iter=5000, tol=1e-8):
    """
    Решает задачу Дирихле для уравнения Пуассона:
        u_xx + u_yy = -f(x,y),  (x,y) ∈ (0,1)x(0,1)
        u = 0 на границе
    Берём точное решение u = sin(pi x) sin(pi y),
    тогда f(x,y) = 2*pi^2 * sin(pi x)*sin(pi y).
    Используем пятиузловую схему и метод Зейделя.
    """
    h = 1.0 / N
    x = np.linspace(0, 1, N+1)
    y = np.linspace(0, 1, N+1)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # точное решение и правая часть
    u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
    f = 2.0 * np.pi**2 * u_exact  # так что u_xx + u_yy = -f

    # численное решение
    u = np.zeros_like(u_exact)

    # граничные условия Дирихле: u = 0 по краю – уже выполнено (массив из нулей)

    # итерации Зейделя
    for k in range(max_iter):
        max_diff = 0.0
        # только внутренние узлы
        for i in range(1, N):
            for j in range(1, N):
                u_new = 0.25 * (
                    u[i+1, j] + u[i-1, j] +
                    u[i, j+1] + u[i, j-1] +
                    h**2 * f[i, j]
                )
                max_diff = max(max_diff, abs(u_new - u[i, j]))
                u[i, j] = u_new
        if max_diff < tol:
            # можно вывести, за сколько сошлось
            # print(f"N={N}: сошлось за {k+1} итераций, max_diff={max_diff:.2e}")
            break

    # ошибка
    error = np.abs(u - u_exact)
    max_err = np.max(error)
    return x, y, u, u_exact, max_err

def main():
    # проверка сходимости на трёх сетках
    Ns = [16, 32, 64]
    errors = []

    for N in Ns:
        print(f"Решаем задачу при N = {N} ...")
        x, y, u_num, u_ex, max_err = solve_poisson_dirichlet(N)
        errors.append(max_err)
        print(f"  N={N}, максимальная погрешность = {max_err:.4e}")

    # оценка порядка сходимости
    print("\nОценка порядка сходимости (по максимуму нормы):")
    for k in range(1, len(Ns)):
        rate = np.log(errors[k-1]/errors[k]) / np.log(2.0)
        print(f"  между N={Ns[k-1]} и N={Ns[k]}: порядок ≈ {rate:.2f}")

    # визуализация для самой тонкой сетки
    N = Ns[-1]
    x, y, u_num, u_ex, max_err = solve_poisson_dirichlet(N)
    X, Y = np.meshgrid(x, y, indexing="ij")

    fig = plt.figure(figsize=(15, 4))

    ax1 = fig.add_subplot(131, projection="3d")
    surf1 = ax1.plot_surface(X, Y, u_ex, cmap="viridis", edgecolor="none")
    ax1.set_title("Лаб.10: Точное решение u(x,y)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("u")
    fig.colorbar(surf1, ax=ax1, shrink=0.5)

    ax2 = fig.add_subplot(132, projection="3d")
    surf2 = ax2.plot_surface(X, Y, u_num, cmap="viridis", edgecolor="none")
    ax2.set_title("Лаб.10: Численное решение (5-точечная схема)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("u")
    fig.colorbar(surf2, ax=ax2, shrink=0.5)

    ax3 = fig.add_subplot(133)
    err = np.abs(u_num - u_ex)
    im = ax3.contourf(X, Y, err, levels=20, cmap="hot")
    ax3.set_title(f"Лаб.10: |u_num − u_exact| (max={max_err:.2e})")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    fig.colorbar(im, ax=ax3)

    plt.tight_layout()
    plt.show()

    # график зависимости ошибки от шага
    hs = [1.0/N for N in Ns]
    plt.figure(figsize=(6, 4))
    plt.loglog(hs, errors, "o-", label="||e||_∞")
    plt.loglog(hs, [errors[0]*(h/hs[0])**2 for h in hs], "--", label="O(h²)")
    plt.gca().invert_xaxis()
    plt.xlabel("Шаг h (log)")
    plt.ylabel("Макс. ошибка (log)")
    plt.title("Лаб.10: Сходимость схемы Пуассона")
    plt.grid(True, which="both", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
