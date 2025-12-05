# lab_13_quadratic_functional.py
import numpy as np
import matplotlib.pyplot as plt

def build_1d_problem(N):
    """
    Строим 1D‑задачу:
        -u''(x) = f(x), 0<x<1,  u(0)=u(1)=0
    Берём точное решение u_exact(x) = x*(1-x),
    тогда -u'' = 2, т.е. f(x) = 2.
    Возвращает:
        x_internal, K (матрица), f_vec, u_exact_internal
    """
    h = 1.0 / N
    x = np.linspace(0, 1, N+1)
    x_internal = x[1:-1]  # внутренние узлы

    # точное решение и правая часть
    u_exact = x * (1 - x)
    f_cont = 2.0 * np.ones_like(x)  # f(x)=2

    # число внутренних узлов
    n_int = N - 1

    # жёсткостная матрица (вторая производная с Дирихле)
    K = np.zeros((n_int, n_int))
    for i in range(n_int):
        K[i, i] = 2.0 / h**2
        if i > 0:
            K[i, i-1] = -1.0 / h**2
        if i < n_int - 1:
            K[i, i+1] = -1.0 / h**2

    # вектор правой части (учитывает u'' ≈ (u_{i-1}-2u_i+u_{i+1})/h² = -f_i)
    # => K u = f_vec, где f_vec ≈ f_i
    f_vec = f_cont[1:-1]

    return x_internal, K, f_vec, u_exact[1:-1]

def quadratic_functional(K, f_vec, u):
    """
    Квадратичный функционал:
        J(u) = 0.5 * u^T K u - f^T u
    """
    return 0.5 * u @ (K @ u) - f_vec @ u

def gradient_of_J(K, f_vec, u):
    """
    Градиент функционала:
        ∇J(u) = K u - f
    """
    return K @ u - f_vec

def solve_direct(K, f_vec):
    """Точное (в смысле дискретной задачи) решение Ku=f."""
    return np.linalg.solve(K, f_vec)

def gradient_descent(K, f_vec, u0, alpha=0.01, max_iter=5000, tol=1e-8):
    """
    Градиентный спуск для минимизации J(u).
    alpha — шаг по градиенту.
    """
    u = u0.copy()
    Js = []
    for k in range(max_iter):
        grad = gradient_of_J(K, f_vec, u)
        J_val = quadratic_functional(K, f_vec, u)
        Js.append(J_val)

        if np.linalg.norm(grad, ord=2) < tol:
            # print(f"Градиентный спуск сошёлся за {k+1} итераций")
            break

        u -= alpha * grad
    return u, np.array(Js)

def main():
    N = 100  # число шагов по x
    x_int, K, f_vec, u_ex_int = build_1d_problem(N)

    # 1) прямое решение дискретной задачи (эквивалент уравнению Эйлера)
    u_direct = solve_direct(K, f_vec)

    # 2) градиентный спуск по функционалу
    u0 = np.zeros_like(u_direct)  # начальное приближение
    u_gd, J_hist = gradient_descent(K, f_vec, u0, alpha=0.01, max_iter=10000)

    # восстановим решение на всей сетке для визуализации
    x = np.linspace(0, 1, N+1)
    u_exact_full = x * (1 - x)
    u_direct_full = np.zeros_like(x)
    u_direct_full[1:-1] = u_direct

    u_gd_full = np.zeros_like(x)
    u_gd_full[1:-1] = u_gd

    # оценки ошибок
    err_direct = np.max(np.abs(u_direct_full - u_exact_full))
    err_gd = np.max(np.abs(u_gd_full - u_exact_full))

    print(f"Макс. ошибка (прямое решение Ku=f): {err_direct:.4e}")
    print(f"Макс. ошибка (градиентный спуск):   {err_gd:.4e}")

    # Графики
    plt.figure(figsize=(10, 5))
    plt.plot(x, u_exact_full, "k--", linewidth=2, label="Точное u(x)")
    plt.plot(x, u_direct_full, "b-", linewidth=2, label="Дискретное решение (Ku=f)")
    plt.plot(x, u_gd_full, "r:", linewidth=2, label="Решение градиентным спуском")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title("Лаб.13: Метод аппроксимации квадратичного функционала")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Сходимость J(u^k) в градиентном спуске
    plt.figure(figsize=(6, 4))
    plt.semilogy(J_hist, "g-")
    plt.xlabel("Номер итерации")
    plt.ylabel("J(u^k)")
    plt.title("Лаб.13: Сходимость квадратичного функционала J(u)")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
