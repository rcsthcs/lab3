import numpy as np
import matplotlib.pyplot as plt

# --- Параметры для Задания 2 ---
j, k, m, l = 1, 1, 1, 1

# --- Функция из варианта 11 ---
def f_z2(x):
    """f(x) = (1-x)^j * ch^k(x^m) - x^l"""
    return (1 - x)**j * np.cosh(x**m)**k - x**l

# --- Метод дихотомии (половинного деления) ---
def bisection_method(f, a, b, tol=1e-6):
    if f(a) * f(b) >= 0:
        return None, "На концах отрезка функция имеет одинаковый знак."
    iterations = 0
    while (b - a) / 2.0 > tol:
        midpoint = (a + b) / 2.0
        if f(midpoint) == 0:
            return midpoint, iterations
        elif f(a) * f(midpoint) < 0:
            b = midpoint
        else:
            a = midpoint
        iterations += 1
    return (a + b) / 2.0, iterations

# --- Метод секущих ---
def secant_method(f, x0, x1, tol=1e-6, max_iter=100):
    iterations = 0
    while abs(x1 - x0) > tol and iterations < max_iter:
        fx0, fx1 = f(x0), f(x1)
        if fx1 - fx0 == 0:
            return x1, "Деление на ноль в методе секущих."
        x_next = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        x0, x1 = x1, x_next
        iterations += 1
    return x1, iterations

# --- Выполнение расчетов и вывод результатов ---
print("--- Задание № 2: Решение нелинейных уравнений ---")
print(f"Уравнение: (1-x)^{j} * cosh(x^{m})^{k} - x^{l} = 0")
print(f"Параметры: j={j}, k={k}, m={m}, l={l}\n")

# Решение методом дихотомии
a, b = 0.0, 1.0
root_bisection, iters_bisection = bisection_method(f_z2, a, b)
print("Метод дихотомии:")
if root_bisection is not None:
    print(f"  Найденный корень: {root_bisection:.8f}")
    print(f"  Число итераций: {iters_bisection}")
else:
    print(f"  Ошибка: {iters_bisection}")

# Решение методом секущих
epsilon = 1e-6
x0 = 0.2
x1 = x0 + 10 * epsilon
root_secant, iters_secant = secant_method(f_z2, x0, x1, tol=epsilon)
print("\nМетод секущих:")
if isinstance(root_secant, float):
    print(f"  Найденный корень: {root_secant:.8f}")
    print(f"  Число итераций: {iters_secant}")
else:
    print(f"  Ошибка: {iters_secant}")

# --- Построение графика ---
x_vals = np.linspace(0, 1, 400)
y_vals = f_z2(x_vals)

plt.figure(figsize=(8, 5))
plt.plot(x_vals, y_vals, label='f(x) = (1-x)ch(x) - x')
plt.axhline(0, color='gray', linestyle='--')
# Отмечаем корень, найденный более быстрым методом секущих
plt.plot(root_secant, f_z2(root_secant), 'ro', label=f'Корень x ≈ {root_secant:.4f}')

plt.title('График функции и найденный корень (Задание 2)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()