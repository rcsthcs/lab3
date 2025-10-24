import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Параметры для Задания 4 ---
q, n = 2, 20

# --- Функция из варианта 11 ---
def y_z4(x):
    """y(x) = tg(x^(1/q))"""
    return np.tan(x**(1/q))

# --- Аппроксимирующий полином ---
def phi(x, c):
    return c[0] + c[1]*x + c[2]*x**2

# --- Выполнение расчетов ---
print("--- Задание № 4: Аппроксимация ---")
print(f"Функция: y(x) = tan(x^(1/{q}))")
print(f"Параметры: q={q}, n={n}\n")

x_nodes = np.linspace(0, 1, n + 1)
y_nodes = y_z4(x_nodes)

# Вычисление моментов
m1 = np.mean(x_nodes); m2 = np.mean(x_nodes**2)
m3 = np.mean(x_nodes**3); m4 = np.mean(x_nodes**4)
K01 = np.mean(y_nodes); K11 = np.mean(x_nodes * y_nodes)
K21 = np.mean(x_nodes**2 * y_nodes)

# Решение СЛУ
A = np.array([[1, m1, m2], [m1, m2, m3], [m2, m3, m4]])
b_vec = np.array([K01, K11, K21])
coeffs = np.linalg.solve(A, b_vec)
c0, c1, c2 = coeffs[0], coeffs[1], coeffs[2]

print(f"Найденные коэффициенты:")
print(f"  c₀ = {c0:.6f}, c₁ = {c1:.6f}, c₂ = {c2:.6f}\n")

y_approx = phi(x_nodes, coeffs)
errors = y_nodes - y_approx
results_df = pd.DataFrame({
    'x_j': x_nodes, 'y_j (точная)': y_nodes,
    'φ(x_j) (аппрокс.)': y_approx, 'Погрешность ε_j': errors
})

print("Таблица с точными и аппроксимированными значениями (первые 10 строк):")
print(results_df.head(10))

max_error = np.max(np.abs(errors))
rms_error = np.sqrt(np.mean(errors**2))
print("\nИтоговые погрешности аппроксимации:")
print(f"  Максимальная погрешность (ε_max): {max_error:.8f}")
print(f"  Среднеквадратичная погрешность (ε_m): {rms_error:.8f}")

# --- Построение графика ---
x_smooth = np.linspace(0, 1, 400)
y_smooth_approx = phi(x_smooth, coeffs)

plt.figure(figsize=(10, 6))
plt.plot(x_nodes, y_nodes, 'o', label='Исходные данные y_j')
plt.plot(x_smooth, y_smooth_approx, 'r-', label='Аппроксимирующий многочлен φ(x)')
plt.title('Аппроксимация функции методом наименьших квадратов (Задание 4)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()