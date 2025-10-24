import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Параметры для Задания 3 ---
a, b, k, m, n = 1, 2, 3, 4, 20

# --- Функция из варианта 11 ---
def y_z3(x):
    """y(x) = (a + b*x^(1/m))^(-1/k)"""
    return (a + b * x**(1/m))**(-1/k)

# --- Выполнение расчетов ---
print("--- Задание № 3: Интерполирование ---")
print(f"Функция: y(x) = (a + b*x^(1/m))^(-1/k)")
print(f"Параметры: a={a}, b={b}, k={k}, m={m}, n={n}\n")

h = 1.0 / n
x_nodes = np.linspace(0, 1, n + 1)
y_nodes = y_z3(x_nodes)
interpolation_results = []

for j in range(n - 1):
    if j + 2 >= len(x_nodes): continue
    x_j, x_j1 = x_nodes[j], x_nodes[j+1]
    y_j, y_j1, y_j2 = y_nodes[j], y_nodes[j+1], y_nodes[j+2]
    x_mid = (j + 0.5) * h
    first_diff = (y_j1 - y_j) / h
    second_diff = (y_j2 - 2*y_j1 + y_j) / (2 * h**2)
    p_x_mid = y_j + (x_mid - x_j) * first_diff + (x_mid - x_j) * (x_mid - x_j1) * second_diff
    y_exact_mid = y_z3(x_mid)
    error = np.abs(y_exact_mid - p_x_mid)
    interpolation_results.append({
        'j': j, 'x_{j+1/2}': x_mid, 'P(x_{j+1/2})': p_x_mid,
        'y(x_{j+1/2})': y_exact_mid, 'Погрешность ε_{j+1/2}': error
    })

errors_df = pd.DataFrame(interpolation_results)
pd.set_option('display.float_format', '{:.6f}'.format)
print("Таблица погрешностей (первые 10 строк):")
print(errors_df.head(10))

max_error = errors_df['Погрешность ε_{j+1/2}'].max()
rms_error = np.sqrt((errors_df['Погрешность ε_{j+1/2}']**2).mean())
print("\nИтоговые погрешности:")
print(f"  Максимальная погрешность (ε_max): {max_error:.8f}")
print(f"  Среднеквадратичная погрешность (ε_m): {rms_error:.8f}")

# --- Построение графика ---
x_smooth = np.linspace(0, 1, 400)
y_smooth = y_z3(x_smooth)

plt.figure(figsize=(10, 6))
plt.plot(x_smooth, y_smooth, label='Точная функция y(x)')
plt.plot(x_nodes, y_nodes, 'o', label='Узлы интерполяции (x_j, y_j)')
plt.plot(errors_df['x_{j+1/2}'], errors_df['P(x_{j+1/2})'], 'rx', label='Интерполированные значения P(x_{j+1/2})')

plt.title('Сравнение точной функции и результатов интерполяции (Задание 3)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()