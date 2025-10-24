import numpy as np
import time
import matplotlib.pyplot as plt
# ============================================================================
# ПАРАМЕТРЫ ЗАДАЧИ
# ============================================================================
n = 15  # размерность системы (n = 5 ÷ 20)
p = 2  # параметр p (p = 1 ÷ 4)
q = 3  # параметр q (q = 1 ÷ 4)
r = 0.5  # параметр r (r = 0.1 ÷ 1.2)
t = 0.8  # параметр t (t = 0.1 ÷ 1.2)

print("=" * 80)
print("РЕШЕНИЕ СИСТЕМЫ ЛИНЕЙНЫХ АЛГЕБРАИЧЕСКИХ УРАВНЕНИЙ")
print("=" * 80)
print(f"\nПараметры задачи:")
print(f"  n = {n}")
print(f"  p = {p}, q = {q}")
print(f"  r = {r}, t = {t}\n")

# ============================================================================
# ФОРМИРОВАНИЕ МАТРИЦЫ A
# ============================================================================
A = np.zeros((n, n))

# Диагональные элементы: a_ii = 10*i^(p/2)
for i in range(n):
    i_actual = i + 1  # индексы начинаются с 1
    A[i, i] = 10 * (i_actual ** (p / 2))

# Внедиагональные элементы: a_ij = ±10^(-3)*(i/j)^(1/q) при i ≠ j
for i in range(n):
    for j in range(n):
        if i != j:
            i_actual = i + 1
            j_actual = j + 1
            # Чередование знаков
            sign = 1 if (i + j) % 2 == 0 else -1
            A[i, j] = sign * (10 ** (-3)) * ((i_actual / j_actual) ** (1 / q))

print("Матрица A (размер {}×{})".format(n, n))
print("Первые 5×5 элементов:")
print(A[:5, :5])
print()

# ============================================================================
# ФОРМИРОВАНИЕ ВЕКТОРА ПРАВОЙ ЧАСТИ b
# ============================================================================
b = np.zeros(n)
for i in range(n):
    i_actual = i + 1
    b[i] = 9 * (i_actual ** (p / 2))

print("Вектор правой части b:")
print("Первые 10 элементов:", b[:min(10, n)])
print()

# ============================================================================
# РЕШЕНИЕ СИСТЕМЫ
# ============================================================================
print("-" * 80)
print("РЕШЕНИЕ МЕТОДОМ LU-РАЗЛОЖЕНИЯ")
print("-" * 80)

start = time.time()
x = np.linalg.solve(A, b)
solve_time = time.time() - start

print(f"Время решения: {solve_time:.6f} сек\n")
print("Решение x:")
for i, xi in enumerate(x):
    print(f"x[{i + 1:2d}] = {xi:.10f}")
print()

# ============================================================================
# ПРОВЕРКА РЕШЕНИЯ
# ============================================================================
print("-" * 80)
print("ПРОВЕРКА РЕШЕНИЯ")
print("-" * 80)

b_check = A @ x
residual = np.linalg.norm(b - b_check)
relative_error = residual / np.linalg.norm(b)

print(f"Невязка ||Ax - b|| = {residual:.6e}")
print(f"Относительная погрешность = {relative_error:.6e}")
print()

# Проверка поэлементно (первые 10 элементов)
print("Проверка Ax = b (первые 10 элементов):")
print(f"{'i':<4} {'b[i]':<15} {'(Ax)[i]':<15} {'Разность':<15}")
print("-" * 50)
for i in range(min(10, n)):
    diff = abs(b[i] - b_check[i])
    print(f"{i + 1:<4} {b[i]:<15.6f} {b_check[i]:<15.6f} {diff:<15.2e}")
print()

# ============================================================================
# АНАЛИЗ СВОЙСТВ МАТРИЦЫ
# ============================================================================
print("-" * 80)
print("АНАЛИЗ СВОЙСТВ МАТРИЦЫ")
print("-" * 80)

det_A = np.linalg.det(A)
cond_A = np.linalg.cond(A)
norm_A = np.linalg.norm(A)
eigenvalues = np.linalg.eigvals(A)

print(f"Определитель det(A) = {det_A:.6e}")
print(f"Число обусловленности cond(A) = {cond_A:.6e}")
print(f"Норма Фробениуса ||A|| = {norm_A:.6e}")
print(f"Ранг матрицы = {np.linalg.matrix_rank(A)}")
print(f"\nСобственные значения (первые 5):")
for i, ev in enumerate(eigenvalues[:5]):
    print(f"  λ[{i + 1}] = {ev:.6f}")
print(f"\nМинимальное |λ| = {np.min(np.abs(eigenvalues)):.6e}")
print(f"Максимальное |λ| = {np.max(np.abs(eigenvalues)):.6e}")
print()

# ============================================================================
# РЕШЕНИЕ ДЛЯ РАЗЛИЧНЫХ РАЗМЕРНОСТЕЙ
# ============================================================================
print("=" * 80)
print("РЕШЕНИЕ ДЛЯ РАЗЛИЧНЫХ РАЗМЕРНОСТЕЙ")
print("=" * 80)
print()

n_values = [5, 10, 15, 20]
print(f"{'n':<5} {'Время (с)':<12} {'Невязка':<15} {'Число обусл.':<15} {'||x||':<12}")
print("-" * 70)

for n_test in n_values:
    # Формируем матрицу
    A_test = np.zeros((n_test, n_test))
    for i in range(n_test):
        i_actual = i + 1
        A_test[i, i] = 10 * (i_actual ** (p / 2))

    for i in range(n_test):
        for j in range(n_test):
            if i != j:
                i_actual = i + 1
                j_actual = j + 1
                sign = 1 if (i + j) % 2 == 0 else -1
                A_test[i, j] = sign * (10 ** (-3)) * ((i_actual / j_actual) ** (1 / q))

    # Формируем вектор b
    b_test = np.zeros(n_test)
    for i in range(n_test):
        i_actual = i + 1
        b_test[i] = 9 * (i_actual ** (p / 2))

    # Решаем
    start = time.time()
    x_test = np.linalg.solve(A_test, b_test)
    time_solve = time.time() - start

    # Анализ
    residual_test = np.linalg.norm(A_test @ x_test - b_test)
    cond_test = np.linalg.cond(A_test)
    norm_x = np.linalg.norm(x_test)

    print(f"{n_test:<5} {time_solve:<12.6f} {residual_test:<15.2e} {cond_test:<15.2e} {norm_x:<12.6f}")

print()
print("=" * 80)
print("РЕШЕНИЕ ЗАВЕРШЕНО")
print("=" * 80)
plt.figure(figsize=(8, 5))
plt.plot(np.arange(1, n + 1), x, marker='o', linestyle='-', color='blue', label='x[i]')
plt.xlabel('Номер компоненты (i)')
plt.ylabel('Значение x[i]')
plt.title('Распределение компонентов решения x')
plt.grid(True)
plt.legend()
plt.show()
