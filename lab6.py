import numpy as np
from scipy.integrate import quad

# Определяем подынтегральную функцию
def f(x):
    return np.cos(0.5*x + 0.125*x**2)**2

# Пределы интегрирования
a = 0
b = np.pi

# Вычисляем интеграл
I, error = quad(f, a, b)

print(f"Значение интеграла I = {I:.10f}")
print(f"Погрешность вычисления ≈ {error:.2e}")
