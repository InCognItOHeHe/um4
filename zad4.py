import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return np.abs(x) + x**2


def df(x):
    if x > 0:
        return 1 + 2 * x
    elif x < 0:
        return -1 + 2 * x
    else:
        return 1


def gradient_descent(start_x, learning_rate, iterations):
    x = start_x
    x_history = [x]
    f_history = [f(x)]

    for i in range(iterations):
        gradient = df(x)
        x = x - learning_rate * gradient
        x_history.append(x)
        f_history.append(f(x))

    return x, f(x), x_history, f_history


# Parametry
start_x = 2.0
learning_rate = 0.1
iterations = 20

# Uruchomienie algorytmu
optimal_x, optimal_value, x_history, f_history = gradient_descent(
    start_x, learning_rate, iterations
)

print(f"Punkt minimum: x = {optimal_x:.6f}")
print(f"Wartość minimum: f(x) = {optimal_value:.6f}")

# Wizualizacja
x_range = np.linspace(-2, 2, 1000)
y_range = [f(x) for x in x_range]

plt.figure(figsize=(12, 6))

# Wykres funkcji
plt.subplot(1, 2, 1)
plt.plot(x_range, y_range)
plt.scatter(x_history, f_history, c="red", s=50)
plt.title("Optymalizacja funkcji f(x) = |x| + x²")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)

# Wykres procesu optymalizacji
plt.subplot(1, 2, 2)
plt.plot(range(iterations + 1), f_history, marker="o")
plt.title("Wartość funkcji w kolejnych iteracjach")
plt.xlabel("Iteracja")
plt.ylabel("f(x)")
plt.grid(True)

plt.tight_layout()
plt.show()
