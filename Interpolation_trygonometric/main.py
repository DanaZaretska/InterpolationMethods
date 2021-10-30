# odd number of interpolation nodes

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

colors = ['#FF2E63', '#08D9D6']

sns.set()
sns.set_palette(colors)
sns.set_style('darkgrid')

data = open("data")
n = int(data.readline())
x_dot = float(data.readline())


def f(x: float) -> float:
    return np.exp(np.sin(x) + np.cos(x))


def g(x: float) -> float:
    return 3 * np.cos(15 * x)


def find_a_k(k: int, y: list) -> float:
    return (2 / (2 * n + 1)) * np.sum([y[i] * np.cos((2 * np.pi * i * k)/(2 * n + 1)) for i in range(2 * n + 1)])


def find_b_k(k: int, y: list) -> float:
    return (2 / (2 * n + 1)) * np.sum([y[i] * np.sin((2 * np.pi * i * k) / (2 * n + 1)) for i in range(2 * n + 1)])


def T_n(a: list, b: list, x: float) -> float:
    return (a[0] / 2) + np.sum([a[k] * np.cos(k * x) + b[k - 1] * np.sin(k * x) for k in range (1, n + 1)])


def find_y_values_for_f_x(x_list: list) -> list:
    return [f(x) for x in x_list]


def find_y_values_for_g_x(x_list: list) -> list:
    return [g(x) for x in x_list]


x_nodes = [float((2 * np.pi * i) / (2 * n + 1)) for i in range(2 * n + 1)]
f_x_values_in_nodes = find_y_values_for_f_x(x_nodes)
g_x_values_in_nodes = find_y_values_for_g_x(x_nodes)

a_for_f_x = [find_a_k(k, find_y_values_for_f_x(x_nodes)) for k in range(n + 1)]
a_for_g_x = [find_a_k(k, find_y_values_for_g_x(x_nodes)) for k in range(n + 1)]

b_for_f_x = [find_b_k(k, find_y_values_for_f_x(x_nodes)) for k in range(1, n + 1)]
b_for_g_x = [find_b_k(k, find_y_values_for_g_x(x_nodes)) for k in range(1, n + 1)]

T_n_values_for_f_x_in_nodes = [T_n(a_for_f_x, b_for_f_x, x) for x in x_nodes]
T_n_values_for_g_x_in_nodes = [T_n(a_for_g_x, b_for_g_x, x) for x in x_nodes]

error_for_f_x = [np.fabs(f - t) for f, t in zip(f_x_values_in_nodes, T_n_values_for_f_x_in_nodes)]
error_for_g_x = [np.fabs(g - t) for g, t in zip(g_x_values_in_nodes, T_n_values_for_g_x_in_nodes)]

error_in_x_dot_f_x = f(x_dot) - T_n(a_for_f_x, b_for_f_x, x_dot)
error_in_x_dot_g_x = g(x_dot) - T_n(a_for_g_x, b_for_g_x, x_dot)

x_plot_dots = np.linspace(0, 2 * np.pi, 1000)
f_x_plot_dots = [f(x) for x in x_plot_dots]
g_x_plot_dots = [g(x) for x in x_plot_dots]
T_n_f_x_plot_dots = [T_n(a_for_f_x, b_for_f_x, x) for x in x_plot_dots]
T_n_g_x_plot_dots = [T_n(a_for_g_x, b_for_g_x, x) for x in x_plot_dots]

plt.figure(figsize=(24, 10))
plt.subplot(1, 2, 1)
plt.plot(x_plot_dots, f_x_plot_dots, x_plot_dots, T_n_f_x_plot_dots)
plt.legend(("F(x)", "Trigonometric polynomial"))
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(x_plot_dots, g_x_plot_dots, x_plot_dots,  T_n_g_x_plot_dots)
plt.legend(("G(x)", "Trigonometric polynomial"))
plt.grid(True)

plt.show()

print("Error in F_x nodes: ", error_for_f_x)
print("Error in G_x nodes: ", error_for_g_x)
print("Error in x dot for F_x : ", error_in_x_dot_f_x)
print("Error in x dot for G_x : ", error_in_x_dot_g_x)