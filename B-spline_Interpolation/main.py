import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

colors = ['#08ffc8', '#e42c64', '#7A57D1']

sns.set()
sns.set_palette(colors)
sns.set_style('ticks')

data = open('data', 'r')
a, b, n = (int(el) for el in data.readlines())
h = float((b - a) / n)


def f(x):
    return (3 * x - 5) ** 3


def df(x):
    return 9 * (3 * x - 5) ** 2


def B_1(x):
    return (1 - np.abs(x)) if np.abs(x) <= 1 else 0


def B_3(x):
    if np.abs(x) <= 1:
        return (1 / 6) * ((2 - np.abs(x)) ** 3 - 4 * (1 - np.abs(x)) ** 3)
    elif 1 <= np.abs(x) <= 2:
        return (1 / 6) * ((2 - np.abs(x)) ** 3)
    else:
        return 0


def s_1(x, x_arr, y_arr):
    return np.sum([y_arr[i] * B_1((x - x_arr[i]) / h) for i in range(n + 1)])


def s_3(x, alpha_arr):
    x_arr = [a + i * h for i in range(-1, n + 2)]
    return np.sum([alpha_arr[i]*B_3((x - x_arr[i])/h) for i in range(n + 3)])


x_array = np.linspace(a, b, n + 1)
y_array = np.array([f(x) for x in x_array])
a_1, b_1 = df(a), df(b)

matrix = np.zeros((n + 3, n + 3))

for column in range(len(matrix[0])):
    if column == 0:
        matrix[column, 0] = - 0.5
        matrix[column, 2] = 0.5
    if 0 < column < n + 2:
        matrix[column, column - 1] = 1 / 6
        matrix[column, column] = 2 / 3
        matrix[column, column + 1] = 1 / 6
    if column == n + 2:
        matrix[column, column ] = 0.5
        matrix[column, column - 2] = - 0.5

print(matrix)

vector = [h * a_1]
for i in range(n + 1):
    vector.append(y_array[i])
vector.append(h * b_1)
vector = np.array(vector)

print(vector)

A_inverse = np.linalg.inv(matrix)
alphas = A_inverse @ vector

x_plot_dots = np.linspace(a, b, 1000)
y_plot_dots = np.array([f(x) for x in x_plot_dots])
y_linear_spline_plot_dots = [s_1(x, x_array, y_array) for x in x_plot_dots]
y_cubic_spline_dots = [s_3(x, alphas) for x in x_plot_dots]

plt.plot(x_plot_dots, y_plot_dots, x_plot_dots, y_linear_spline_plot_dots, x_plot_dots, y_cubic_spline_dots)
plt.plot(x_array,y_array,'ro')
plt.legend(("F", "Linear Spline", "Cubic Spline"))
plt.grid(True)
plt.show()
