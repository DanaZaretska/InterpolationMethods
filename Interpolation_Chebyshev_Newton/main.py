import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
colors = ['#08ffc8', '#e42c64', '#7A57D1']

sns.set()
sns.set_palette(colors)
sns.set_style('darkgrid')

def f_x(x):
    return 1 / (1 + 25 * x* x)


def g_x(x):
    return np.log(x + 2)


def h_x(x):
    return 3*x**2 + x + 4


def lagrange_polynomial(x_arr_, y_arr_, x_):
    p = 0
    for j in range(len(y_arr_)):
        upper, lower = 1, 1
        for i in range(len(x_arr_)):
            if i == j:
                upper, lower = upper * 1, lower * 1
            else:
                upper = upper * (x_ - x_arr_[i])
                lower = lower * (x_arr_[j] - x_arr_[i])

        l = upper / lower
        p = p + y_arr_[j] * l
    return p


def divided_differences(x, y, n_ ):

    coefficient = np.zeros([n_, n_])
    coefficient[:, 0] = y

    for k in range(1, n_):
        for j in range(n_ - k):
            coefficient[j][k] = \
                (coefficient[j + 1][k - 1] - coefficient[j][k - 1]) / (x[j + k] - x[j])

    return coefficient


def newton_polynomial(coefficient, x_arr_, x):

    n = len(x_arr_) - 1
    p = coefficient[n]
    for k in range(1, n + 1):
        p = coefficient[n - k] + (x - x_arr_[n - k]) * p
    return p


a, b, n, x_dot = open('data').readlines()
a, b, n, x_dot = float(a), float(b), int(n), float(x_dot)
m = n + 1
h = (b - a) / n
x_arr_dot = [x_dot]

x_arr = np.linspace(a, b, 1000)
y_arr_for_f_x = list(map(f_x, x_arr))
y_arr_for_g_x = list(map(g_x, x_arr))
y_arr_for_h_x = list(map(h_x, x_arr))

x_arr_uniform = np.linspace(a, b, n + 1)
y_arr_uniform_for_f_x = list(map(f_x, x_arr_uniform))
y_arr_uniform_for_g_x = list(map(g_x, x_arr_uniform))
y_arr_uniform_for_h_x = list(map(h_x, x_arr_uniform))

x_arr_Chebyshev = [ ((a + b) / 2) + (((b - a) / 2) * np.cos(((2 * k + 1) * np.pi) / (2 * (n + 1)))) for k in range(n + 1)]
y_arr_Chebyshev_for_f_x = list(map(f_x, x_arr_Chebyshev))
y_arr_Chebyshev_for_g_x = list(map(g_x, x_arr_Chebyshev))
y_arr_Chebyshev_for_h_x = list(map(h_x, x_arr_Chebyshev))

y_lagrange_uniform_arr_for_f_x = [lagrange_polynomial(x_arr_uniform, y_arr_uniform_for_f_x, x) for x in x_arr]
y_lagrange_uniform_arr_for_g_x = [lagrange_polynomial(x_arr_uniform, y_arr_uniform_for_g_x, x) for x in x_arr]
y_lagrange_uniform_arr_for_h_x = [lagrange_polynomial(x_arr_uniform, y_arr_uniform_for_h_x, x) for x in x_arr]

y_lagrange_Chebyshev_arr_for_f_x = [lagrange_polynomial(x_arr_Chebyshev, y_arr_Chebyshev_for_f_x, x) for x in x_arr]
y_lagrange_Chebyshev_arr_for_g_x = [lagrange_polynomial(x_arr_Chebyshev, y_arr_Chebyshev_for_g_x, x) for x in x_arr]
y_lagrange_Chebyshev_arr_for_h_x = [lagrange_polynomial(x_arr_Chebyshev, y_arr_Chebyshev_for_h_x, x) for x in x_arr]

differences_uniform_for_f_x = divided_differences(x_arr_uniform, y_arr_uniform_for_f_x, n + 1)[0, :]
differences_Chebyshev_for_f_x = divided_differences(x_arr_Chebyshev, y_arr_Chebyshev_for_f_x, n + 1)[0, :]
differences_uniform_for_g_x = divided_differences(x_arr_uniform, y_arr_uniform_for_g_x, n + 1)[0, :]
differences_Chebyshev_for_g_x = divided_differences(x_arr_Chebyshev, y_arr_Chebyshev_for_g_x, n + 1)[0, :]
differences_uniform_for_h_x = divided_differences(x_arr_uniform, y_arr_uniform_for_h_x, n + 1)[0, :]
differences_Chebyshev_for_h_x = divided_differences(x_arr_Chebyshev, y_arr_Chebyshev_for_h_x, n + 1)[0, :]


plt.figure(figsize=(26, 16))
plt.subplot(3, 2, 1)
plt.plot(x_arr, y_arr_for_f_x, x_arr, y_lagrange_uniform_arr_for_f_x, x_arr, y_lagrange_Chebyshev_arr_for_f_x)
plt.legend(("F(x)", "Lagrange Uniform", "Lagrange Chebyshev"))
plt.grid(True)

plt.subplot(3, 2, 3)
plt.plot(x_arr, y_arr_for_g_x, x_arr, y_lagrange_uniform_arr_for_g_x, x_arr, y_lagrange_Chebyshev_arr_for_g_x)
plt.legend(("G(x)", "Lagrange Uniform", "Lagrange Chebyshev"))
plt.grid(True)

plt.subplot(3, 2, 5)
plt.plot(x_arr, y_arr_for_h_x, x_arr, y_lagrange_uniform_arr_for_h_x, x_arr, y_lagrange_Chebyshev_arr_for_h_x)
plt.legend(("H(x)", "Lagrange Uniform", "Lagrange Chebyshev"))
plt.grid(True)

plt.subplot(3, 2, 2)
plt.plot(x_arr, y_arr_for_f_x, x_arr, newton_polynomial(differences_uniform_for_f_x, x_arr_uniform, x_arr), x_arr, newton_polynomial(differences_Chebyshev_for_f_x, x_arr_Chebyshev, x_arr))
plt.legend(("F(x)", "Newton Uniform", "Newton Chebyshev"))
plt.grid(True)

plt.subplot(3, 2, 4)
plt.plot(x_arr, y_arr_for_g_x, x_arr, newton_polynomial(differences_uniform_for_g_x, x_arr_uniform, x_arr), x_arr, newton_polynomial(differences_Chebyshev_for_g_x, x_arr_Chebyshev, x_arr))
plt.legend(("G(x)", "Newton Uniform", "Newton Chebyshev"))
plt.grid(True)

plt.subplot(3, 2, 6)
plt.plot(x_arr, y_arr_for_h_x, x_arr, newton_polynomial(differences_uniform_for_h_x, x_arr_uniform, x_arr), x_arr, newton_polynomial(differences_Chebyshev_for_h_x, x_arr_Chebyshev, x_arr))
plt.legend(("H(x)", "Newton Uniform", "Newton Chebyshev"))
plt.grid(True)

plt.show()


print([np.abs(f - p) for f, p in zip(y_arr_uniform_for_f_x, [lagrange_polynomial(x_arr_uniform, y_arr_uniform_for_f_x, x) for x in x_arr_uniform])])
print([np.abs(f - p) for f, p in zip(y_arr_uniform_for_g_x, [lagrange_polynomial(x_arr_uniform, y_arr_uniform_for_g_x, x) for x in x_arr_uniform])])
print([np.abs(f - p) for f, p in zip(y_arr_uniform_for_h_x, [lagrange_polynomial(x_arr_uniform, y_arr_uniform_for_h_x, x) for x in x_arr_uniform])])

print([np.abs(f - p) for f, p in zip(y_arr_Chebyshev_for_f_x, [lagrange_polynomial(x_arr_Chebyshev, y_arr_Chebyshev_for_f_x, x) for x in x_arr_Chebyshev])])
print([np.abs(f - p) for f, p in zip(y_arr_Chebyshev_for_g_x, [lagrange_polynomial(x_arr_Chebyshev, y_arr_Chebyshev_for_g_x, x) for x in x_arr_Chebyshev])])
print([np.abs(f - p) for f, p in zip(y_arr_Chebyshev_for_h_x, [lagrange_polynomial(x_arr_Chebyshev, y_arr_Chebyshev_for_h_x, x) for x in x_arr_Chebyshev])])

print([np.abs(f - p) for f, p in zip(y_arr_uniform_for_f_x, newton_polynomial(differences_uniform_for_f_x, x_arr_uniform, x_arr_uniform))])
print([np.abs(f - p) for f, p in zip(y_arr_uniform_for_g_x, newton_polynomial(differences_uniform_for_g_x, x_arr_uniform, x_arr_uniform))])
print([np.abs(f - p) for f, p in zip(y_arr_uniform_for_h_x, newton_polynomial(differences_uniform_for_h_x, x_arr_uniform, x_arr_uniform))])

print([np.abs(f - p) for f, p in zip(y_arr_Chebyshev_for_f_x, newton_polynomial(differences_Chebyshev_for_f_x, x_arr_Chebyshev, x_arr_Chebyshev))])
print([np.abs(f - p) for f, p in zip(y_arr_Chebyshev_for_g_x, newton_polynomial(differences_Chebyshev_for_g_x, x_arr_Chebyshev, x_arr_Chebyshev))])
print([np.abs(f - p) for f, p in zip(y_arr_Chebyshev_for_h_x, newton_polynomial(differences_Chebyshev_for_h_x, x_arr_Chebyshev, x_arr_Chebyshev))])



def mistake_in_dot(x_dot):
    print('____________________________________________________')
    print('Lagrange uniform f_x')
    print(np.abs(f_x(x_dot) - lagrange_polynomial(x_arr_uniform, y_arr_uniform_for_f_x, x_dot)))
    print('Lagrange Chebyshev f_x')
    print(np.abs(f_x(x_dot) - lagrange_polynomial(x_arr_Chebyshev, y_arr_Chebyshev_for_f_x, x_dot)))

    print('Lagrange uniform g_x')
    print(np.abs(g_x(x_dot) - lagrange_polynomial(x_arr_uniform, y_arr_uniform_for_g_x, x_dot)))
    print('Lagrange Chebyshev g_x')
    print(np.abs(g_x(x_dot) - lagrange_polynomial(x_arr_Chebyshev, y_arr_Chebyshev_for_g_x, x_dot)))

    print('Lagrange uniform h_x')
    print(np.abs(h_x(x_dot) - lagrange_polynomial(x_arr_uniform, y_arr_uniform_for_h_x, x_dot)))
    print('Lagrange Chebyshev h_x')
    print(np.abs(h_x(x_dot) - lagrange_polynomial(x_arr_Chebyshev, y_arr_Chebyshev_for_h_x, x_dot)))

    print('Newton uniform f_x')
    print(np.abs(f_x(x_dot) - newton_polynomial(differences_uniform_for_f_x, x_arr_uniform, x_dot)))
    print('Newton Chebyshev f_x')
    print(np.abs(f_x(x_dot) - newton_polynomial(differences_Chebyshev_for_f_x, x_arr_Chebyshev, x_dot)))

    print('Newton uniform g_x')
    print(np.abs(g_x(x_dot) - newton_polynomial(differences_uniform_for_g_x, x_arr_uniform, x_dot)))
    print('Newton Chebyshev g_x')
    print(np.abs(g_x(x_dot) - newton_polynomial(differences_Chebyshev_for_g_x, x_arr_Chebyshev, x_dot)))

    print('Newton uniform h_x')
    print(np.abs(h_x(x_dot) - newton_polynomial(differences_uniform_for_h_x, x_arr_uniform, x_dot)))
    print('Newton Chebyshev h_x')
    print(np.abs(h_x(x_dot) - newton_polynomial(differences_Chebyshev_for_h_x, x_arr_Chebyshev, x_dot)))

mistake_in_dot(x_dot)




# print(y_arr_uniform_for_f_x, '\n',  [lagrange_polynomial(x_arr_uniform, y_arr_uniform_for_f_x, x) for x in x_arr_uniform])
# print(y_arr_uniform_for_g_x, '\n', [lagrange_polynomial(x_arr_uniform, y_arr_uniform_for_g_x, x) for x in x_arr_uniform])
# print(y_arr_uniform_for_f_x, '\n', newton_polynomial(differences_uniform_for_f_x, x_arr_uniform, x_arr_uniform))
# print(y_arr_uniform_for_g_x, '\n', newton_polynomial(differences_uniform_for_g_x, x_arr_uniform, x_arr_uniform))