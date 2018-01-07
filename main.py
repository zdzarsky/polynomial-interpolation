from math import *

import matplotlib.pyplot as plt
import numpy as np


def foo1(x: float) -> float:
    return 2 * (sin(x)) + 3


def foo2(x: float) -> float:
    return (2 * x ** 3 + x ** 2 + 3 * x + 7) / 2000


def foo3(x: float) -> float:
    return log(1 + exp(x))


def foo4(x: float) -> float:
    return sin(exp(x))


def foo5(x: float) -> float:
    a = 1
    t = 1
    if x > 0:
        return a * (-1) ** int(abs(x) / t)
    else:
        return a * (-1) ** int(abs(x - 1) / t)


def add_noise(function_values: np.ndarray, distribution_name) -> np.ndarray:
    elements = function_values.size
    noise = _get_noise(distribution_name, elements)
    m_function_values = function_values.copy()
    for i in range(m_function_values.size):
        m_function_values[i] += noise[i]
    return m_function_values


def _get_noise(name: str, s: int) -> np.ndarray:
    if name == 'normal':
        return np.random.normal(0, 0.20, s)
    elif name == 'poisson':
        return np.random.poisson(0.1, s)
    elif name == 'exp':
        return np.random.exponential(0.1, s)
    else:
        raise ValueError('Bad name of the distribution')


def get_polynomial_fit(x: np.ndarray, y: np.ndarray, degree: int) -> np.poly1d:
    z = np.polyfit(x, y, degree)
    return np.poly1d(z)


def d2n(element, x: tuple) -> str:
    if element == x[0]:
        return 'sparse'
    return 'dense'


def plot_results(x, y, polynomial, chart_name):
    xp = np.linspace(x[1], x[-1], 100)
    plt.plot(x, y, '.', x, y, '-', xp, polynomial(xp), '--')
    plt.savefig('plots/' + chart_name + '.png')
    plt.close()


def make_csv(name, x, y):
    with open('csv/' + name + '.csv', 'w') as f:
        for i in range(x.size):
            f.write(str(x[i]) + ',' + str(y[i]) + '\n')
        f.close()


if __name__ == '__main__':
    functions = [foo1, foo2, foo3, foo4, foo5]
    distributions = ['normal', 'poisson', 'exp']
    density = (150, 300)
    pd = 5
    for function in functions:
        for size in density:
            mx = np.linspace(-10, 10, size)
            my = np.asarray(list(map(function, mx)))
            name_and_size = function.__name__ + '_' + d2n(size, density)
            plot_results(mx, my, get_polynomial_fit(mx, my, pd), name_and_size)
            make_csv(name_and_size, mx, my)
            for d in distributions:
                new_y = add_noise(my, d)
                label = str(name_and_size) + '_' + d
                plot_results(mx, new_y, get_polynomial_fit(mx, new_y, pd),
                             label)
                make_csv(label, mx, new_y)
