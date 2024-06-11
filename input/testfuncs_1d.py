import numpy as np


def f1(x):
    return x * np.sin(x)


def f2(x):
    return np.sin(4*x) + 1.0


def f3(x):
    return x**2+x**3-5.0


def f4(x):
    return np.abs(x)


def f5(x):
    return np.exp(x)


def f6(x):
    return np.sin(5*x) + x
