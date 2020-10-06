import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt


def plot_mse(x, y, filename):
    filter_arr1 = y == 1
    filter_arr2 = y == -1
    x1 = np.transpose(np.transpose(x)[filter_arr1])
    x2 = np.transpose(np.transpose(x)[filter_arr2])
    print(x, np.ones(x.shape[1]))
    x = np.vstack((x, np.ones(x.shape[1])))
    compound = np.matmul(x, np.transpose(x))
    all_but_y = np.matmul(np.linalg.inv(compound), x)
    w = np.matmul(all_but_y, y)
    a, b, c = w[0], w[1], w[2]
    h = [0, -1 * c / a]
    vd = [-1 * c / b, 0]

    plt.plot(h, vd)
    plt.plot(x1[0], x1[1], '.b')
    plt.plot(x2[0], x2[1], '.r')
    return w


def plot_fisher(x, y, filename):
    # x = np.vstack((x, np.ones(x.shape[1])))
    filter_arr1 = y == 1
    filter_arr2 = y == -1
    x1 = np.transpose(np.transpose(x)[filter_arr1])
    x2 = np.transpose(np.transpose(x)[filter_arr2])
    m1 = np.array([x1[0].sum(), x1[1].sum()]) / x1.shape[1]
    m2 = np.array([x2[0].sum(), x2[1].sum()]) / x2.shape[1]
    x1 = np.transpose(x1) - m1
    x2 = np.transpose(x2) - m2
    s = np.matmul(np.transpose(x1), x1) + np.matmul(np.transpose(x2), x2)
    print(s)
    w = np.matmul(np.linalg.inv(s), m1 - m2)
    a, b = w[0], w[1]
    h = [0, b]
    vd = [a, 0]
    print(h, vd)

    plt.plot(h, vd)
    # plt.plot(x1[0], x1[1], '.b')
    # plt.plot(x2[0], x2[1], '.r')
    plt.savefig(filename)

    return w


def self_checker(*args):
    w = plot_mse(*args)
    print(type(w), w)


def self_checker2(*args):
    w = plot_fisher(*args)
    print(type(w), w)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    N = 10
    X1 = np.vstack((np.random.normal(1, .1, N),
                    np.random.normal(1, .1, N)))
    X2 = np.vstack((np.random.normal(10, 0.1, N),
                    np.random.normal(10, .1, N)))
    # Each sample is 2-D
    X = np.hstack((X1, X2))
    # now each row is a feature, each column is a sample
    y = np.hstack((np.ones(N) * -1, 1 * np.ones(N)))
    self_checker(X, y, "test1.png")
    self_checker2(X, y, "test2.png")
