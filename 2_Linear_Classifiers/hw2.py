import numpy as np
import matplotlib.pyplot as plt

import numpy  # import again
import matplotlib.pyplot  # import again

import numpy.linalg
import numpy.random


def generate_data(Para1, Para2, seed=0):
    """Generate binary random data

    Para1, Para2: dict, {str:float} for each class, 
      keys are mx (center on x axis), my (center on y axis), 
               ux (sigma on x axis), ux (sigma on y axis), 
               y (label for this class)
    seed: int, seed for NUMPy's random number generator. Not Python's random.

    """
    numpy.random.seed(seed)
    X1 = numpy.vstack((numpy.random.normal(Para1['mx'], Para1['ux'], Para1['N']),
                       numpy.random.normal(Para1['my'], Para1['uy'], Para1['N'])))
    X2 = numpy.vstack((numpy.random.normal(Para2['mx'], Para2['ux'], Para2['N']),
                       numpy.random.normal(Para2['my'], Para2['uy'], Para2['N'])))
    Y = numpy.hstack((Para1['y'] * numpy.ones(Para1['N']),
                      Para2['y'] * numpy.ones(Para2['N'])))
    X = numpy.hstack((X1, X2))
    X = numpy.transpose(X)
    return X, Y


def plot_mse(X, y, filename):
    """
    X: 2-D numpy array, each row is a sample, not augmented 
    y: 1-D numpy array

    Examples
    -----------------
    >>> X,y = generate_data(\
        {'mx':1,'my':2, 'ux':0.1, 'uy':1, 'y':1, 'N':20}, \
        {'mx':2,'my':4, 'ux':.1, 'uy':1, 'y':-1, 'N':50},\
        seed=10)
    >>> plot_mse(X, y, 'test1.png')
    array([-1.8650779 , -0.03934209,  2.91707992])
    >>> X,y = generate_data(\
    {'mx':1,'my':-2, 'ux':0.1, 'uy':1, 'y':1, 'N':20}, \
    {'mx':-1,'my':4, 'ux':.1, 'uy':1, 'y':-1, 'N':50},\
    seed=10)
    >>> # print (X, y)
    >>> plot_mse(X, y, 'test2.png')
    array([ 0.93061084, -0.01833983,  0.01127093])
    """
    w = np.array([0, 0, 0])  # just a placeholder

    filter_arr1 = y == 1
    filter_arr2 = y == -1
    x1 = np.transpose(X[filter_arr1])
    x2 = np.transpose(X[filter_arr2])
    # print(X, np.ones(X.shape[1]))
    x = np.vstack((np.transpose(X), np.ones(X.shape[0])))
    compound = np.matmul(x, np.transpose(x))
    all_but_y = np.matmul(np.linalg.inv(compound), x)
    w = np.matmul(all_but_y, y)

    h = [np.min(X[:, 0]), np.max(X[:, 0])]
    vd = [(-w[0] * h[0] - w[2]) / w[1], (-w[0] * h[1] - w[2]) / w[1]]

    matplotlib.pyplot.plot(h, vd)
    matplotlib.pyplot.plot(x1[0], x1[1], '.b')
    matplotlib.pyplot.plot(x2[0], x2[1], '.r')
    # limit the range of plot to the dataset only
    matplotlib.pyplot.xlim(numpy.min(X[:, 0]), numpy.max(X[:, 0]))
    matplotlib.pyplot.ylim(numpy.min(X[:, 1]), numpy.max(X[:, 1]))
    matplotlib.pyplot.savefig(filename)
    matplotlib.pyplot.close('all')  # it is important to always clear the plot
    return w


def plot_fisher(X, y, filename):
    """
    X: 2-D numpy array, each row is a sample, not augmented 
    y: 1-D numpy array

    Examples
    -----------------
    >>> X,y = generate_data(\
        {'mx':1,'my':2, 'ux':0.1, 'uy':1, 'y':1, 'N':20}, \
        {'mx':2,'my':4, 'ux':.1, 'uy':1, 'y':-1, 'N':50},\
        seed=10)
    >>> plot_fisher(X, y, 'test3.png')
    array([-1.61707972, -0.0341108 ,  2.54419773])
    >>> X,y = generate_data(\
        {'mx':-1.5,'my':2, 'ux':0.1, 'uy':2, 'y':1, 'N':200}, \
        {'mx':2,'my':-4, 'ux':.1, 'uy':1, 'y':-1, 'N':50},\
        seed=1)
    >>> plot_fisher(X, y, 'test4.png')
    array([-1.54593468,  0.00366625,  0.40890079])
    """

    filter_arr1 = y == 1
    filter_arr2 = y == -1
    x1 = np.transpose(X[filter_arr1])
    x2 = np.transpose(X[filter_arr2])
    m1 = np.array([x1[0].sum(), x1[1].sum()]) / x1.shape[1]
    m2 = np.array([x2[0].sum(), x2[1].sum()]) / x2.shape[1]
    x1r = np.transpose(x1) - m1
    x2r = np.transpose(x2) - m2
    s = np.matmul(np.transpose(x1r), x1r) + np.matmul(np.transpose(x2r), x2r)
    w = np.matmul(np.linalg.inv(s), m1 - m2)
    wb = -np.matmul(w, (m1 + m2) / 2)

    h = [np.min(X[:, 0]), np.max(X[:, 0])]
    vd = [(-w[0]*h[0]-wb)/w[1], (-w[0]*h[1]-wb)/w[1]]

    w = np.append(w, wb)

    plt.plot(h, vd)
    plt.plot(x1[0], x1[1], '.b')
    plt.plot(x2[0], x2[1], '.r')
    # limit the range of plot to the dataset only
    matplotlib.pyplot.xlim(numpy.min(X[:, 0]), numpy.max(X[:, 0]))
    matplotlib.pyplot.ylim(numpy.min(X[:, 1]), numpy.max(X[:, 1]))
    matplotlib.pyplot.savefig(filename)
    matplotlib.pyplot.close('all')  # it is important to always clear the plot
    return w


if __name__ == "__main__":
    import doctest

    doctest.testmod()
