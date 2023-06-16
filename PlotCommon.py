import matplotlib.pyplot as plt
import torch as tr

def helper():

    x = [1, 2, 5, 8, 14, 19, 24, 30, 37, 41]
    y = [2, 3, 4, 5,  6,  7,  8,  9, 10, 11]

    from math import ceil
    x2 = range(1, 42)
    test = [ceil(val**(63/100)) for val in x2]
    test2 = [ceil(val**(2/3)) for val in x2]

    fig = plt.figure()

    ax = fig.add_subplot(1, 2, 2)
    ax.step(x, y)
    ax.step(x2, test, color='red')
    ax.grid()

    ax = fig.add_subplot(1, 2, 1)
    ax.step(x, y)
    ax.step(x2, test2, color='red')
    ax.grid()

    plt.show()

def plot_r_dr_C(theta_vector: tr.Tensor, output_size: int, r: tr.Tensor, dr: tr.Tensor, C: tr.Tensor):

    fig = plt.figure()

    fig.suptitle('Bernoulli: input_size={} output_size={}'.format(len(theta_vector), output_size)) 

    bx = fig.add_subplot(1, 3, 1)
    bx.plot(theta_vector, r, '*')
    bx.set_title(r'w($\theta$)') 
    bx.set_xlabel(r'$\theta$')
    bx.set_ylabel(r'w($\theta$)') 
    bx.set_yscale("log", base=10)
    bx.set_xlim([0, 1])
    bx.grid()

    cx = fig.add_subplot(1, 3, 2)
    cx.plot(theta_vector, dr, '*')
    # cx.set_title('Bernoulli: input_size={} output_size={}'.format(len(theta_vector), output_size)) 
    cx.set_title(r"w'($\theta$)")
    cx.set_xlabel(r'$\theta$')
    cx.set_ylabel(r"w'($\theta$)") 
    cx.set_xlim([0, 1])
    cx.grid()

    ax = fig.add_subplot(1, 3, 3)
    ax.plot(C, '*')
    ax.set_title('C={}'.format(C[-1]))
    ax.set_xlabel('{} iterations'.format(len(C)))
    ax.set_ylabel('Capacity') 
    ax.set_ylim([0, max(C) * 1.1])
    ax.grid()

    plt.show()


def plot_r_and_dr(theta_vector: tr.Tensor, output_size: int, r: tr.Tensor, dr: tr.Tensor):

    fig = plt.figure()

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(theta_vector, dr, '*')
    ax.set_title('Bernoulli: input_size={} output_size={}'.format(len(theta_vector), output_size)) 
    ax.set_title(r"w'($\theta$)")
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r"w'($\theta$)") 
    ax.set_yscale("log", base=10)
    ax.set_xlim([0, 1])
    ax.grid()

    bx = fig.add_subplot(1, 2, 1)
    bx.plot(theta_vector, r, '*')
    bx.set_title(r'w($\theta$)') 
    bx.set_xlabel(r'$\theta$')
    bx.set_ylabel(r'w($\theta$)') 
    # bx.set_yscale("log", base=10)
    bx.set_xlim([0, 1])
    bx.grid()

    plt.show()

def plot_r_and_C(theta_vector: tr.Tensor, output_size: int, r: tr.Tensor, C: tr.Tensor):

    fig = plt.figure()

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(C, '*')
    ax.set_title('Bernoulli: input_size={} output_size={}\nC={}'.format(len(theta_vector), output_size, C[-1])) 
    ax.set_xlabel('{} iterations'.format(len(C)))
    ax.set_ylabel('Capacity') 
    ax.set_ylim([0, max(C) * 1.1])
    ax.grid()

    bx = fig.add_subplot(1, 2, 1)
    bx.plot(theta_vector, r, '*')
    bx.set_title(r'w($\theta$)') 
    bx.set_xlabel(r'$\theta$')
    bx.set_ylabel(r'w($\theta$)') 
    # bx.set_yscale("log", base=10)
    bx.set_xlim([0, 1])
    bx.grid()

    plt.show()

def plot_r(theta_vector: tr.Tensor, output_size: int, r: tr.Tensor):

    fig = plt.figure()

    fig.suptitle('Bernoulli: input_size={} output_size={}'.format(len(theta_vector), output_size)) 

    bx = fig.add_subplot(1, 1, 1)
    bx.plot(theta_vector, r, '*')
    bx.set_title(r'w($\theta$)') 
    bx.set_xlabel(r'$\theta$')
    bx.set_ylabel(r'w($\theta$)') 
    bx.set_yscale("log", base=10)
    bx.set_xlim([0, 1])
    bx.grid()

    plt.show()

def plot_dr(theta_vector: tr.Tensor, output_size: int, dr: tr.Tensor):

    fig = plt.figure()

    fig.suptitle('Bernoulli: input_size={} output_size={}'.format(len(theta_vector), output_size)) 

    cx = fig.add_subplot(1, 1, 1)
    cx.plot(theta_vector, dr, '*')
    # cx.set_title('Bernoulli: input_size={} output_size={}'.format(len(theta_vector), output_size)) 
    cx.set_title(r"w'($\theta$)")
    cx.set_xlabel(r'$\theta$')
    cx.set_ylabel(r"w'($\theta$)") 
    cx.set_xlim([0, 1])
    cx.grid()

    plt.show()

def plot_C(theta_vector: tr.Tensor, output_size: int, C: tr.Tensor):

    fig = plt.figure()

    fig.suptitle('Bernoulli: input_size={} output_size={}'.format(len(theta_vector), output_size)) 

    ax = fig.add_subplot(1, 1, 1)
    ax.plot(C, '*')
    ax.set_title('C={}'.format(C[-1]))
    ax.set_xlabel('{} iterations'.format(len(C)))
    ax.set_ylabel('Capacity') 
    ax.set_ylim([0, max(C) * 1.1])
    ax.grid()

    plt.show()