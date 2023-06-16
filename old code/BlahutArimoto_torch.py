# import numpy as np
import torch as tr

CUDA_DEVICE = tr.device('cuda')

def frange(start, stop, step):
     x = start
     while x < stop and abs(stop - x) > 1e-6:
         yield x
         x += step

def hamming_weight(num: int) -> int:
        weight = 0

        while num:
            weight += 1
            num &= num - 1

        return weight

def calculate_capacity(Pyx: tr.Tensor, r: tr.Tensor, q: tr.Tensor,  log_base: float) -> float:
    
    m = Pyx.shape[0]
    c = 0
    for i in range(m):
        if r[i] > 0:
            c += tr.sum(r[i] * Pyx[i, :] *
                        tr.log(q[i, :] / r[i] + 1e-16))
    return c / tr.log(tr.Tensor([log_base]))[0]

def calculate_derivative(Pyx: tr.Tensor, r: tr.Tensor, theta_vec: tr.Tensor,  log_base: float = 2) -> float:

    output_bits = int(tr.log(tr.Tensor([Pyx.shape[1]]))[0] / tr.log(tr.Tensor([log_base]))[0])

    r = r.reshape(Pyx.shape[0], 1)

    denominator = tr.sum(r * Pyx, axis=0)
    log_vec = tr.log(Pyx / denominator) / tr.log(tr.Tensor([log_base]))[0]

    dPyx = tr.zeros(Pyx.shape[0], Pyx.shape[1])

    for row in range(Pyx.shape[0]):

        theta = theta_vec[row]

        for col in range(Pyx.shape[1]):

            n_plus = hamming_weight(col)
            n_minus = output_bits - n_plus

            if(theta != 0 and theta != 1):
                dPyx[row, col] = Pyx[row, col] * (n_plus/theta - n_minus/(1-theta))

    helper = dPyx * log_vec
    print(helper)

    derivative = tr.sum(dPyx * log_vec, axis=1)

    return derivative


def get_bernoulli_pyx(in_size, out_size: int) -> tr.Tensor:

    if type(in_size) is int:
        theta_vec = [i for i in frange(0, float(1 + 1/(in_size - 1)), float(1/(in_size - 1)))]
        theta_vec[-1] = 1
    else:
        theta_vec = in_size

    Pyx = tr.zeros(len(theta_vec), 2**out_size)

    row = 0
    for theta in theta_vec:#frange(0, 1, float(1/in_size)):

        col = 0
        for j in range(2**out_size):
            hw = hamming_weight(j)
            n_plus = hamming_weight(col)
            n_minus = out_size - n_plus
            Pyx[row, col] = ((theta)**(n_plus)) * ((1-theta)**(n_minus))
            col += 1

        row += 1

    return Pyx

def get_bernoulli_pyxy(in_size: int, out_size: int) -> tr.Tensor:

    Pyxy = tr.zeros((in_size*(2**(out_size - 1)), 2**out_size))

    row = 0
    for i in frange(0, 1, float(1/in_size)):
        
        for k in range(2**(out_size - 1)):

            col = 0
            for j in range(2**out_size):
                eq_to_prev = 1 if (j >> 1) == k else 0
                Pyxy[row, col] = ((i)**(1 - (j & 1)) * ((1-i)**(j & 1)) * eq_to_prev)
                col += 1

            row += 1

    return Pyxy

def check_if_odd_deltas(output_size: int,  log_base: float = 2, thresh: float = 1e-100, max_iter: int = 1e3) -> bool:

    # Get Pyx for 3 deltas
    theta_vec = [0, 0.49, 0.51, 1]
    Pyx = get_bernoulli_pyx(theta_vec, output_size)
    if Pyx is None:
        print("Error in Pyx calculation")
        return None

    _, r_delta_in_middle = blahut_arimoto_cuda(Pyx)

    # Move middle delta a bit to the side to see if the value is going up  aor down
    theta_vec[1] -= float(1/100)
    theta_vec[2] += float(1/100)
    Pyx = get_bernoulli_pyx(theta_vec, output_size)
    if Pyx is None:
        print("Error in Pyx calculation")
        return None

    _, r_delta_diverted = blahut_arimoto_cuda(Pyx)

    print(r_delta_in_middle)
    print(r_delta_diverted)

    return True if r_delta_in_middle[1] > r_delta_diverted[1] else False
    

def blahut_arimoto_cuda(Pyx: tr.Tensor,  log_base: float = 2, thresh: float = 1e-100, max_iter: int = 1e3) -> tuple:
    '''
    Maximize the capacity between I(X;Y)
    Pyx: each row represnets probability assinmnet
    log_base: the base of the log when calaculating the capacity
    thresh: the threshold of the update, finish the calculation when gettting to it.
    max_iter: the maximum iterations of the calculation
    '''

    # Input test
    assert tr.abs(Pyx.sum(axis=1).mean() - 1) < 1e-6
    assert Pyx.shape[0] > 1

    # The number of inputs: size of |X|
    m = Pyx.shape[0]

    # The number of outputs: size of |Y|
    n = Pyx.shape[1]

    # Initialize the prior uniformly
    r = tr.ones((1, m)) / m
    c = tr.zeros((int(max_iter), 1))

    Pyx_cuda = Pyx.to(device=CUDA_DEVICE)
    r_cuda = r.to(device=CUDA_DEVICE)

    with tr.cuda.device(CUDA_DEVICE):

        # Compute the r(x) that maximizes the capacity
        iteration = 0
        for _ in range(int(max_iter)):

            q = (r_cuda.T * Pyx_cuda).to(device=CUDA_DEVICE)
            helper = tr.sum(q, axis=0)
            q = tr.Tensor(q / tr.sum(q, axis=0))

            r1 = tr.Tensor((tr.prod(tr.pow(q, Pyx_cuda), axis=1))).to(device=CUDA_DEVICE)
            r1 = tr.Tensor(r1 / tr.sum(r1))

            tolerance = tr.linalg.norm(r1.T - r_cuda)
            r_cuda = r1.reshape(1, m)

            c[iteration] = calculate_capacity(Pyx_cuda, r1, q, log_base)

            if tolerance < thresh:
                break

            iteration += 1

    r = r_cuda.flatten().to('cpu')

    return c[:iteration+1], r

def blahut_arimoto(Pyx: tr.Tensor,  log_base: float = 2, thresh: float = 1e-100, max_iter: int = 1e3) -> tuple:
    '''
    Maximize the capacity between I(X;Y)
    Pyx: each row represnets probability assinmnet
    log_base: the base of the log when calaculating the capacity
    thresh: the threshold of the update, finish the calculation when gettting to it.
    max_iter: the maximum iterations of the calculation
    '''

    # Input test
    assert tr.abs(Pyx.sum(axis=1).mean() - 1) < 1e-6
    assert Pyx.shape[0] > 1

    # The number of inputs: size of |X|
    m = Pyx.shape[0]

    # The number of outputs: size of |Y|
    n = Pyx.shape[1]

    # Initialize the prior uniformly
    r = tr.ones((1, m)) / m
    c = tr.zeros((int(max_iter), 1))

    # Compute the r(x) that maximizes the capacity
    iteration = 0
    for _ in range(int(max_iter)):

        q = r.T * Pyx
        helper = tr.sum(q, axis=0)
        q = tr.Tensor(q / tr.sum(q, axis=0))

        r1 = tr.Tensor((tr.prod(tr.pow(q, Pyx), axis=1)))
        r1 = tr.Tensor(r1 / tr.sum(r1))

        tolerance = tr.linalg.norm(r1.T - r)
        r = r1.reshape(1, m)

        c[iteration] = calculate_capacity(Pyx, r1, q, log_base)

        if tolerance < thresh:
            break

        iteration += 1

    r = r.flatten()

    return c[:iteration+1], r

output_size = 5
input_size = 8

theta = [i for i in frange(0, float(1 + 1/(input_size - 1)), float(1/(input_size - 1)))]
if len(theta) > input_size:
    theta = theta[:-1]
theta[-1] = 1

# print(check_if_odd_deltas(output_size))
# exit()

Pyx = get_bernoulli_pyx(theta, output_size)
if Pyx is None:
    print("Error in Pyx calculation")
    exit()

C, r = blahut_arimoto_cuda(Pyx)#frange(0, 1, float(1/input_size))]

derivative = calculate_derivative(Pyx, r, theta)
print(derivative)

import matplotlib.pyplot as plt
fig = plt.figure()

ax = fig.add_subplot(1, 2, 1)
ax.plot(theta, derivative, '*')
# ax.set_title('Bernoulli: input_size={} output_size={}\nC={}'.format(input_size, output_size, C[-1])) 
ax.set_title(r"w'($\theta$)")
ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r"w'($\theta$)") 
# ax.set_yscale("log", base=10)
# ax.set_ylim([0, 1])
ax.grid()

# ax = fig.add_subplot(1, 2, 1)
# ax.plot(C, '*')
# ax.set_title('Bernoulli: input_size={} output_size={}\nC={}'.format(input_size, output_size, C[-1])) 
# ax.set_xlabel('{} iterations'.format(len(C)))
# ax.set_ylabel('Capacity') 
# ax.set_ylim([0, max(C) * 1.1])
# ax.grid()

bx = fig.add_subplot(1, 2, 2)
bx.plot(theta, r, '*')
bx.set_title(r'w($\theta$)') 
bx.set_xlabel(r'$\theta$')
bx.set_ylabel(r'w($\theta$)') 
bx.set_yscale("log", base=10)
bx.set_xlim([0, 1])
bx.grid()

plt.show()

import time
from pathlib import Path
t = time.localtime()
file_name = time.strftime("%H-%M-%S_%d-%m-%Y.pt", t)
file_obj = {'theta': theta, 'r': r, 'C': C}

path = './outputs/' #r'D:/User/University/MSc/final_project/files/'
tr.save(file_obj, path+file_name)



'''
# # Example

# ## Binary symmetric channel
# The BSC is a binary channel; that is, it can transmit only one of two symbols (usually called 0 and 1). <br>
# The transmission is not perfect, and occasionally the receiver gets the wrong bit.  <br> 
# The capacity of this channel  <br> 
# $C = 1 - H_b(P_e)$

e = 0.002
p1 = [1-e, e]
p2 = [e, 1-e]
# p3 = [1-e**2, e**2]
# p4 = [e**2, 1-e**2]
# p5 = [1-e**3, e**3]
# p6 = [e**3, 1-e**3]
# p7 = [1-e**4, e**4]
# p8 = [e**4, 1-e**4]
# p9 = [1-e**5, e**5]
# p10 = [e**5, 1-e**5]
# Pyx = np.asarray([p1, p2, p3, p4, p5, p6, p7, p8, p9, p10])
Pyx = np.asarray([p1, p2])
C, r = blahut_arimoto(Pyx)
print('Capacity: ', C)
print('The prior: ', r)

# The analytic solution of the capaciy
H_P_e = - e * np.log2(e) - (1-e) * np.log2(1-e)
# H_P_e += - (e**2) * np.log2(e**2) - (1-e**2) * np.log2(1-e**2)
# H_P_e += - (e**3) * np.log2(e**3) - (1-e**3) * np.log2(1-e**3)
# H_P_e += - (e**4) * np.log2(e**4) - (1-e**4) * np.log2(1-e**4)
# H_P_e += - (e**5) * np.log2(e**5) - (1-e**5) * np.log2(1-e**5)
print('Anatliyic capacity: ', (1 - H_P_e))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(C, '*')
ax.set_title('Capacity per Iterations') 
ax.set_xlabel('{} iterations'.format(len(C))) 
plt.show()


# ## Erasure channel
# A binary erasure channel (or BEC) is a common communications channel.  <br> 
# In this model, a transmitter sends a bit (a zero or a one), and the receiver either receives the bit or it receives a message that the bit was not received ("erased").  <br> 
# The capacity of this channel is  <br> 
# $C = 1 - P_e$.

e = 0.1
p1 = [1-e, e, 0]
p2 = [0, e, 1-e]
Pyx = np.asarray([p1, p2])
C, r = blahut_arimoto(Pyx)
print('Capacity: ', C)
print('The prior: ', r)

# The analytic solution of the capaciy
print('Anatliyic capacity: ', (1 - e))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(C, '*')
ax.set_title('Capacity per Iterations') 
ax.set_xlabel('{} iterations'.format(len(C))) 
plt.show()
'''