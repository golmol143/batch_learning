
# import numpy as np
import numpy as np
import matplotlib.pyplot as plt

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

def calculate_capacity(Pyx: np.ndarray, r: np.ndarray, q: np.ndarray,  log_base: float) -> float:
    
    m = Pyx.shape[0]
    c = 0
    for i in range(m):
        if r[i] > 0:
            c += np.sum(r[i] * Pyx[i, :] *
                        np.log(q[i, :] / r[i] + 1e-16))
    return c / np.log(log_base)

def get_bernoulli_pyx(in_size: int, out_size: int) -> np.ndarray:

    if in_size % 2 == 1:
        return None

    Pyx = np.zeros((in_size + 1, 2**out_size))

    row = 0
    for i in frange(0, float(1 + 1/in_size), float(1/in_size)):#frange(0, 1, float(1/in_size)):

        i = 1 if i > 1 else i
        col = 0
        for j in range(2**out_size):
            hw = hamming_weight(j)
            Pyx[row, col] = ((i)**(out_size - hw)) * ((1-i)**(hw))
            col += 1

        row += 1

    return Pyx

def get_bernoulli_pyxy(in_size: int, out_size: int) -> np.ndarray:

    Pyxy = np.zeros((in_size*(2**(out_size - 1)), 2**out_size))

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

def blahut_arimoto_improved(Pyx: np.ndarray, Pyxy: np.ndarray, log_base: float = 2, thresh: float = 1e-100, max_iter: int = 1e3) -> tuple:
    '''
    Maximize the capacity between I(X;Y)
    Pyx: each row represnets probability assinmnet
    log_base: the base of the log when calaculating the capacity
    thresh: the threshold of the update, finish the calculation when gettting to it.
    max_iter: the maximum iterations of the calculation
    '''

    # Input test
    assert np.abs(Pyx.sum(axis=1).mean() - 1) < 1e-6
    assert Pyx.shape[0] > 1

    # The number of inputs: size of |X|
    m = Pyx.shape[0]

    # The number of outputs: size of |Y|
    n = Pyx.shape[1]

    # Initialize the prior uniformly
    r = np.ones((1, m)) / m
    c = np.zeros((int(max_iter), 1))

    # Compute the r(x) that maximizes the capacity
    iteration = 0
    for _ in range(int(max_iter)):

        q = r.T * Pyx
        helper = (r.repeat(n / 2) / (n / 2)).reshape(1, int(m * (n / 2)))
        q_tag = helper.T * Pyxy
        hleper = np.sum(q_tag, axis=0)
        q = np.array(q / np.sum(q_tag, axis=0))

        r1 = np.array((np.prod(np.power(q, Pyx), axis=1)))
        r1 = np.array(r1 / np.sum(r1))

        tolerance = np.linalg.norm(r1.T - r)
        r = r1.reshape(1, m)

        c[iteration] = calculate_capacity(Pyx, r1, q, log_base)

        if tolerance < thresh:
            break

        iteration += 1

    r = r.flatten()

    return c[:iteration+1], r

def blahut_arimoto(Pyx: np.ndarray,  log_base: float = 2, thresh: float = 1e-100, max_iter: int = 1e3) -> tuple:
    '''
    Maximize the capacity between I(X;Y)
    Pyx: each row represnets probability assinmnet
    log_base: the base of the log when calaculating the capacity
    thresh: the threshold of the update, finish the calculation when gettting to it.
    max_iter: the maximum iterations of the calculation
    '''

    # Input test
    assert np.abs(Pyx.sum(axis=1).mean() - 1) < 1e-6
    assert Pyx.shape[0] > 1

    # The number of inputs: size of |X|
    m = Pyx.shape[0]

    # The number of outputs: size of |Y|
    n = Pyx.shape[1]

    # Initialize the prior uniformly
    r = np.ones((1, m)) / m
    c = np.zeros((int(max_iter), 1))

    # Compute the r(x) that maximizes the capacity
    iteration = 0
    for _ in range(int(max_iter)):

        q = r.T * Pyx
        helper = np.sum(q, axis=0)
        q = np.array(q / np.sum(q, axis=0))

        r1 = np.array((np.prod(np.power(q, Pyx), axis=1)))
        r1 = np.array(r1 / np.sum(r1))

        tolerance = np.linalg.norm(r1.T - r)
        r = r1.reshape(1, m)

        c[iteration] = calculate_capacity(Pyx, r1, q, log_base)

        if tolerance < thresh:
            break

        iteration += 1

    r = r.flatten()

    return c[:iteration+1], r

fig = plt.figure()

input_size = 100
output_size = 5

Pyx = get_bernoulli_pyx(input_size, output_size)
if Pyx is None:
    print("Error in Pyx calculation")
    exit()


C, r = blahut_arimoto(Pyx)

print(r)

ax = fig.add_subplot(1, 2, 1)
ax.plot(C, '*')
ax.set_title('Bernoulli: input_size={} output_size={}\nC={}'.format(input_size, output_size, C[-1])) 
ax.set_xlabel('{} iterations'.format(len(C)))
ax.set_ylabel('Capacity') 
# ax.set_yscale("log", base=10)

ax.set_ylim([0, max(C) * 1.1])
ax.grid()

bx = fig.add_subplot(1, 2, 2)
theta = [i for i in frange(0, float(1 + 1/input_size), float(1/input_size))]#frange(0, 1, float(1/input_size))]
bx.plot(theta, r, '*')
bx.set_title(r'w($\theta$)') 
bx.set_xlabel(r'$\theta$')
bx.set_ylabel(r'w($\theta$)')
bx.set_yscale("log", base=10)

bx.set_xlim([0, 1])
# bx.set_ylim([0, 1])
bx.grid()

# Pyxy = get_bernoulli_pyxy(input_size, output_size)
# C_improverd, r_improverd = blahut_arimoto_improved(Pyx, Pyxy)

# bx = fig.add_subplot(1, 2, 2)
# bx.plot(C_improverd, '*')
# bx.set_title('IMPROVED \n Bernoulli with input_size={} output_size={}'.format(input_size, output_size)) 
# bx.set_xlabel('{} iterations'.format(len(C_improverd)))
# bx.set_ylabel('Capacity') 

# bx.set_ylim([0, max(C_improverd) * 1.1])
# bx.grid()

plt.show()


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