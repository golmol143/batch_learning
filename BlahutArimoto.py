# import numpy as np
from numpy import float64
import torch as tr
import common as cm
from BernoulliCommon import BernoulliObject as Ber
import math

def calculate_capacity(Pyx: tr.Tensor, r: tr.Tensor, q: tr.Tensor,  log_base: float) -> float:
    
    m = Pyx.shape[0]
    c = 0
    for i in range(m):
        if r[i] > 0:
            c += tr.sum(r[i] * Pyx[i, :] *
                        tr.log(q[i, :] / r[i] + 1e-16))
    return c / tr.log(tr.Tensor([log_base]))[0]

def calculate_capacity_conditional(P_theta: tr.Tensor, P_Yn_given_theta: tr.Tensor, P_Yn1_given_theta: tr.Tensor, P_theta_given_Yn: tr.Tensor, P_theta_given_Yn1: tr.Tensor,  log_base: float) -> float:
    
    m = P_Yn1_given_theta.shape[0]
    n = P_Yn1_given_theta.shape[1]
    l = P_Yn1_given_theta.shape[2]
    c = 0
    helper = tr.sum(P_theta_given_Yn1, axis=2)
    helper2 = tr.sum(P_Yn1_given_theta, axis=2)
    for i in range(m):
        for j in range(n):
            for k in range(l):
                # c += P_theta[i] * tr.sum(helper2[i, :] *
                #             tr.log(helper[i, :] / P_theta_given_Yn[i, :] + 1e-16))
                if P_Yn1_given_theta[i,j,k] > 1e-50:
                    c += P_theta[i] * P_Yn1_given_theta[i,j,k] * tr.log(P_theta_given_Yn1[i,j,k] / P_theta_given_Yn[i,j] + 1e-16)
    return c / tr.log(tr.Tensor([log_base]))[0]

def get_number_of_deltas(output_size: int) -> int:

    initial_input_size = math.ceil(output_size ** (63/100))
    BerObj = Ber(initial_input_size, output_size)
    middle_delta = check_if_middle_delta(BerObj)
    odd = True if BerObj.input_size % 2 == 1 else False
    if odd:
        if middle_delta:
            return BerObj.input_size
        else:
            return BerObj.input_size + 1

    if middle_delta:
        return BerObj.input_size + 1
    return BerObj.input_size

def check_if_one_more_delta(BerObj: Ber, C : float, d_threshold: float = 0.02, d_step:float = 0.00001, use_cuda: bool = True) -> bool:

    old_theta_vector = BerObj.theta_vector
    
    step = 1/(2*BerObj.input_size)

    odd = True if BerObj.input_size % 2 == 1 else False
    if odd == False:
        theta_vector_helper = BerObj.theta_vector + [0.5]
    else:
        theta_vector_helper = BerObj.theta_vector + [0.5 - step, 0.5 + step]
        theta_vector_helper.remove(theta_vector_helper[math.floor(BerObj.input_size / 2)])
    theta_vector_helper.sort()

    BerObj.update_input_size(len(theta_vector_helper), theta_vector_helper)

    _ = BerObj.update_Pyx_get_dPyx()
    C_new, r, dr = blahut_arimoto_derivative(BerObj, d_threshold=d_threshold, d_step=d_step, use_cuda=use_cuda)
    # dr = BerObj.get_dr(r, dPyx)

    # print("[+] Capacities are:")
    # print("[=] Initial # of deltas: {}".format(C))
    # print("[=] New # of deltas:     {}".format(C_new[-1]))
    if C < C_new[-1]:
        return True, C_new, r, dr
    
    BerObj.update_input_size(len(old_theta_vector), old_theta_vector)
    return False, None, None, None

def check_if_middle_delta(BerObj: Ber, step: float = 0.001, use_cuda: bool = True) -> bool:

    old_theta_vector = BerObj.theta_vector

    theta_vector_helper = BerObj.theta_vector + [0.5 - step, 0.5 + step]
    odd = True if BerObj.input_size % 2 == 1 else False
    if odd == False:
        theta_vector_helper += [0.5]
    theta_vector_helper.sort()

    BerObj.update_input_size(len(theta_vector_helper), theta_vector_helper)

    dPyx = BerObj.update_Pyx_get_dPyx()
    _, r, dr = blahut_arimoto_derivative(BerObj, 0.05)
    # dr = BerObj.get_dr(r, dPyx)

    middle_index = int((BerObj.input_size - 1) / 2 - 1)

    BerObj.update_input_size(len(old_theta_vector), old_theta_vector)
    if dr[middle_index] > 0:
        return True
    return False

def blahut_arimoto_derivative(BerObj: Ber,  d_threshold: float, d_step: float = 0.001, use_cuda: bool = True, log_base: float = 2, threshold: float = 1e-5, max_iter: int = 1e3) -> tuple:

    odd = True if BerObj.input_size % 2 == 1 else False

    dPyx = BerObj.update_Pyx_get_dPyx()
    C, r = blahut_arimoto(BerObj, use_cuda=use_cuda, log_base=log_base, threshold=threshold, max_iter=max_iter)
    dr = BerObj.get_dr(r, dPyx)

    need_fixing = True
    iterations = 0
    while need_fixing:# and iterations < max_iter:
        # print("[-] Calibrating deltas, iteration {}".format(iterations + 1))
        new_theta_vector = BerObj.theta_vector
        middle_index = int((BerObj.input_size - 1) / 2) if odd else int(BerObj.input_size / 2)

        need_fixing = False
        for index in range(1, middle_index):
            if abs(dr[index]) > d_threshold:
                direction = 1 if dr[index] > 0 else -1
                new_theta_vector[index] += direction * d_step
                new_theta_vector[-(index + 1)] -= direction * d_step
                need_fixing = True
        
        # print("[-] theta: {}".format(new_theta_vector))
        # print("[-] dr: {}".format(dr))

        if need_fixing == False:
            break

        BerObj.update_theta_vector(new_theta_vector)
        dPyx = BerObj.update_Pyx_get_dPyx()
        C, r = blahut_arimoto(BerObj, use_cuda=use_cuda, log_base=log_base, threshold=threshold, max_iter=max_iter)
        dr = BerObj.get_dr(r, dPyx)
        
        iterations += 1

    return C, r, dr


def blahut_arimoto(BerObj: Ber, conditional: bool = False, use_cuda: bool = True,  log_base: float = 2, threshold: float = 1e-5, max_iter: int = 1e3) -> tuple:

    if conditional:
        C, r = blahut_arimoto_conditional_cpu(BerObj, log_base=log_base, threshold=threshold, max_iter=max_iter)
    else:
        if use_cuda is True:
            C, r = blahut_arimoto_cuda(BerObj.Pyx, log_base=log_base, threshold=threshold, max_iter=max_iter)
        
        else:
            C, r = blahut_arimoto_cpu(BerObj.Pyx, log_base=log_base, threshold=threshold, max_iter=max_iter)

    return C, r

def blahut_arimoto_cuda(Pyx: tr.Tensor,  log_base: float = 2, threshold: float = 1e-10, max_iter: int = 1e3) -> tuple:
    '''
    Maximize the capacity between I(X;Y)
    Pyx: each row represnets probability assinmnet
    log_base: the base of the log when calaculating the capacity
    thresh: the threshold of the update, finish the calculation when gettting to it.
    max_iter: the maximum iterations of the calculation
    '''

    CUDA_DEVICE = tr.device('cuda')

    # Input test
    if tr.abs(Pyx.sum(axis=1).mean() - 1) > 1e-6 or Pyx.shape[0] <= 1:
        raise Exception("[!] BlahutArimoto: Bad Pyx matrix")

    # The number of inputs: size of |X|
    m = Pyx.shape[0]

    # The number of outputs: size of |Y|
    n = Pyx.shape[1]

    # Initialize the prior uniformly
    r = tr.ones((m, 1)) / m
    c = tr.zeros((int(max_iter), 1), dtype=tr.float64)

    Pyx_cuda = Pyx.to(device=CUDA_DEVICE)
    r_cuda = r.to(device=CUDA_DEVICE)
    c_cuda = c.to(device=CUDA_DEVICE)

    with tr.cuda.device(CUDA_DEVICE):

        # Compute the r(x) that maximizes the capacity
        iteration = 0
        for _ in range(int(max_iter)):

            q = (r_cuda * Pyx_cuda)
            q = tr.Tensor(q / tr.sum(q, axis=0))

            r1 = tr.Tensor((tr.prod(tr.pow(q, Pyx_cuda), axis=1)))
            r1 = tr.Tensor(r1 / tr.sum(r1)).reshape(m, 1)

            # tolerance = tr.linalg.norm(r1 - r_cuda)
            r_cuda = r1

            c_cuda[iteration] = calculate_capacity(Pyx_cuda, r1, q, log_base)

            if iteration == 0:
                tolerance = threshold + 1
            else:
                tolerance = c_cuda[iteration] - c_cuda[iteration-1]

            if tolerance < threshold:
                break

            iteration += 1

    r = r_cuda.to('cpu')
    c = c_cuda.to('cpu')

    return c[:iteration+1], r

def blahut_arimoto_cpu(Pyx: tr.Tensor,  log_base: float = 2, threshold: float = 1e-10, max_iter: int = 1e3) -> tuple:
    '''
    Maximize the capacity between I(X;Y)
    Pyx: each row represnets probability assinmnet
    log_base: the base of the log when calaculating the capacity
    thresh: the threshold of the update, finish the calculation when gettting to it.
    max_iter: the maximum iterations of the calculation
    '''

    # Input test
    if tr.abs(Pyx.sum(axis=1).mean() - 1) > 1e-6 or Pyx.shape[0] <= 1:
        raise Exception("[!] BlahutArimoto: Bad Pyx matrix")

    # The number of inputs: size of |X|
    m = Pyx.shape[0]

    # The number of outputs: size of |Y|
    n = Pyx.shape[1]

    # Initialize the prior uniformly
    r = tr.ones((m, 1)) / m
    c = tr.zeros((int(max_iter), 1), dtype=tr.float64)

    # Compute the r(x) that maximizes the capacity
    iteration = 0
    for _ in range(int(max_iter)):

        q = (r * Pyx)
        q = tr.Tensor(q / tr.sum(q, axis=0))

        r1 = tr.Tensor((tr.prod(tr.pow(q, Pyx), axis=1)))
        r1 = tr.Tensor(r1 / tr.sum(r1)).reshape(m, 1)

        # tolerance = tr.linalg.norm(r1 - r)
        r = r1

        c[iteration] = calculate_capacity(Pyx, r1, q, log_base)

        if iteration == 0:
            tolerance = threshold + 1
        else:
            tolerance = c[iteration] - c[iteration-1]

        if tolerance < threshold:
            break

        iteration += 1

    return c[:iteration+1], r

def blahut_arimoto_conditional_cpu(BerObj: Ber,  log_base: float = 2, threshold: float = 1e-10, max_iter: int = 1e3) -> tuple:
    '''
    Maximize the capacity between I(X;Y)
    Pyx: each row represnets probability assinmnet
    log_base: the base of the log when calaculating the capacity
    thresh: the threshold of the update, finish the calculation when gettting to it.
    max_iter: the maximum iterations of the calculation
    '''

    # Input test
    if tr.abs(BerObj.Pyx.sum(axis=1).mean() - 1) > 1e-6 or BerObj.Pyx.shape[0] <= 1:
        raise Exception("[!] BlahutArimoto: Bad Pyx matrix")

    # The number of inputs: size of |X|
    m = BerObj.Pyx.shape[0]

    # The number of outputs: size of |Y|
    n = BerObj.Pyx.shape[1]

    # Initialize the prior uniformly
    P_theta_given_Yn = BerObj.Pyx / sum(BerObj.Pyx)
    c = tr.zeros((int(max_iter), 1), dtype=tr.float64)

    P_Yn1_given_theta = tr.zeros((BerObj.input_size, 2), dtype=tr.float64)
    index = 0
    for theta in BerObj.theta_vector:
        P_Yn1_given_theta[index,0] = theta
        P_Yn1_given_theta[index,1] = 1 - theta
        index += 1

    helper = P_Yn1_given_theta.reshape((BerObj.input_size,1,2))
    P_Yn1_given_theta = P_Yn1_given_theta.reshape((BerObj.input_size,1,2))
    for _ in range(BerObj.output_size_matrix - 1):
        P_Yn1_given_theta = tr.cat((P_Yn1_given_theta,helper), dim=1)
    helper2 = BerObj.Pyx
    helper2 = tr.stack((helper2,helper2), dim=2)

    P_Yn1_given_theta = P_Yn1_given_theta * helper2

    for row in range(P_Yn1_given_theta.shape[0]):
        output_size = P_Yn1_given_theta.shape[1]
        for col in range(output_size):
            P_Yn1_given_theta[row, col, 0] = P_Yn1_given_theta[row, col, 0] / math.comb(output_size+1, col) * math.comb(output_size+1, col)
            P_Yn1_given_theta[row, col, 1] = P_Yn1_given_theta[row, col, 1] / math.comb(output_size+1, col+1) * math.comb(output_size+1, col+1)
    
    helper = helper.reshape((BerObj.input_size,2))

    # Compute the r(x) that maximizes the capacity
    iteration = 0
    for _ in range(int(max_iter)):

        P_theta_given_Yn_step = tr.stack((P_theta_given_Yn, P_theta_given_Yn), dim=2)

        P_theta_given_Yn1 = P_theta_given_Yn_step
        for index in range(BerObj.output_size_matrix):
            P_theta_given_Yn1[:,index,:] = P_theta_given_Yn_step[:,index,:] * helper
        
        P_theta_given_Yn1 = tr.Tensor(P_theta_given_Yn1 / tr.sum(P_theta_given_Yn1, axis=0))

        P_theta_given_Yn = tr.Tensor((tr.prod(tr.pow(P_theta_given_Yn1, P_Yn1_given_theta), axis=2)))
        P_theta_given_Yn = tr.Tensor(P_theta_given_Yn / tr.sum(P_theta_given_Yn))

        P_theta = 1 / tr.sum(BerObj.Pyx / P_theta_given_Yn, axis=1)

        c[iteration] = calculate_capacity_conditional(P_theta, BerObj.Pyx, P_Yn1_given_theta, P_theta_given_Yn, P_theta_given_Yn1, log_base)

        if iteration == 0:
            tolerance = threshold + 1
        else:
            tolerance = c[iteration] - c[iteration-1]

        if abs(tolerance) < threshold:
            break

        iteration += 1

    return c[:iteration+1], P_theta

def blahut_arimoto_conditional_onestep_cpu(BerObj: Ber,  log_base: float = 2, threshold: float = 1e-10, max_iter: int = 1e3) -> tuple:
    '''
    Maximize the capacity between I(X;Y)
    Pyx: each row represnets probability assinmnet
    log_base: the base of the log when calaculating the capacity
    thresh: the threshold of the update, finish the calculation when gettting to it.
    max_iter: the maximum iterations of the calculation
    '''

    # Input test
    if tr.abs(BerObj.Pyx.sum(axis=1).mean() - 1) > 1e-6 or BerObj.Pyx.shape[0] <= 1:
        raise Exception("[!] BlahutArimoto: Bad Pyx matrix")

    # The number of inputs: size of |X|
    m = BerObj.Pyx.shape[0]

    # The number of outputs: size of |Y|
    n = BerObj.Pyx.shape[1]

    # Initialize the prior uniformly
    P_theta_given_Yn = BerObj.Pyx / sum(BerObj.Pyx)
    c = tr.zeros((int(max_iter), 1), dtype=tr.float64)

    P_Yn1_given_theta = tr.zeros((BerObj.input_size, 2), dtype=tr.float64)
    index = 0
    for theta in BerObj.theta_vector:
        P_Yn1_given_theta[index,0] = theta
        P_Yn1_given_theta[index,1] = 1 - theta
        index += 1

    helper = P_Yn1_given_theta.reshape((BerObj.input_size,1,2))
    P_Yn1_given_theta = P_Yn1_given_theta.reshape((BerObj.input_size,1,2))
    for _ in range(BerObj.output_size_matrix - 1):
        P_Yn1_given_theta = tr.cat((P_Yn1_given_theta,helper), dim=1)
    helper2 = BerObj.Pyx
    helper2 = tr.stack((helper2,helper2), dim=2)

    P_Yn1_given_theta = P_Yn1_given_theta * helper2

    for row in range(P_Yn1_given_theta.shape[0]):
        output_size = P_Yn1_given_theta.shape[1]
        for col in range(output_size):
            P_Yn1_given_theta[row, col, 0] = P_Yn1_given_theta[row, col, 0] / math.comb(output_size, col) * math.comb(output_size+1, col)
            P_Yn1_given_theta[row, col, 1] = P_Yn1_given_theta[row, col, 1] / math.comb(output_size, col) * math.comb(output_size+1, col+1)
    
    helper = helper.reshape((BerObj.input_size,2))

    # Compute the r(x) that maximizes the capacity
    iteration = 0
    for _ in range(int(max_iter)):

        P_theta_given_Yn_step = tr.stack((P_theta_given_Yn, P_theta_given_Yn), dim=2)

        P_theta_given_Yn1 = P_theta_given_Yn_step
        for index in range(BerObj.output_size_matrix):
            P_theta_given_Yn1[:,index,:] = P_theta_given_Yn_step[:,index,:] * helper
        
        P_theta_given_Yn1 = tr.Tensor(P_theta_given_Yn1 / tr.sum(P_theta_given_Yn1, axis=0))

        P_theta_given_Yn = tr.Tensor((tr.prod(tr.pow(P_theta_given_Yn1, P_Yn1_given_theta), axis=2)))
        P_theta_given_Yn = tr.Tensor(P_theta_given_Yn / tr.sum(P_theta_given_Yn))

        P_theta = 1 / tr.sum(BerObj.Pyx / P_theta_given_Yn, axis=1)

        c[iteration] = calculate_capacity_conditional(P_theta, BerObj.Pyx, P_Yn1_given_theta, P_theta_given_Yn, P_theta_given_Yn1, log_base)

        if iteration == 0:
            tolerance = threshold + 1
        else:
            tolerance = c[iteration] - c[iteration-1]

        if abs(tolerance) < threshold:
            break

        iteration += 1

    return c[:iteration+1], P_theta
