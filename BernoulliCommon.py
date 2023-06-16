import torch as tr
import common as cm
from math import comb

class BernoulliObject:
    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        self.theta_vector = [i for i in cm.frange(0, float(1 + 1/(input_size - 1)), float(1/(input_size - 1)))]
        self.theta_vector[-1] = 1
        self.output_size_matrix = 2**self.output_size
        self.Pyx = None
        self.Py_plus_1_x = None

    def update_input_size(self, input_size: int, theta_vector: list = None):

        self.input_size = input_size

        if theta_vector is None:
            theta_vector = [i for i in cm.frange(0, float(1 + 1/(input_size - 1)), float(1/(input_size - 1)))]    
        self.update_theta_vector(theta_vector)

    def update_theta_vector(self, theta_vector: list) -> bool:

        if self.input_size != len(theta_vector):
            raise Exception("[!] Bernoulli: New theta vector is not the same size")


        if min(theta_vector) < 0 or max(theta_vector) > 1:
            raise Exception("[!] Bernoulli: New theta vector has bad values")

        theta_vector.sort()

        self.theta_vector = theta_vector
        self.Pyx = None

    def update_Py_plus_1_x(self):

        self.Py_plus_1_x = tr.zeros(self.input_size, self.output_size_matrix + 1, dtype=tr.float64)
        for row in range(self.input_size):

            theta = self.theta_vector[row]
            for col in range(self.output_size_matrix + 1):
                n_plus = cm.hamming_weight(col)
                n_minus = self.output_size - n_plus
                self.Py_plus_1_x[row, col] = ((theta)**(n_plus)) * ((1-theta)**(n_minus)) * comb(self.output_size, col)

    def update_Pyx(self):

        self.Pyx = tr.zeros(self.input_size, self.output_size_matrix, dtype=tr.float64)
        for row in range(self.input_size):

            theta = self.theta_vector[row]
            for col in range(self.output_size_matrix):
                n_plus = cm.hamming_weight(col)
                n_minus = self.output_size - n_plus
                self.Pyx[row, col] = ((theta)**(n_plus)) * ((1-theta)**(n_minus))

    def update_Pyx_get_dPyx(self) -> tr.Tensor:

        self.Pyx = tr.zeros(self.input_size, self.output_size_matrix, dtype=tr.float64)
        dPyx = tr.zeros(self.input_size, self.output_size_matrix, dtype=tr.float64)

        for row in range(self.input_size):

            theta = self.theta_vector[row]
            for col in range(self.output_size_matrix):
                n_plus = cm.hamming_weight(col)
                n_minus = self.output_size - n_plus
                self.Pyx[row, col] = ((theta)**(n_plus)) * ((1-theta)**(n_minus))
                if(theta != 0 and theta != 1):
                    dPyx[row, col] = self.Pyx[row, col] * (n_plus/theta - n_minus/(1-theta))

        return dPyx

    def get_dPyx(self) -> tr.Tensor:

        dPyx = tr.zeros(self.input_size, self.output_size_matrix, dtype=tr.float64)

        for row in range(self.input_size):

            theta = self.theta_vector[row]
            for col in range(self.output_size_matrix):
                n_plus = cm.hamming_weight(col)
                n_minus = self.output_size - n_plus
                if(theta != 0 and theta != 1):
                    dPyx[row, col] = self.Pyx[row, col] * (n_plus/theta - n_minus/(1-theta))

        return dPyx

    def get_dr(self, r: tr.Tensor, dPyx: tr.Tensor = None,  log_base: float = 2) -> tr.Tensor:

        if len(r) != self.input_size:
            raise Exception("[!] Bernoulli: r vector is not the same size as theta vector")

        r = r.reshape(self.input_size, 1)

        denominator = tr.sum(r * self.Pyx, axis=0)
        log_vec = tr.log(self.Pyx / denominator) / tr.log(tr.Tensor([log_base]))[0]

        if dPyx is None:
            dPyx = self.get_dPyx(self.theta_vector, self.output_size, self.Pyx)

        dr = tr.sum(dPyx * log_vec, axis=1)

        return dr

'''
def get_Pyx_and_dPyx(theta_vector: list, output_size: int) -> tr.Tensor:

    input_size = len(theta_vector)
    Pyx = tr.zeros(input_size, 2**output_size)
    dPyx = tr.zeros(input_size, 2**output_size)

    for row in range(input_size):

        theta = theta_vector[row]

        for col in range(2**output_size):
            n_plus = cm.hamming_weight(col)
            n_minus = output_size - n_plus
            Pyx[row, col] = ((theta)**(n_plus)) * ((1-theta)**(n_minus))
            if(theta != 0 and theta != 1):
                dPyx[row, col] = Pyx[row, col] * (n_plus/theta - n_minus/(1-theta))

    return Pyx, dPyx

def get_Pyx(theta_vector: list, output_size: int) -> tr.Tensor:

    input_size = len(theta_vector)
    Pyx = tr.zeros(input_size, 2**output_size)

    for row in range(input_size):

        theta = theta_vector[row]

        for col in range(2**output_size):
            n_plus = cm.hamming_weight(col)
            n_minus = output_size - n_plus
            Pyx[row, col] = ((theta)**(n_plus)) * ((1-theta)**(n_minus))

    return Pyx

def get_dPyx(theta_vector: list, output_size: int, Pyx: tr.Tensor) -> tr.Tensor:

    input_size = len(theta_vector)
    dPyx = tr.zeros(input_size, 2**output_size)

    for row in range(input_size):

        theta = theta_vector[row]

        for col in range(2**output_size):
            n_plus = cm.hamming_weight(col)
            n_minus = output_size - n_plus
            if(theta != 0 and theta != 1):
                dPyx[row, col] = Pyx[row, col] * (n_plus/theta - n_minus/(1-theta))

    return dPyx

def get_Pyxy(in_size: int, out_size: int) -> tr.Tensor:

    Pyxy = tr.zeros((in_size*(2**(out_size - 1)), 2**out_size))

    row = 0
    for i in cm.frange(0, 1, float(1/in_size)):
        
        for k in range(2**(out_size - 1)):

            col = 0
            for j in range(2**out_size):
                eq_to_prev = 1 if (j >> 1) == k else 0
                Pyxy[row, col] = ((i)**(1 - (j & 1)) * ((1-i)**(j & 1)) * eq_to_prev)
                col += 1

            row += 1

    return Pyxy
'''