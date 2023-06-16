import torch as tr
import common as cm
from math import comb
from BernoulliCommon import BernoulliObject as Ber

class BernoulliTypesObject(Ber):
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size)
        self.output_size_matrix = self.output_size + 1
        
    def update_Pyx(self):

        self.Pyx = tr.zeros(self.input_size, self.output_size_matrix, dtype=tr.float64)

        for row in range(self.input_size):

            theta = self.theta_vector[row]
            for col in range(self.output_size_matrix):
                n_plus = col
                n_minus = self.output_size - n_plus
                self.Pyx[row, col] = ((theta)**(n_plus)) * ((1-theta)**(n_minus)) * comb(self.output_size, col)
                if self.Pyx[row, col] == 0 and (theta != 0 and theta != 1):
                    self.Pyx[row, col] = 1e-50

    def update_Pyx_get_dPyx(self) -> tr.Tensor:

        self.Pyx = tr.zeros(self.input_size, self.output_size_matrix, dtype=tr.float64)
        dPyx = tr.zeros(self.input_size, self.output_size_matrix, dtype=tr.float64)

        for row in range(self.input_size):

            theta = self.theta_vector[row]
            for col in range(self.output_size_matrix):
                n_plus = col
                n_minus = self.output_size - n_plus
                self.Pyx[row, col] = ((theta)**(n_plus)) * ((1-theta)**(n_minus)) * comb(self.output_size, col)
                if self.Pyx[row, col] == 0 and (theta != 0 and theta != 1):
                    self.Pyx[row, col] = 1e-50
                if(theta != 0 and theta != 1):
                    dPyx[row, col] = self.Pyx[row, col] * (n_plus/theta - n_minus/(1-theta))

        return dPyx

    def get_dPyx(self) -> tr.Tensor:

        dPyx = tr.zeros(self.input_size, self.output_size_matrix, dtype=tr.float64)

        for row in range(self.input_size):

            theta = self.theta_vector[row]
            for col in range(self.output_size_matrix):
                n_plus = col
                n_minus = self.output_size - n_plus
                if(theta != 0 and theta != 1):
                    dPyx[row, col] = self.Pyx[row, col] * (n_plus/theta - n_minus/(1-theta))

        return dPyx