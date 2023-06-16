import time
from dataclasses import dataclass
from typing import List, Dict, Optional

from scipy.stats import binom
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb


@dataclass
class IterativeAlgoResults:
    thetas: np.ndarray
    weights: List[np.ndarray]
    capacities: List[float]


@dataclass
class NextSymbolProbabilities:
    training_type: List[int]
    probability: List[float]


def get_type_probabilities_given_theta(theta: float, number_of_outcomes: int) -> np.ndarray:
    return np.asarray([binom.pmf(index, number_of_outcomes, theta) for index in range(number_of_outcomes + 1)])


def get_binary_entropy(theta: float) -> float:
    if theta == 0 or theta == 1:
        return 0.
    return - theta * np.log(theta) - ((1-theta) * np.log(1-theta))


def _get_types_log_binomial_coefficients(number_of_outcomes: int) -> np.ndarray:
    return np.log(np.asarray([comb(number_of_outcomes, k) for k in range(number_of_outcomes + 1)]))


def _get_type_probabilities_vs_theta(thetas: np.ndarray, number_of_outcomes: int) -> Dict[float, np.ndarray]:
    return {
        theta: get_type_probabilities_given_theta(theta=theta, number_of_outcomes=number_of_outcomes)
        for theta in thetas
    }


def get_probability_for_next_symbol_being_one(
    thetas: np.ndarray,
    weights: np.ndarray,
    training_set_size: int,
    probability_of_type_given_theta: Optional[Dict[float, np.ndarray]] = None,
) -> NextSymbolProbabilities:
    training_types: List[int] = []
    probabilities: List[float] = []
    if probability_of_type_given_theta is None:
        probability_of_type_given_theta = _get_type_probabilities_vs_theta(
            thetas=thetas, number_of_outcomes=training_set_size
        )
    for curr_type in range(training_set_size + 1):
        probability_of_type_and_theta = [
            weights[curr_index]*probability_of_type_given_theta[thetas[curr_index]][curr_type]
            for curr_index in range(len(thetas))
        ]
        probability_of_type = sum(probability_of_type_and_theta)
        probability_of_type_and_theta_times_theta = [
            weights[curr_index]*probability_of_type_given_theta[thetas[curr_index]][curr_type]*thetas[curr_index]
            for curr_index in range(len(thetas))
        ]
        training_types.append(curr_type)
        probabilities.append(sum(probability_of_type_and_theta_times_theta) / probability_of_type)
    return NextSymbolProbabilities(
        training_type=training_types, probability=probabilities,
    )


class IterativeAlgorithmSolver:

    def __init__(self, number_of_thetas: int, training_set_size: int):
        self.number_of_thetas = number_of_thetas
        self.thetas = np.linspace(0, 1, self.number_of_thetas)
        self.weights = self.thetas*0 + np.divide(1, number_of_thetas)
        self.training_set_size = training_set_size
        self.whole_sequence_size = self.training_set_size + 1
        self.type_probabilities_given_theta: Dict[int, Dict[float, np.ndarray]] = {
            self.training_set_size: self._get_type_probability_all_thetas(self.training_set_size),
            self.whole_sequence_size: self._get_type_probability_all_thetas(self.whole_sequence_size)
        }
        self.types_log_binomial_coefficients: Dict[int, np.ndarray] = {
            self.training_set_size: _get_types_log_binomial_coefficients(self.training_set_size),
            self.whole_sequence_size: _get_types_log_binomial_coefficients(self.whole_sequence_size)
        }
        self.solution: Optional[IterativeAlgoResults] = None
        self._divergences_per_iteration = []

    def _get_type_probability_all_thetas(self, number_of_outcomes: int) -> Dict[float, np.ndarray]:
            return _get_type_probabilities_vs_theta(number_of_outcomes=number_of_outcomes, thetas=self.thetas)

    def get_sequence_entropy_given_types_probabilities(self, vector: np.ndarray):
        return -np.sum(vector * (np.log(vector) - self.types_log_binomial_coefficients[len(vector) - 1]))

    def get_average_probability_of_outcome(self, weights: np.ndarray, thetas: np.ndarray, number_of_outcomes: int):
        probabilities = np.asarray([0. for _ in range(number_of_outcomes + 1)])
        for index in range(len(weights)):
            curr_probabilities = self.type_probabilities_given_theta[number_of_outcomes][thetas[index]]
            probabilities += weights[index] * curr_probabilities
        return probabilities

    def get_weighted_outcomes_log_loss_wrt_true_theta(
        self, theta: float, average_type_probabilities: np.ndarray
    ) -> float:
        number_of_outcomes = len(average_type_probabilities) - 1
        true_probabilities = self.type_probabilities_given_theta[number_of_outcomes][theta]
        return -sum(
            true_probabilities*(
                    np.log(average_type_probabilities) - self.types_log_binomial_coefficients[number_of_outcomes]
            )
        )

    def get_divergences(self, thetas: np.ndarray, weights: np.ndarray, training_size: int) -> np.ndarray:
        divergences = thetas * 0
        average_probabilities_training_size = self.get_average_probability_of_outcome(
            weights=weights, thetas=thetas, number_of_outcomes=training_size
        )
        average_probabilities_whole_sequence = self.get_average_probability_of_outcome(
            weights=weights, thetas=thetas, number_of_outcomes=self.whole_sequence_size
        )
        for curr_index in range(len(thetas)):
            curr_theta = thetas[curr_index]
            a0 = get_binary_entropy(theta=curr_theta)
            a1 = self.get_weighted_outcomes_log_loss_wrt_true_theta(
                theta=curr_theta, average_type_probabilities=average_probabilities_training_size
            )
            a2 = self.get_weighted_outcomes_log_loss_wrt_true_theta(
                theta=curr_theta, average_type_probabilities=average_probabilities_whole_sequence
            )
            divergences[curr_index] = a2 - (a0 + a1)
        return divergences

    def update_weights(self):
        divergences = self.get_divergences(thetas=self.thetas, weights=self.weights, training_size=self.training_set_size)
        self._divergences_per_iteration.append(divergences)
        unweighted_probabilities = self.weights * np.exp(divergences)
        self.weights = unweighted_probabilities / sum(unweighted_probabilities)

    def get_average_binary_entropy(self) -> float:
        return sum([self.weights[i] * get_binary_entropy(self.thetas[i]) for i in range(len(self.thetas))])

    def get_capacity(self):
        average_probabilities_training_size = self.get_average_probability_of_outcome(
            weights=self.weights, thetas=self.thetas, number_of_outcomes=self.training_set_size
        )
        average_probabilities_whole_sequence = self.get_average_probability_of_outcome(
            weights=self.weights, thetas=self.thetas, number_of_outcomes=self.whole_sequence_size
        )
        whole_sequence_log_loss = self.get_sequence_entropy_given_types_probabilities(
            average_probabilities_whole_sequence
        )
        training_log_loss = self.get_sequence_entropy_given_types_probabilities(average_probabilities_training_size)
        return whole_sequence_log_loss - (training_log_loss + self.get_average_binary_entropy())

    def plot_results(self):
        if self.solution is None:
            print('cant plot results because no solution was obtained')
            return
        plt.plot(self.solution.capacities)
        plt.title('capacity per iteration')
        plt.waitforbuttonpress()
        plt.close()
        best_weights = self.solution.weights[-1]
        plt.plot(self.solution.thetas, best_weights, label='weights')
        plt.plot(self.solution.thetas, self._divergences_per_iteration[-1], label='divergences')
        plt.title('weight per theta')
        plt.legend()
        plt.waitforbuttonpress()
        plt.close()
        probabilities = get_probability_for_next_symbol_being_one(
            thetas=self.thetas, weights=best_weights, training_set_size=self.training_set_size
        )
        plt.plot(probabilities.training_type, probabilities.probability)
        plt.title('probability of next symbol being one given type')
        plt.waitforbuttonpress()
        plt.close()

    def single_step_am(self, iterations: int) -> IterativeAlgoResults:
        iteration = 0
        weights_per_iteration = [self.weights]
        capacities_per_iteration = [self.get_capacity()]
        while iteration < iterations:
            iter_t0 = time.time()
            self.update_weights()
            weights_per_iteration.append(self.weights)
            capacities_per_iteration.append(self.get_capacity())
            iteration += 1
            print(f'total time for iteration {iteration}: {time.time() - iter_t0}')
        last_capacity = self.get_capacity()
        normalized_last_capacity = last_capacity * self.training_set_size
        print(f'last capacity is {last_capacity}, normalized by N: {normalized_last_capacity}')
        self.solution = IterativeAlgoResults(
            weights=weights_per_iteration, capacities=capacities_per_iteration, thetas=self.thetas
        )
        return self.solution


if __name__ == '__main__':
    t0 = time.time()
    solver = IterativeAlgorithmSolver(number_of_thetas=100, training_set_size=100)
    results = solver.single_step_am(iterations=2500)
    print(f'total running time: {time.time() - t0}')
    solver.plot_results()