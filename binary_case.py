import pickle
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable
import os

from scipy.stats import binom
import numpy as np
import torch as tr
import matplotlib.pyplot as plt
from scipy.special import comb

Samples = int
PositiveSamples = int

ProbabilityAssigner = Callable[[Samples, PositiveSamples], float]

@dataclass
class IterativeAlgoResults:
    thetas: np.ndarray
    weights: List[np.ndarray]
    capacities: List[float]

@dataclass
class SingleSolution:
    thetas: np.ndarray
    weights: np.ndarray

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


def _zero_probabilities_way_below_median(weights: np.ndarray, minimal_ratio_to_median: float = 0.05) -> np.ndarray:
    median_weight = np.median(weights)
    threshold = median_weight * minimal_ratio_to_median
    indices_below_threshold = np.where(weights < threshold)[0]
    if len(indices_below_threshold) == 0:
        return weights
    sum_of_weights_below_threshold = sum(weights[indices_below_threshold])
    new_weights = weights / (1 - sum_of_weights_below_threshold)
    new_weights[indices_below_threshold] = 0
    return new_weights


def _rescale_resolution(thetas: np.ndarray, weights: np.ndarray, new_thetas) -> np.ndarray:
    return np.interp(x=new_thetas, xp=thetas, fp=weights)


def get_divergence(p: float, q: float):
    if p == 0:
        return -np.log(1-q)
    if p == 1:
        return -np.log(q)
    return p * np.log(p / q) + (1-p) * np.log((1-p) / (1-q))


def get_regret_specific_empirical_distribution(
        p: float, samples: int, positives: int, probability_assigner: ProbabilityAssigner
) -> float:
    q = probability_assigner(samples, positives)
    return get_divergence(p, q)


def get_average_regret(p: float, samples: int, probability_assigner: ProbabilityAssigner) -> float:
    return sum([binom.pmf(n, samples, p) * get_regret_specific_empirical_distribution(
        p=p, samples=samples, positives=n, probability_assigner=probability_assigner
    ) for n in range(samples + 1)])


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


def braess_probability_assignment(samples: int, positives: int):
    if positives == 0 or positives == samples:
        beta = 0.5
        return np.divide(positives + beta, samples + 2*beta)
    if positives == samples - 1 or positives == 1:
        beta = 1
        return np.divide(positives + beta, samples + 2*beta)
    return np.divide(positives + 0.75, samples + 1.5)


def get_braess_probability_assignment_for_each_case(samples: int) -> np.ndarray:
    ret_list = [float(braess_probability_assignment(samples=samples, positives=p)) for p in range(samples + 1)]
    return np.asarray(ret_list)


def _translate_probability_to_add_beta(positives: int, samples: int, probability: float, default_val: float = 0) -> float:
    if np.abs(probability - 0.5) < 0.001:
        return default_val
    return float(np.divide(probability * samples - positives, 1 - 2 * probability))


def _translate_probabilities_to_add_beta(probabilities: NextSymbolProbabilities) -> np.ndarray:
    samples = probabilities.training_type[-1]
    ret_list = []
    for curr_ind in range(len(probabilities.probability)):
        probability = probabilities.probability[curr_ind]
        positives = probabilities.training_type[curr_ind]
        ret_list.append(_translate_probability_to_add_beta(positives=positives, probability=probability, samples=samples))
    return np.asarray(ret_list)


class IterativeAlgorithmSolver:

    def __init__(
            self,
            number_of_thetas: int,
            training_set_size: int,
            use_braess_sauer_as_initialization: bool = False,
            initialization: Optional[SingleSolution] = None
    ):
        t0 = time.time()
        self.number_of_thetas = number_of_thetas
        self.thetas = np.linspace(0.2, 0.8, self.number_of_thetas)
        self.weights = self.thetas*0 + np.divide(1, number_of_thetas)
        self.training_set_size = training_set_size
        self.whole_sequence_size = self.training_set_size + 1
        if initialization is not None:
            self.weights = _rescale_resolution(
                thetas=initialization.thetas, weights=initialization.weights, new_thetas=self.thetas
            )
        else:
            if use_braess_sauer_as_initialization:
                divergences_exp = np.exp(np.asarray(
                    [get_average_regret(
                        probability_assigner=braess_probability_assignment, p=p, samples=self.training_set_size
                    ) for p in self.thetas]))
                normalization_factor = sum(divergences_exp)
                self.weights = np.divide(divergences_exp, normalization_factor)
        self.type_probabilities_given_theta: Dict[int, Dict[float, np.ndarray]] = {
            self.training_set_size: self._get_type_probability_all_thetas(self.training_set_size),
            self.whole_sequence_size: self._get_type_probability_all_thetas(self.whole_sequence_size)
        }
        self.max_divergence_to_average_divergence_ratio = []
        self.types_log_binomial_coefficients: Dict[int, np.ndarray] = {
            self.training_set_size: _get_types_log_binomial_coefficients(self.training_set_size),
            self.whole_sequence_size: _get_types_log_binomial_coefficients(self.whole_sequence_size)
        }
        self.solution: Optional[IterativeAlgoResults] = None
        self._divergences_per_iteration = []
        self._probability_for_next_symbol_being_one: Optional[NextSymbolProbabilities] = None
        print(f'solver initialization time is {time.time() - t0}')

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

    def _set_probability_of_next_symbol_being_one(self):
        if self.solution is None:
            print('cant set probabilities because no solution was obtained')
            return
        best_weights = self.solution.weights[-1]
        self._probability_for_next_symbol_being_one = get_probability_for_next_symbol_being_one(
            thetas=self.thetas, weights=best_weights, training_set_size=self.training_set_size
        )

    def probabilities_of_next_symbol_being_one(self) -> NextSymbolProbabilities:
        if self._probability_for_next_symbol_being_one is None:
            self._set_probability_of_next_symbol_being_one()
        return self._probability_for_next_symbol_being_one

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
        average_divergence = sum(self.weights * divergences)
        self.max_divergence_to_average_divergence_ratio.append((max(divergences) / average_divergence))
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
        plt.savefig(f'plots/conditional/solver_n_{self.training_set_size}_capacity.png')
        # plt.waitforbuttonpress()
        plt.close()
        plt.plot(self.max_divergence_to_average_divergence_ratio)
        plt.title('max to average divergence ratio')
        plt.savefig(f'plots/conditional/solver_n_{self.training_set_size}_divergence.png')
        # plt.waitforbuttonpress()
        plt.close()
        best_weights = self.solution.weights[-1]
        plt.plot(self.solution.thetas, best_weights, label='weights')
        plt.plot(self.solution.thetas, self._divergences_per_iteration[-1], label='divergences')
        x_range = max(self.solution.thetas) - min(self.solution.thetas)
        plt.xlim(min(self.solution.thetas) + x_range/5, max(self.solution.thetas) - x_range/5)
        plt.title('weight per theta')
        plt.legend()
        plt.savefig(f'plots/conditional/solver_n_{self.training_set_size}_weight_short.png')
        # plt.waitforbuttonpress()
        plt.close()
        best_weights = self.solution.weights[-1]
        plt.semilogy(self.solution.thetas, best_weights, label='weights')
        plt.semilogy(self.solution.thetas, self._divergences_per_iteration[-1], label='divergences')
        plt.xlim(min(self.solution.thetas) + x_range/5, max(self.solution.thetas) - x_range/5)
        plt.title('weight per theta')
        plt.legend()
        plt.savefig(f'plots/conditional/solver_n_{self.training_set_size}_weight_log_short.png')
        # plt.waitforbuttonpress()
        plt.close()
        probabilities = self.probabilities_of_next_symbol_being_one()
        brass_probabilities = get_braess_probability_assignment_for_each_case(samples=self.training_set_size)
        if isinstance(probabilities, list):
            plt.plot(list(range(self.training_set_size + 1)), probabilities, label='AM results')
        else:
            plt.plot(list(range(self.training_set_size + 1)), probabilities.probability, label='AM results')
        plt.plot(list(range(self.training_set_size + 1)), brass_probabilities, label='Braess-Sauer')
        plt.legend()
        plt.title('probability of next symbol being one given type')
        plt.savefig(f'plots/conditional/solver_n_{self.training_set_size}_p_next_symbol.png')
        # plt.waitforbuttonpress()
        plt.close()
        if not isinstance(probabilities, list):
            betas = _translate_probabilities_to_add_beta(probabilities)
            plt.plot(probabilities.training_type, betas, label='derived add-beta')
            plt.plot([0, max(probabilities.training_type)], [0.5, 0.5], label='beta=0.5')
            plt.plot([0, max(probabilities.training_type)], [0.75, 0.75], label='beta=0.75')
            plt.plot([0, max(probabilities.training_type)], [1, 1], label='beta=1')
            plt.title('derived add-beta')
            plt.savefig(f'plots/conditional/solver_n_{self.training_set_size}_add_beta.png')
            # plt.waitforbuttonpress()
            plt.close()

    def print_interesting_solution_properties(self):
        if self.solution is None:
            print('cant plot results because no solution was obtained')
            return
        best_weights = self.solution.weights[-1]
        max_weight = max(best_weights)
        median_weight = np.median(best_weights)
        min_weight = min(best_weights)
        much_lower_than_median = best_weights < median_weight*0.1
        much_lower_than_median_area = sum(much_lower_than_median) / len(best_weights)
        print(f'results for N={self.training_set_size}')
        print(f' Number of Iterations: {len(self.solution.weights) - 1}, number of thetas: {self.number_of_thetas}')
        print(f'estimated normalized capacity: {self.solution.capacities[-1] * self.training_set_size}')
        print(f'last ratio between max and average divergences {self.max_divergence_to_average_divergence_ratio[-1]}')
        print(f'max weight is {max_weight}, median weight is {median_weight}, min_weight is {min_weight}')
        print(f'max to median ratio is {max_weight / median_weight}')
        if min_weight > 0:
            print(f'median to min ratio is {median_weight / min_weight}')
        print(f'total much lower than median area size is : {much_lower_than_median_area}')

    def single_step_am(
            self, iterations: int, calc_probability: bool = False, minimal_divergence_ratio: float = 1.
    ) -> IterativeAlgoResults:
        iteration = 0
        # t0 = time.time()
        # total_iterations_for_expected_time_calculation = 10
        weights_per_iteration = [self.weights]
        capacities_per_iteration = [self.get_capacity()]
        total_t0 = time.time()
        while iteration < iterations:
            self.update_weights()
            weights_per_iteration.append(self.weights)
            capacities_per_iteration.append(self.get_capacity())
            iteration += 1
            if iteration == 1:
                delta_t = time.time() - total_t0
                print(f'expected time for iterations is: {iterations * delta_t}')
            if np.mod(iteration, 100) == 0:
                print(f'total time up to iteration {iteration}: {time.time() - total_t0}')
            if len(self.max_divergence_to_average_divergence_ratio) > 0:
                last_ratio = self.max_divergence_to_average_divergence_ratio[-1]
                if last_ratio < minimal_divergence_ratio:
                    print(
                        f'exited after {iteration} iterations due to max to average divergence ratio being {last_ratio}'
                    )
        last_capacity = self.get_capacity()
        normalized_last_capacity = last_capacity * self.training_set_size
        print(f'total single step am iterations time: {time.time() - total_t0}')
        print(f'last capacity is {last_capacity}, normalized by N: {normalized_last_capacity}')
        self.solution = IterativeAlgoResults(
            weights=weights_per_iteration, capacities=capacities_per_iteration, thetas=self.thetas
        )
        if calc_probability:
            probability_calc_t0 = time.time()
            self._set_probability_of_next_symbol_being_one()
            print(f'probability calculation time {time.time() - probability_calc_t0 }')
        return self.solution

    def zero_low_probabilities(self, ratio_to_median_threshold: float = 0.1):
        self.weights = _zero_probabilities_way_below_median(
            weights=self.weights, minimal_ratio_to_median=ratio_to_median_threshold
        )


def _main_dump_results(training_set_size: int, num_of_thetas: int, iterations: int):
    solver = IterativeAlgorithmSolver(
        number_of_thetas=num_of_thetas, training_set_size=training_set_size, use_braess_sauer_as_initialization=True
    )
    solver.single_step_am(iterations=iterations, calc_probability=True, minimal_divergence_ratio=1.001)
    # time_str = time.strftime("_%H-%M-%S_%d-%m-%Y", time.localtime())
    pickle.dump(solver, open(f'outputs/conditional/solver_n_{training_set_size}.pkl', 'wb'))


def _main_plot_results(training_set_size: int):
    a: IterativeAlgorithmSolver = pickle.load(open(f'outputs/conditional/solver_n_{training_set_size}.pkl', 'rb'))
    a.plot_results()


def _main_print_results(training_set_size: int):
    a: IterativeAlgorithmSolver = pickle.load(open(f'outputs/conditional/solver_n_{training_set_size}.pkl', 'rb'))
    a.print_interesting_solution_properties()


# def check_initialization_difference(training_set_size: int):
#     number_of_thetas = training_set_size * 10
#     a = IterativeAlgorithmSolver(
#         number_of_thetas=number_of_thetas, training_set_size=training_set_size, use_braess_sauer_as_initialization=True
#     )
#     a1 = a.get_capacity()*training_set_size
#     print(f'Braess-Sauer init gives c*N={a.get_capacity()* training_set_size}')
#     c: IterativeAlgorithmSolver = pickle.load(open(f'solver_n_{100}.pkl', 'rb'))
#     b = IterativeAlgorithmSolver(
#         number_of_thetas=number_of_thetas, training_set_size=training_set_size, use_braess_sauer_as_initialization=False,
#         initialization=SingleSolution(thetas=c.thetas, weights=c.solution.weights[-1]),
#     )
#     b1 = b.get_capacity() * training_set_size
#     print(f'Uniform init gives c*N={b.get_capacity()* training_set_size}')
#     print(f'delta is {b1 - a1}')

def _check_zero_low_probabilities(training_set_size: int):
    a: IterativeAlgorithmSolver = pickle.load(open(f'solver_n_{training_set_size}.pkl', 'rb'))
    c0 = a.get_capacity()
    a.zero_low_probabilities(ratio_to_median_threshold=0.1)
    c1 = a.get_capacity()
    delta = c1 - c0
    normalized_delta = delta * training_set_size
    print(f'normalized delta is {normalized_delta}')

if __name__ == '__main__':

    training_set_size = 100
    training_set_size_step = 100
    number_of_thetas = 2000
    number_of_thetas_step = 0
    iterations = 10000
    iterations_step = 0

    for training_set_size in range(100, 200, 100):
        _main_dump_results(training_set_size=training_set_size, num_of_thetas=number_of_thetas, iterations=iterations)

    # check_initialization_difference(200)
    # _main_dump_results(45)
    # _main_dump_results(50)
    # _main_dump_results(100)
    # _main_dump_results(200)
    # _main_plot_results(30)
    # _main_print_results(45)
    # _main_print_results(50)
    # _main_plot_results(200)
    # _main_plot_results(100)
    # _main_print_results(100)
    # _check_zero_low_probabilities(50)
    for training_set_size in range(100, 200, 100):
        _main_print_results(training_set_size)
        _main_plot_results(training_set_size)