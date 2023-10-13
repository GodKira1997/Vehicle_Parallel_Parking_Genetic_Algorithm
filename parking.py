"""
file: parking.py
description: An agent program that finds the state and control histories for
a Dubins’ car that implements parallel parking while avoiding obstacles (e.g.,
sidewalk, other cars)
language: python3
author: Anurag Kallurwar, ak6491@rit.edu
"""


import random
import math
import numpy as np
import scipy as sc
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


def convert_to_graycode(num: int, num_of_bits: int) -> str:
    """
    Convert decimal to graycode
    :param num: decimal
    :param num_of_bits: number of bits
    :return: graycode string with given number of bits
    """
    dec_equivalent = num ^ (num >> 1)
    return np.binary_repr(dec_equivalent, width=num_of_bits)


def convert_gray_to_num(graycode, bound_contraint) -> float:
    """
    Convert graycode to float in the bound range
    :param graycode: graycode string
    :param bound_contraint: bound range
    :return: float in the given range
    """
    range = bound_contraint[1] - bound_contraint[0]
    s = len(graycode)
    lb = bound_contraint[0]
    graynum = int(graycode, 2)
    mask = graynum >> 1
    while mask:
        graynum ^= mask
        mask >>= 1
    dec = graynum
    return ((dec / (2 ** s - 1)) * range) + lb


def create_generation_zero(population: int, num_of_bits: int,
                           num_of_parameters: int) -> list:
    """
    Randomly generates a generation zero
    :param population: population of generation
    :param num_of_bits: bit size of each control parameter
    :param num_of_parameters: number of optimization parameters for each control
    :return: list of individuals
    """
    individuals = []
    for i in range(population):
        random_individual = np.random.randint(0, 2 ** num_of_bits,
                                          (num_of_parameters * 2))
        random_individual = ''.join([convert_to_graycode(num, num_of_bits)
                                     for num in random_individual])
        individuals += [random_individual]
    return individuals


def convert_to_parameters(individual: str, parameters: dict) -> list:
    """
    Convert the individual string to optimization parameters
    :param individual: Individual string
    :param parameters: parameter config
    :return: optimization parameters
    """
    # Splitting the individual string into optimization parameters
    individual = [individual[size: size + parameters['s']] for size in range(
        0, len(individual), parameters['s'])]
    # Calculating actual values of optimization parameters
    optimization_parameters = [[], []]
    for index in range(len(individual)):
        if index % 2 == 0:
            optimization_parameters[0] += [convert_gray_to_num(individual[
                                                                   index],
                                                               parameters[
                                                                   'bound_contraints'][
                                                                   0])]
        else:
            optimization_parameters[1] += [convert_gray_to_num(individual[
                                                                   index],
                                                               parameters[
                                                                   'bound_contraints'][
                                                                   1])]
    return optimization_parameters


def check_boundary_conditions(state: list) -> bool:
    """
    Check boundary conditions (Obstacle)
    :param state: Current state
    :return: Feasible (True) or Infeasible (False)
    """
    if state[0] <= -4 and state[1] > 3:
        return True
    if state[0] > -4 and state[0] < 4 and state[1] > -1:
        return True
    if state[0] >= 4 and state[1] > 3:
        return True
    return False


def find_next_state(state: list, control: np.ndarray, h: float) -> list:
    """
    Find the next state in the history by apply ODEs
    :param state: Current state
    :param control: Current controls
    :param h: time segment size
    :return: The next state
    """
    new_state = [0] * len(state)
    new_state[0] = state[0] + h * (state[3] * math.cos(state[2]))
    new_state[1] = state[1] + h * (state[3] * math.sin(state[2]))
    new_state[2] = state[2] + h * control[0]
    new_state[3] = state[3] + h * control[1]
    return new_state


def interpolate(optimization_parameters: np.ndarray, time_segments:
np.ndarray) -> np.ndarray:
    """
    Interpolate the given optimization parameters to high dimension of values 
    using CubicSpline Interpolation
    :param optimization_parameters: List of optimization parameters
    :param time_segments: Time segments from 0 to 10s
    :return: Control histories
    """
    time_interval = time_segments.max() / len(optimization_parameters[0])
    # Equally sepearted times for the optimization parameters
    control_times = []
    for count in range(0, len(optimization_parameters[0])):
        control_times.append(count * time_interval)

    function_gamma = CubicSpline(control_times, optimization_parameters[0],
                                 bc_type='natural')
    function_beta = CubicSpline(control_times, optimization_parameters[1],
                                 bc_type='natural')

    # Calulating the curves for both controls
    gamma_high = function_gamma(time_segments)
    beta_high = function_beta(time_segments)

    # Concatenating values of both controls
    control_histories = np.concatenate((gamma_high.reshape((gamma_high.size,
                                                            1)),
                                        beta_high.reshape((beta_high.size,
                                                           1))), axis=1)
    return control_histories


def calculate_cost_j(optimization_parameters: np.ndarray, parameters: dict) \
        -> (float, list, np.ndarray):
    """
    Calculate the state history and the Cost J with respect to expected
    :param optimization_parameters: Optimization parameters of individual
    :param parameters: Parameter config
    :return: Cost, state history, control history
    """
    time_segments = np.linspace(0, parameters['task_time'], parameters[
        'num_of_values'])
    states = [parameters['initial_state']]
    expected_final = parameters['final_state']
    h = time_segments[1] - time_segments[0]
    control_histories = interpolate(optimization_parameters, time_segments)
    # Calculating State histories using control histories
    for index in range(1, parameters['num_of_values']):
        state = find_next_state(states[index - 1], control_histories[index -
                                                                     1], h)
        if check_boundary_conditions(state):
            states += [state]
        else:
            return parameters['K'], None, None
    # Calculating Cost J
    cost_j = np.sqrt(np.sum(np.square(np.array(states[-1]) - np.array(
        expected_final))))
    return cost_j, states, control_histories


def calculate_fitness(individual: str, parameters: dict) -> (float, float):
    """
    Calulate fitness for the individual
    :param individual: individual string
    :param parameters: parameter config
    :return: fitness, cost_j
    """
    optimization_parameters = convert_to_parameters(individual, parameters)
    cost_j, states, controls = calculate_cost_j(np.array(
        optimization_parameters), parameters)
    return (1 / (cost_j + 1)), cost_j


def cross(parent1: str, parent2: str) -> (str, str):
    """
    Cross over the given individuals at random crossover point
    :param parent1: Parent Individual
    :param parent2: Parent Individual
    :return: Child Individual 1, Child Individual 2
    """
    crossover_point = random.randint(1, len(parent1))
    child1 = parent1[: crossover_point] + parent2[crossover_point:]
    child2 = parent2[: crossover_point] + parent1[crossover_point:]
    return child1, child2


def mutate(child: str, mutation_rate: float) -> str:
    """
    Mutate an individual if chances less than equal to mutation rate
    :param child: Individual
    :param mutation_rate: Mutation rate
    :return: Mutated Child Individual
    """
    mutated = ''
    for index in range(len(child)):
        chance = random.random()  # Random chance whether to mutate bit
        if chance <= mutation_rate:
            if child[index] == '0':
                mutated += '1'
            else:
                mutated += '0'
        else:
            mutated += child[index]
    return mutated


def operate(generation: list, mutation_rate: float) -> list:
    """
    Perform GA operations on the whole population
    :param generation: List of individuals
    :param mutation_rate: Mutation rate
    :return: New Generation of Individual
    """
    new_individuals = []
    parents = [member[0] for member in generation]
    weights = [member[1] for member in generation]
    for count in range((len(generation) - 2) // 2):
        # Selection
        parent1, parent2 = random.choices(parents, weights, k=2)
        # Cross-over
        child1, child2 = cross(parent1, parent2)
        # Mutation
        child1 = mutate(child1, mutation_rate)
        child2 = mutate(child2, mutation_rate)
        new_individuals += [child1] + [child2]
    # Elitism
    new_individuals += [parents[0]] + [parents[1]]
    return new_individuals


def genetic_algo_solver(parameters: dict) -> str:
    """
    This function simulates the genetic algorithm to optimize parallel parking
    :param parameters: paramter config
    :return: Approximate solution individual
    """
    gen_individuals = create_generation_zero(parameters['population'],
                                              parameters['s'], parameters[
                                                  'num_of_parameters'])
    for gen in range(parameters['max_generations']):
        generation = []
        # Calculating fitness for each individual
        for individual in gen_individuals:
            g, j = calculate_fitness(individual, parameters)
            generation += [[individual, g, j]]
        # Sorting populating by fitness
        generation.sort(key = lambda x: x[1], reverse=True)
        print("Generation " + str(gen) + " : J = " + str(generation[0][2]))
        # Convergence point if cost_j < cost_tolerance
        if generation[0][2] < parameters['cost_tolerance']:
            break;
        # The next generation of individuals
        gen_individuals = operate(generation, parameters['mutation_rate'])
    return generation[0][0]


def plot_variable(x, y, title: str, x_name, y_name) -> None:
    """
    Plot History graph
    :param x: x axis
    :param y: y axis
    :param title: Title of plot
    :param x_name: x axis name
    :param y_name: y axis name
    :return: None
    """
    plt.figure(figsize=(7, 7))
    plt.plot(x, y, 'b')
    plt.grid()
    plt.title(title)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.show()


def print_and_plot(solution: str, parameters: dict) -> None:
    """
    Print and plot pretty output
    :param solution: Solution (Fittest) individual from the last genreation
    :param parameters: Parameter config
    :return: None
    """
    optimization_parameters = convert_to_parameters(solution, parameters)
    cost_j, states, controls = calculate_cost_j(np.array(
        optimization_parameters), parameters)
    final = states[-1]
    print("\nFinal state values:")
    print("x_f = " + str(final[0]))
    print("y_f = " + str(final[1]))
    print("alpha_f = " + str(final[2]))
    print("v_f = " + str(final[3]))

    # Writing Optimization Parameters to Control file
    with open("controls.dat", "w") as output:
        for index in range(len(optimization_parameters[0])):
            output.write(str(optimization_parameters[0][index]) + "\n")
            output.write(str(optimization_parameters[1][index]) + "\n")

    # Solution state & control histories
    time_segments = np.linspace(0, parameters['task_time'], parameters[
        'num_of_values'])
    x = [state[0] for state in states]
    y = [state[1] for state in states]
    alpha = [state[2] for state in states]
    v = [state[3] for state in states]
    gamma = [control[0] for control in controls]
    beta = [control[1] for control in controls]

    # Plotting Solution Trajectory
    obstacle_x = [-10, -4, -4.0001, 3.9999, 4, 10]
    obstacle_y = [3, 3, -1, -1, 3, 3]
    plt.figure(figsize=(7, 7))
    plt.plot(obstacle_x, obstacle_y, 'b')
    plt.plot(x, y, 'g')
    plt.grid()
    plt.title("Solution Trajectory")
    plt.xlabel('x (ft)')
    plt.ylabel('y (ft)')
    plt.show()

    # Plotting all state histories and control histories
    plot_variable(time_segments, x, "State History for x", "Time (s)", "x (ft)")
    plot_variable(time_segments, y, "State History for y", "Time (s)", "y (ft)")
    plot_variable(time_segments, alpha, "State History for \u03B1", "Time (s)",
                  "\u03B1 (rad)")
    plot_variable(time_segments, v, "State History for v", "Time (s)",
                  "v (ft/s)")
    plot_variable(time_segments, gamma, "Control History for \u03B3", "Time ("
                                                                      "s)",
                  "\u03B3 (rad/s)")
    plot_variable(time_segments, beta, "Control History for \u03B2", "Time (s)",
                  "\u03B2 (ft/s\u00b2)")


def main():
    """
    The main function
    :return: None
    """
    parameters = {'task_time': 10,
    'num_of_parameters': 10,
    'num_of_values': 100,
    's': 7,
    'population': 200,
    'mutation_rate': 0.005,
    'K': 200.00,
    'max_generations': 1200,
    'cost_tolerance': 0.1,
    'bound_contraints': [[-0.524, 0.524], [-5, 5]],
    'initial_state': [0, 8, 0, 0],
    'final_state': [0, 0, 0, 0]}
    solution = genetic_algo_solver(parameters)
    print_and_plot(solution, parameters)


if __name__ == '__main__':
    main()  # Calling Main Function
