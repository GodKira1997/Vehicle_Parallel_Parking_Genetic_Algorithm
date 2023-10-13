Execution Instructions:
1. Python Libraries needed:
	1. numpy
	2. scipy
	3. matplotlib
2. Run parking.py

For custom configuration, change the values Parameters dictionary in the Main Function on line 366, for eg.
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