
from deap import benchmarks
from pymoo.problems import get_problem
import numpy as np
import random

"Benchmark problem suites "
"DEAP Framework"
"dtlz1-4"
"zdt1-6"
problem_names_DEAP = ['dtlz1', 'dtlz2', 'dtlz3', 'dtlz4', 'dtlz5', 'dtlz6',
                'dtlz7', 'zdt1', 'zdt2', 'zdt3', 'zdt4', 'zdt6']

problems_DEAP = {'dtlz1': get_problem('dtlz1', n_var=7, n_obj=3),
             'dtlz2': get_problem('dtlz2', n_var=12, n_obj=3),
            'dtlz3': get_problem('dtlz3', n_var=12, n_obj=3),
            'dtlz4': get_problem('dtlz4', n_var=12, n_obj=3),
            'dtlz5': get_problem('dtlz5', n_var=12, n_obj=3),
            'dtlz6': get_problem('dtlz6', n_var=12, n_obj=3),
            'dtlz7': get_problem('dtlz7', n_var=22, n_obj=3),
            'zdt1': get_problem('zdt1'),
            'zdt2': get_problem('zdt2'),
            'zdt3': get_problem('zdt3'),
            'zdt4': get_problem('zdt4'),
            'zdt6': get_problem('zdt6')}


"IEEEE CEC framework 2018 on Dynamic Multiobjective Optimisation"
"DF1-14"
"Benchmark Problems for CEC 2018 competition on Dynamic Multiobjective Optimisation"

problem_names_CEC = ['DF1', 'DF2', 'DF3', 'DF4', 'DF5', 'DF6', 'DF7', 'DF8', 'DF9', 'DF10' 
                     , 'DF11', 'DF12', 'DF13', 'DF14']

problems_CEC = {'DF1': get_problem('df1', n_var=10),
                'DF2': get_problem('df2', n_var=10),
                'DF3': get_problem('df3', n_var=10),
                'DF4': get_problem('df4', n_var=10),
                'DF5': get_problem('df5', n_var=10),
                'DF6': get_problem('df6', n_var=10),
                'DF7': get_problem('df7', n_var=10),
                'DF8': get_problem('df8', n_var=10),
                'DF9': get_problem('df9', n_var=10),
                'DF10': get_problem('df10', n_var=10),
                'DF11': get_problem('df11', n_var=10),
                'DF12': get_problem('df12', n_var=10),
                'DF13': get_problem('df13', n_var=10),
                'DF14': get_problem('df14', n_var=10)}



"BBOB framework"




"""Problem Bounds"""
def get_bounds(problem:str) -> dict:
    if problem.startswith('DF'):
        problem = problems_CEC[problem]
    else:
        problem = problems_DEAP[problem]
    var = problem.n_var
    individual = [random.random() for _ in range(var)]
    print(individual)

    evals = [problem.evaluate(np.array([random.random() for _ in range(var)])) for _ in range(100000)]
    arr = np.array(evals)
    eval_dict = {
        "min":arr.min(axis=0),
        "max":arr.max(axis=0),
        "mean":arr.mean(axis=0),
        "std":arr.std(axis=0)
    }
    return print(eval_dict)



problem_bounds = {'dtlz1': {'val': [(0.0, 450), (0, 460.0), (0.0, 490.0)], 'mean': [67.8, 68.7, 135.7], 'std': [65.4, 64.3, 90.8]},
             'dtlz2': {'val': [(0.0, 2.61), (0.0, 2.63), (0.0, 2.71)], 'mean': [0.75, 0.74, 1.18], 'std': [0.55, 0.55, 0.59]},
            'dtlz3': {'val': [(0.0, 1766.0), (0.0, 1802.0), (0.0, 1847.0)], 'mean': [440.3, 441.1, 690.2], 'std': [343.5, 339.5, 371.0]},
            'dtlz4': {'val': [(0.0, 3.12), (0.0, 2.53), (0.0, 2.48)], 'mean': [1.8, 0.03, 0.03], 'std': [0.28, 0.18, 0.17]},
            'dtlz5': {'val': [(0.0, 2.46), (0.0, 2.46), (0.0, 2.96)], 'mean': [0.81, 0.81, 1.17], 'std': [0.45, 0.45, 0.59]},
            'dtlz6': {'val': [(0.0, 10.63), (0.0, 10.75), (0.0, 10.77)], 'mean': [4.2, 4.2, 6.4], 'std': [2.84, 2.84, 3.12]},
            'dtlz7': {'val': [(0.0, 1.0), (0.0, 1.0), (10.7, 26.18)], 'mean': [0.5, 0.5, 18.3], 'std': [0.29, 0.29, 1.93]},
            'zdt1': {'val': [(0.0, 1.0), (1.82, 6.76)], 'mean': [0.5, 3.9], 'std': [0.29, 0.69]},
            'zdt2': {'val': [(0.0, 1.0), (3.16, 7.38)], 'mean': [0.5, 5.4], 'std': [0.29, 0.49]},
            'zdt3': {'val': [(0.0, 1.0), (1.03, 6.82)], 'mean': [0.5, 4.0], 'std': [0.29, 0.77]},
            'zdt4': {'val': [(0.0, 1.0), (12,47, 163.69)], 'mean': [0.5, 87.5], 'std': [0.29, 20.57]},
            'zdt6': {'val': [(0.28, 1.0), (5.79, 9.57)], 'mean': [0.9, 8.5], 'std': [0.14, 0,38]},
            'DF1': {'val': [(0.0, 1.0), (0.0, 7.8)], 'mean': [0.5, 3.68], 'std': [0.29, 0.94]},
            'DF2': {'val': [(0.0, 1.0), (0.0, 7.1)], 'mean': [0.5, 2.68], 'std': [0.29, 0.88]},
            'DF3': {'val': [(0.0, 1.0), (0.0, 7.85)], 'mean': [0.5, 2.38], 'std': [0.29, 1.05]},
            'DF4': {'val': [(0.0, 7.1), (1.52, 22.04)], 'mean': [1.59, 7.45], 'std': [1.28, 2.73]},
            'DF5': {'val': [(0.0, 7.1), (0.0, 7.7)], 'mean': [2.0, 2.0], 'std': [1.26, 1.26]},
            'DF6': {'val': [(9.21, 168.7), (4.72, 165.8)], 'mean': [77.6, 77.3], 'std': [21.32, 21.27]},
            'DF7': {'val': [(1.53, 1108516), (0.0, 7.43)], 'mean': [67.60, 2.00], 'std': [4910.83, 1.25]},
            'DF8': {'val': [(0.0, 7.8), (0.0, 7.3)], 'mean': [2.07, 0.98], 'std': [1.29, 1.45]},
            'DF9': {'val': [(0.0, 13.03), (0.0, 9.23)], 'mean': [1.98, 1.79], 'std': [1.16, 1.41]},
            'DF10': {'val': [(0.0, 27.14), (0.0, 23.37), (0.0, 19.15)], 'mean': [2.75, 1.02, 1.06], 'std': [4.31, 2.35, 2.52]},
            'DF11': {'val': [(0.0, 7.16), (0.0, 7.27), (0.0, 7.27)], 'mean': [2.34, 1.49, 1.49], 'std': [1.28, 1.15, 1.15]},
            'DF12': {'val': [(0.0,7.27), (0.0, 7.27), (0.0, 7.16)], 'mean': [1.49, 1.49, 2.34], 'std': [1.15, 1.15, 1.28]},
            'DF13': {'val': [(0.0, 7.58), (0.0, 7.58), (0.0, 9.7)], 'mean': [1.83, 1.83, 3.61], 'std': [1.41, 1.40, 1.81]},
            'DF14': {'val': [(0.0, 3.8), (0.0, 3.5), (0.0, 3.5)], 'mean': [1.83, 0.92, 0.92], 'std': [0.42, 0.60, 0.57]}}

