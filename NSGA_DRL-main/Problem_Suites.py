#from pymoo.factory import get_problem
from pymop.factory import get_problem
# Set problems and problem dimensions
problem_names = ['dtlz2', 'dascmop7.1', 'dascmop7.2', 'dascmop7.3', 'dascmop7.4', 'dascmop7.5', 'dascmop7.6', 
                 'dascmop7.7', 'dascmop7.8', 'dascmop7.9', 'dascmop7.10', 'dascmop7.11', 'dascmop7.12', 
                 'dascmop7.13', 'dascmop7.14', 'dascmop7.15', 'dascmop7.16', 'mw4', 'mw8', 'mw14']

problems = {'dtlz2': get_problem('dtlz2', n_var=96, n_obj=3)}
# , 
#             'dascmop7.1': get_problem('dascmop7', 1), 
#             'dascmop7.2': get_problem('dascmop7', 2),
#             'dascmop7.3': get_problem('dascmop7', 3),
#             'dascmop7.4': get_problem('dascmop7', 4),
#             'dascmop7.5': get_problem('dascmop7', 5),
#             'dascmop7.6': get_problem('dascmop7', 6),
#             'dascmop7.7': get_problem('dascmop7', 7),
#             'dascmop7.8': get_problem('dascmop7', 8),
#             'dascmop7.9': get_problem('dascmop7', 9),
#             'dascmop7.10': get_problem('dascmop7', 10),
#             'dascmop7.11': get_problem('dascmop7', 11),
#             'dascmop7.12': get_problem('dascmop7', 12),
#             'dascmop7.13': get_problem('dascmop7', 13),
#             'dascmop7.14': get_problem('dascmop7', 14),
#             'dascmop7.15': get_problem('dascmop7', 15),
#             'dascmop7.16': get_problem('dascmop7', 16),
#             'mw4': get_problem("mw4"), 
#             'mw8': get_problem("mw8"), 
#             'mw14': get_problem("mw14")}

problem_dims = {'dtlz2': 96, 'dascmop7.1': 30, 'dascmop7.2': 30, 'dascmop7.3': 30, 'dascmop7.4': 30, 
                'dascmop7.5': 30, 'dascmop7.6': 30, 'dascmop7.7': 30, 'dascmop7.8': 30, 'dascmop7.9': 30, 
                'dascmop7.10': 30, 'dascmop7.11': 30, 'dascmop7.12': 30, 'dascmop7.13': 30, 'dascmop7.14': 30,
                'dascmop7.15': 30, 'dascmop7.16': 30, 'mw4': 15, 'mw8': 15, 'mw14': 15}

problem_bounds = {'dtlz2': {'val': [(0.0, 4.0), (0.0, 5.0), (0.0, 6.0)], 'std': [(0.0, 3.0), (0.0, 3.0), (0.0, 3.0)]}, 
                  'dascmop7.1': {'val': [(3.0, 32.0), (4.0, 32.0), (3.0, 32.0)], 'std': [(0.0, 4.0), (0.0, 4.0), (0.0, 4.0)]}, 
                  'dascmop7.2': {'val': [(3.0, 31.0), (3.0, 31.0), (3.0, 31.0)], 'std': [(0.0, 4.0), (0.0, 4.0), (0.0, 4.0)]}, 
                  'dascmop7.3': {'val': [(3.5, 31.0), (3.5, 31.0), (3.5, 31.0)], 'std': [(0.0, 4.0), (0.0, 4.0), (0.0, 4.0)]}, 
                  'dascmop7.4': {'val': [(2.5, 32.0), (2.5, 32.0), (2.5, 32.0)], 'std': [(0.0, 4.5), (0.0, 4.5), (0.0, 4.5)]}, 
                  'dascmop7.5': {'val': [(3.0, 31.0), (3.0, 31.0), (3.0, 31.0)], 'std': [(0.0, 5.0), (0.0, 5.0), (0.0, 5.0)]}, 
                  'dascmop7.6': {'val': [(3.0, 31.0), (3.0, 31.0), (3.0, 31.0)], 'std': [(0.0, 3.0), (0.0, 3.5), (0.0, 3.5)]}, 
                  'dascmop7.7': {'val': [(4.0, 31.0), (4.0, 31.0), (4.0, 31.0)], 'std': [(0.0, 3.5), (0.0, 3.5), (0.0, 3.0)]}, 
                  'dascmop7.8': {'val': [(4.0, 30.0), (3.0, 30.0), (4.0, 30.0)], 'std': [(0.0, 4.0), (0.0, 4.0), (0.0, 4.5)]}, 
                  'dascmop7.9': {'val': [(3.5, 30.0), (3.5, 30.0), (3.5, 30.0)], 'std': [(0.0, 4.0), (0.0, 4.0), (0.0, 4.0)]}, 
                  'dascmop7.10': {'val': [(4.0, 32.0), (4.0, 32.0), (4.0, 32.0)], 'std': [(0.0, 4.5), (0.0, 4.5), (0.0, 4.5)]}, 
                  'dascmop7.11': {'val': [(2.5, 30.0), (2.5, 30.0), (3.0, 30.0)], 'std': [(0.0, 4.5), (0.0, 4.5), (0.0, 4.5)]}, 
                  'dascmop7.12': {'val': [(3.0, 31.0), (3.0, 31.0), (3.0, 31.0)], 'std': [(0.0, 4.0), (0.0, 4.0), (0.0, 4.0)]}, 
                  'dascmop7.13': {'val': [(3.5, 32.0), (3.0, 32.0), (3.0, 32.0)], 'std': [(0.0, 4.0), (0.0, 4.0), (0.0, 4.0)]}, 
                  'dascmop7.14': {'val': [(3.0, 30.0), (3.0, 30.0), (3.0, 30.0)], 'std': [(0.0, 3.5), (0.0, 3.0), (0.0, 3.0)]},
                  'dascmop7.15': {'val': [(2.5, 32.0), (3.0, 32.0), (3.5, 32.0)], 'std': [(0.0, 4.0), (0.0, 4.0), (0.0, 4.0)]}, 
                  'dascmop7.16': {'val': [(3.0, 32.0), (3.0, 32.0), (4.0, 32.0)], 'std': [(0.0, 4.0), (0.0, 4.0), (0.0, 4.0)]}, 
                  'mw4': {'val': [(0.0, 6.0), (0.0, 4.0), (0.0, 7.0)], 'std': [(1.0, 4.0), (1.0, 3.5), (1.0, 4.5)]}, 
                  'mw8': {'val': [(0.0, 7.0), (0.0, 4.0), (0.0, 8.0)], 'std': [(0.0, 5.5), (0.0, 4.0), (0.0, 5.0)]}, 
                  'mw14': {'val': [(0.0, 1.0), (0.0, 1.0), (2.5, 29.0)], 'std': [(0.0, 1.0), (0.0, 1.0), (0.0, 10.0)]}}
