"Benchmark problem suites "
"DEAP Framework"
"dtlz1-4"
"zdt1-6"
from deap import benchmarks
from pymop.factory import get_problem

problem_names = ['dtlz1', 'dtlz2', 'dtlz3', 'dtlz4', 'dtlz5', 'dtlz6',
                'dtlz7', 'zdt1', 'zdt2', 'zdt3', 'zdt4', 'zdt6']

problems1 = {'dtlz1': getattr(benchmarks, 'dtlz1'),
            'dtlz2': getattr(benchmarks, 'dtlz2'),
            'dtlz3': getattr(benchmarks, 'dtlz3'),
            'dtlz4': getattr(benchmarks, 'dtlz4'),
            'dtlz5': getattr(benchmarks, 'dtlz5'),
            'dtlz6': getattr(benchmarks, 'dtlz6'),
            'dtlz7': getattr(benchmarks, 'dtlz7'),
            'zdt1': getattr(benchmarks, 'zdt1'),
            'zdt2': getattr(benchmarks, 'zdt2'),
            'zdt3': getattr(benchmarks, 'zdt3'),
            'zdt4': getattr(benchmarks, 'zdt4'),
            'zdt6': getattr(benchmarks, 'zdt6')}

problems = {'dtlz1': get_problem('dtlz1', n_var=7, n_obj=3),
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





"IEEEE CEC framework"






"BBOB framework"
