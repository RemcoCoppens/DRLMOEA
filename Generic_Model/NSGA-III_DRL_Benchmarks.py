import numpy as np
import multiprocessing
import random
import os
import array
from copy import deepcopy
from scipy.spatial import distance
import matplotlib.pyplot as plt

from deap import base, creator, tools
from deap.benchmarks.tools import hypervolume

from Agent_Classes.DuelingDQN_Agent import Agent
import Benchmark_Problems_Suites as ps

from tqdm import tqdm
import pickle

class NSGA_III_Learning:
    """
    Description: Class for the NSGA-III algorithm
    Input:  - problem_name: The name of the problem to be solved
            - problem: The problem to be solved
            - num_gen: The number of generations
            - pop_size: The number of individuals in the population
            - cross_prob: Cross-over probability
            - mut_prob: Mutation probability
            - MP: if set to a value higher than 0, the given number of cores will be utilized
            - verbose: If True the performance of every population will be printed
    """

    def __init__(self, problem_name, problem, num_gen, pop_size, cross_prob, mut_prob, MP, verbose = True, learn_agent = True, load_agent = None):
        #General
        self.PROBLEM = problem
        self.NBOJ = problem.n_obj
        self.NDIM = problem.n_var
        self.P = 12
        self.BOUND_L, self.BOUND_U = 0.0, 1.0
        self.NGEN = num_gen
        self.POP_SIZE = pop_size
        self.CXPB = cross_prob
        self.MUTPB = mut_prob
        self.MP = MP
        self.verbose = verbose
        self.directory = self.check_results_directory()
        self.val_bounds = ps.problem_bounds[problem_name]['val']
        self.std_bounds = ps.problem_bounds[problem_name]['std']
        

        #Agent
        self.agent = Agent(lr  = 1e-4,
                           gamma = 0.99,
                           actions = [[10.0, 50.0, 100.0],
                           [0.01, 0.05, 0.10, 0.15, 0.20]],
                           batch_size = 32,
                           input_size = 7
                           )
        if load_agent != None:
            self.agent.load_model(load_agent)
            self.agent.epsilon = 0

        self.stagnation_counter = 0
        self.hv_reference_point = np.array([1.0]*self.NBOJ)
        self.hv_trace = []
        self.hv_dict = {}
        self.policy_dict = {}

    
    def check_results_directory(self):
        """ Check if there are already results in the NSGA-II file, if so ask for overwrite and if requested create new file """
        if len(os.listdir("Results/NSGA-III_DRL")) > 0:
            selection = input("Existing result files found, do you want to overwrite? [y/n]")
            if selection == 'y' or selection == 'yes' or selection == 'Y' or selection == 'YES':
                return 'NSGA-III'
            elif selection == 'n' or selection == 'no' or selection == 'N' or selection == 'NO':
                folder_extension = input("Insert Folder Extension")
                os.mkdir(path=f'Results/NSGA-III_{folder_extension}') 
                return f'NSGA-III_{folder_extension}'
        else:
            return 'NSGA-III'   
    
    
    def save_generation(self, gen, population, avg_eval_time, gen_time, pareto, hv, final_pop=None, alg_exec_time=None):
        """ Save performance of generation to file """
        # Summarize performance in dictionary object and save to file
        performance_dict = {}
        performance_dict['gen'] = [gen]
        performance_dict['pareto_front'] = [pareto]
        performance_dict['hypervolume'] = hv
        performance_dict['avg_eval_time'] = avg_eval_time
        performance_dict['gen_time'] = gen_time
        performance_dict['avg_obj'] = [self.logbook[gen]['avg']]
        performance_dict['max_obj'] = [self.logbook[gen]['max']]
        performance_dict['min_obj'] = [self.logbook[gen]['min']]
        performance_dict['std_obj'] = [self.logbook[gen]['std']]
        performance_dict['population'] = [[list(np.array(indiv.fitness.values)) for indiv in population]]
        performance_dict['algorithm_execution_time'] = None
        if final_pop != None and alg_exec_time != None:
            performance_dict['algorithm_execution_time'] = alg_exec_time

        return performance_dict
    
    def save_run_to_file(self, performance, run, problem_name):
        """ Save performance of run to file """
        file = open(f"Results/{f'{self.directory}/Problem_{problem_name}_Run_{run}'}.pkl", "wb")
        pickle.dump(performance, file)
        file.close()
        

    def normalize(self, val, LB, UB, clip=True):
        """ Apply (bounded) normalization on the given value using the given bounds (LB, UB) """
        if clip:
            return min(max((val - LB)/(UB - LB), 0.0), 1.0)
        else:
            return (val - LB)/(UB - LB)
    

    def retrieve_pareto_front(self, population):
        """ Calculate and return the pareto optimal set """
        pareto_front = tools.sortNondominated(individuals=population, 
                                              k=self.POP_SIZE, 
                                              first_front_only=True)[0]
        
        indivs = [list(np.array(indiv.fitness.values)) for indiv in population]
        
        return [np.array(indiv.fitness.values) for indiv in pareto_front]
    
    def calculate_hypervolume(self, pareto_front) -> float:
        """ Normalize values and calculate the hypervolume indicator of the current pareto front """
        # Retrieve and calculate pareto front figures
        normalized_pareto_set = np.array([tuple([self.normalize(val=obj_v[i], 
                                                                LB=self.val_bounds[i][0], 
                                                                UB=self.val_bounds[i][1]) for i in range(self.NBOJ)]) for obj_v in pareto_front])        

        hv = hypervolume(normalized_pareto_set, self.hv_reference_point)
        #self.hv_tracking.append(hv)
        return hv

    def create_offspring(self, population, operator, use_agent = True) -> list:
        """Create offspring from the current population (retrieved from DEAP varAnd module)"""
        offspring = [deepcode(indiv) for indiv in population]


    def call_agent (self, gen, hv, pareto_size, state = [], action = None) -> tuple:
        """ Call the agent to retrieve the next action """
        if action == None:
            action = self.agent.choose_action(state)
        else:
            self.agent.epsilon = 0
        return action
    
    def _RUN(self, use_agent = True):
        """Run the NSGA-III loop until the termination criterion is met"""
        """Initialization"""
        # Initialize creator class
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0,)*self.NBOJ)
        creator.create("Individual", np.ndarray , fitness=creator.FitnessMulti)
        
        print(f'Problem: {self.PROBLEM}')
        print(f'---start NSGA-III Run for {self.NGEN} generations and a population of size {self.POP_SIZE} distributing work over {self.MP} cores ---')
        #Start time
        Start_timer = time.time()

        #Initialize trace lists of the agents interactions
        states_trace = []
        actions_trace = []
        rewards_trace = []
        reward_idx_trace = []

        #Set up the toolbox for individuals and population
        toolbox = base.Toolbox()
        toolbox.register('attr_float', random.random)
        toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.attr_float, self.NDIM)
        toolbox.register('population', tools.initRepeat, list, toolbox.individual)
        
        #Set up the toolbox for reference points and evolutionary process
        ref_points = tools.uniform_reference_points(nobj=self.NBOJ, p=self.P)
        toolbox.register('evaluate', self.PROBLEM.evaluate)
        toolbox.register("select", tools.selNSGA3, ref_points=ref_points)
        toolbox.register('mate', tools.cxSimulatedBinaryBounded, low=self.BOUND_L, up=self.BOUND_U, eta=30.0)
        toolbox.register('mutate', tools.mutPolynomialBounded, low=self.BOUND_L, up=self.BOUND_U, eta=20.0, indpb=1.0/self.NDIM)

        #parallel 
        if self.MP > 0:
            pool = multiprocessing.Pool(processes = self.MP) 
            toolbox.register('map', pool.map)
    
        #initialize statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register('avg', np.mean, axis=0)
        stats.register('min', np.min, axis=0)
        stats.register('max', np.max, axis=0)
        stats.register('std', np.std, axis=0)

        #Initialize logbook
        self.logbook = tools.Logbook()
        self.logbook.header = ['gen', 'evals'] + stats.fields