
import optuna
import numpy as np
import pandas as pd
import multiprocessing
import random
import array
import time
import os 

from copy import deepcopy
from scipy.spatial import distance
import matplotlib.pyplot as plt

from deap import base, creator, tools, algorithms
from deap.benchmarks.tools import hypervolume, igd

import Benchmark_Problem_Suites as ps


class NSGA_III_HYPERPARAMETER:
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

    def __init__(self, problem_name, problem, num_gen, pop_size, cross_prob, mut_prob, MP, crossover_distr, mutation_distr, indpbmut, verbose = True):
        self.PROBLEM = problem
        self.NBOJ = problem.n_obj
        self.NDIM = problem.n_var
        self.P = 12
        self.BOUND_L, self.BOUND_U = 0.0, 1.0
        self.NGEN = num_gen
        self.POP_SIZE = pop_size
        self.CXPB = cross_prob
        self.MUTPB = mut_prob
        self.mutation_distr = mutation_distr
        self.crossover_distr = crossover_distr
        self.indpbmut = indpbmut    
        self.MP = MP
        self.verbose = verbose
        self.val_bounds = ps.problem_bounds[problem_name]['val']
        self.std_bounds = ps.problem_bounds[problem_name]['std']
        self.hv_reference_point = np.array([1.0]*self.NBOJ)
    
    def _RUN(self): 
        """Run the NSGA-III loop until the termination criterion is met"""
        """Initialization"""
        # Initialize creator class
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0,)*self.NBOJ)
        creator.create("Individual", np.ndarray , fitness=creator.FitnessMulti)
        
        #print(f'Problem: {self.PROBLEM}')
        #print(f'---start NSGA-III Run for {self.NGEN} generations and a population of size {self.POP_SIZE} distributing work over {self.MP} cores ---')

        #Set up the toolbox for individuals and population
        toolbox = base.Toolbox()
        toolbox.register('attr_float', random.random)
        toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.attr_float, self.NDIM)
        toolbox.register('population', tools.initRepeat, list, toolbox.individual)
        
        #Set up the toolbox for reference points and evolutionary process
        ref_points = tools.uniform_reference_points(nobj=self.NBOJ, p=self.P)
        toolbox.register('evaluate', self.PROBLEM.evaluate)
        toolbox.register("select", tools.selNSGA3, ref_points=ref_points)
        toolbox.register('mate', tools.cxSimulatedBinaryBounded, low=self.BOUND_L, up=self.BOUND_U, eta=self.crossover_distr)
        toolbox.register('mutate', tools.mutPolynomialBounded, low=self.BOUND_L, up=self.BOUND_U, eta=self.mutation_distr, indpb=self.indpbmut)

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

        #Generate initial population
        pop = toolbox.population(n=self.POP_SIZE)

        #Evaluate the individuals of the initial population with an invalid fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        #calculate hypervolumen of initial population

        pareto_front = self.retrieve_pareto_front(population = pop)
        hv = self.calculate_hypervolume(pareto_front = pareto_front)

        #Compile statistics about the population
        record = stats.compile(pop)
        self.logbook.record(gen=0, evals=len(invalid_ind), **record)
        if self.verbose:
            print(self.logbook.stream)

        """Evolutionary process"""
        #Start the evolutionary process (measure total procesing time of evolution)
        for gen in range(1, self.NGEN+1):
            #Create offspring
            offspring = algorithms.varAnd(pop, toolbox, self.CXPB, self.MUTPB)

            #Evaluate the individuals of the offspring with an invalid fitness (measure evaluation time)
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            #Select the next population from parent and offspring
            pop = toolbox.select(pop + offspring, self.POP_SIZE)
            #Compile statistics about the new population
            record = stats.compile(pop)
            self.logbook.record(gen=gen, evals=len(invalid_ind), **record)
            if self.verbose:
                print(self.logbook.stream)

            #Calculate hypervolume of population
            pareto_front = self.retrieve_pareto_front(population=pop)
            hv = self.calculate_hypervolume(pareto_front=pareto_front)           

            """"Save results"""
            #Save generation to file
            if gen== self.NGEN:
                final_hv = hv

        # Close the multiprocessing pool if used
        if self.MP > 0:
            pool.close()
            pool.join()
        return final_hv


def objective(trial):
    crossover = trial.suggest_int('crossover', 0, 100)
    mutation = trial.suggest_int('mutation', 0, 100)
    indpbmut = trial.suggest_float('indpbmut', 0.0, 1.0)

    problem_name = 'dtlz1'
    if problem_name.startswith('DF'):
        problem = ps.problems_CEC[problem_name]
    else:
        problem = ps.problems_DEAP[problem_name]

    nsga = NSGA_III_HYPERPARAMETER(problem_name = problem_name,
                                      problem = problem, num_gen=50, 
                                      pop_size=20, 
                                      cross_prob=1.0, 
                                      mut_prob=1.0, 
                                      MP=12, 
                                      crossover_distr = crossover,
                                      mutation_distr = mutation,
                                      indpbmut = indpbmut,
                                      verbose=False)
    
    performance= nsga._RUN()

    return performance

if __name__ == '__main__':
    start_time = time.time()
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    print('Best trial:')
    trial = study.best_trial
    print('value:', trial.value)
    print("Params: ")
    for key, value in trial.params.items():
       print(f"    {key}: {value}")    

    print("--- %f minutes ---" % ((time.time() - start_time)/60))
    
    optuna.visualization.matplotlib.plot_contour(study, params=["crossover", "mutation", "indpbmut"])