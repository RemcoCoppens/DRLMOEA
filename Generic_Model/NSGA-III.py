import numpy as np
import multiprocessing
import random
import array
import time
import os 

from copy import deepcopy
from scipy.spatial import distance
import matplotlib.pyplot as plt

from deap import base, creator, tools
from deap.benchmarks.tools import hypervolume

import Benchmark_Problem_Suites as ps

from tqdm import tqdm
import pickle




class NSGA_III:

    def __init__(self, problem, num_gen, pop_size, cross_prob, mut_prob, MP, verbose = True):
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

    def check_results_directory(self):
        """ Check if there are already results in the NSGA-II file, if so ask for overwrite and if requested create new file """
        if len(os.listdir("Results/NSGA-III")) > 0:
            selection = input("Existing files found, do you want to overwrite? [y/n]")
            if selection == 'y' or selection == 'yes' or selection == 'Y' or selection == 'YES':
                return 'NSGA-III'
            elif selection == 'n' or selection == 'no' or selection == 'N' or selection == 'NO':
                folder_extension = input("Insert Folder Extension")
                os.mkdir(path=f'Results/NSGA-III_{folder_extension}') 
                return f'NSGA-III_{folder_extension}'
        else:
            return 'NSGA-III'
    
    def save_to_file(data, filename):
        """ Create a file using the given file name and save the given data in this pickled file """
        file = open(f"Results/{filename}.pkl", "wb")
        pickle.dump(data, file)
        file.close()
        
    def save_generation_to_file(self, gen, population, avg_eval_time, gen_time, final_pop=None, alg_exec_time=None):
        """ Save performance of generation to file """
        # Summarize performance in dictionary object and save to file
        performance_dict = {}
        performance_dict['pareto_front'], performance_dict['pareto_front_indivs'] = self.retrieve_pareto_front(population)
        performance_dict['avg_eval_time'] = avg_eval_time
        performance_dict['gen_time'] = gen_time
        performance_dict['avg_obj'] = self.logbook[gen]['avg']
        performance_dict['max_obj'] = self.logbook[gen]['max']
        performance_dict['min_obj'] = self.logbook[gen]['min']
        performance_dict['std_obj'] = self.logbook[gen]['std']
        performance_dict['population'] = [list(np.array(indiv.fitness.values)) for indiv in population]
        if final_pop != None and alg_exec_time != None:
            performance_dict['algorithm_execution_time'] = alg_exec_time
        
        
        self.save_to_file(performance_dict, f'{self.directory}/Gen_{gen}')

    def retrieve_pareto_front(self, population):
        """ Calculate and return the pareto optimal set """
        pareto_front = tools.sortNondominated(individuals=population, 
                                              k=self.POP_SIZE, 
                                              first_front_only=True)[0]
        indivs = [list(np.array(indiv.fitness.values)) for indiv in population]
        return [np.array(indiv.fitness.values) for indiv in pareto_front], indivs
     

    def _RUN(self): 
        """Run the NSGA-III loop until the termination criterion is met"""
        # Initialize creator class
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0,)*self.NBOJ)
        creator.create("Individual", list, typecode="d", fitness=creator.FitnessMulti)
        
        print(f'Problem: {self.PROBLEM}')
        print(f'---start NSGA-III Run for {self.NGEN} generations and a population of size {self.POP_SIZE} distributing work over {self.MP} cores ---')
        #Start time
        Start_timer = time.time()

        #Set up the toolbox for individuals and population
        toolbox = base.Toolbox()
        toolbox.register('attr_float', random.random)
        toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.attr_float, self.NDIM)
        toolbox.register('population', tools.initRepeat, list, toolbox.individual)
        
        #Set up the toolboc for reference points and evolutionary process
        ref_points = tools.uniform_reference_points(nobj=self.NBOJ, p=self.P)
        toolbox.register('evaluate', self.PROBLEM.evaluate)
        toolbox.register('crossover', tools.cxSimulatedBinaryBounded, low=self.BOUND_L, up=self.BOUND_U, eta=30.0)
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

        #Generate initial population
        pop = toolbox.population(n=self.POP_SIZE)
        
        #Evaluate the individuals of the initial population with an invalid fitness (measure evaluation time)
        eval_start = time.time()
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        avg_eval_time = (time.time() - eval_start) / len(invalid_ind)

        #Compile statistics about the population
        record = stats.compile(pop)
        self.logbook.record(gen=0, evals=len(invalid_ind), **record)
        if self.verbose:
            print(self.logbook.stream)
        
        #Save generation to file
        self.save_generation_to_file(gen=0, population=pop, avg_eval_time=avg_eval_time, gen_time=0)

        #Start the evolutionary process (measure total procesing time of evolution)
        for gen in range(1, self.NGEN+1):
            gen_start = time.time()

            #Create offspring
            offspring = algorithms.VarAnd(pop, toolbox, self.CXPB, self.MUTPB)

            #Evaluate the individuals of the offspring with an invalid fitness (measure evaluation time)
            eval_start = time.time()
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            avg_eval_time = (time.time() - eval_start) / len(invalid_ind)

            #Select the next population from parent and offspring
            pop = toolbox.select(pop + offspring, self.POP_SIZE)

            #Compile statistics about the new population
            record = stats.compile(pop)
            self.logbook.record(gen=gen, evals=len(invalid_ind), **record)
            if self.verbose:
                print(self.logbook.stream)

            gen_time = time.time() - gen_start

            #Save generation to file

            if gen!= self.NGEN:
                self.save_generation_to_file(gen=gen, population=pop, avg_eval_time=avg_eval_time, gen_time=gen_time)
            #Final generation
            else:
                algorithm_execution_time = time.time() - algorithm_start
                self.save_generation_to_file(gen=gen, population=pop, avg_eval_time=avg_eval_time, gen_time=gen_time,
                                             final_pop=[list(self.eval.transform_individual(indiv)) for indiv in pop],
                                             alg_exec_time=algorithm_execution_time) 
        # Close the multiprocessing pool if used
        if self.MP > 0:
            pool.close()
            pool.join()

if __name__ == '__main__':
    problem_name = 'dtlz1'
    problem = ps.problems[problem_name]


    nsga = NSGA_III(problem = problem, num_gen=5, pop_size=20, cross_prob=1.0, mut_prob=1.0, MP=12, verbose=True)
    
    nsga._RUN()
    