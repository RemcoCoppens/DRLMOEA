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

from tqdm import tqdm
import pickle




class NSGA_III:
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

    def __init__(self, problem_name, problem, num_gen, pop_size, cross_prob, mut_prob, MP, verbose = True, save = True):
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
        self.save = save
        #if self.save: 
        #    self.directory = self.check_results_directory()
        self.directory = 'NSGA-III_100_runs'
        self.val_bounds = ps.problem_bounds[problem_name]['val']
        self.std_bounds = ps.problem_bounds[problem_name]['std']
        self.hv_reference_point = np.array([1.0]*self.NBOJ)
        self.hv_absolute_reference_point = np.array([0.0]*self.NBOJ)


    def check_results_directory(self):
        """ Check if there are already results in the NSGA-III file, if so ask for overwrite and if requested create new file """
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
        
    def save_generation(self, gen, population, avg_eval_time, gen_time, pareto, hv, abs_hv, final_pop=None, alg_exec_time=None):
        """ Save performance of generation to file """
        # Summarize performance in dictionary object and save to file
        performance_dict = {}
        performance_dict['gen'] = [gen]
        performance_dict['pareto_front'] = [pareto]
        performance_dict['hypervolume'] = hv
        performance_dict['absolute_hypervolume'] = abs_hv
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
        if run == 1:
            # Create new file
            file = open(f"Results/{f'{self.directory}/Problem_{problem_name}_POP_size_{self.POP_SIZE}'}.pkl", "wb")
            pickle.dump(performance, file)
            file.close()
        else:
            with open(f"Results/{f'{self.directory}/Problem_{problem_name}_POP_size_{self.POP_SIZE}'}.pkl", "ab") as input_file:
                pickle.dump(performance, input_file)
            input_file.close()
        

    def normalize(self, val, LB, UB, clip=True):
        """ Apply (bounded) normalization on the given value using the given bounds (LB, UB) """
        if clip:
            return min(max((val - LB)/(UB - LB), 0.0), 1.0)
        else:
            return (val - LB)/(UB - LB)

    def denormalize(self, val):
        """ Apply denormalization on the given value using the given bounds (LB, UB) """
        return val


    def retrieve_pareto_front(self, population):
        """ Calculate and return the pareto optimal set """
        pareto_front = tools.sortNondominated(individuals=population, 
                                              k=self.POP_SIZE, 
                                              first_front_only=True)[0]
        
        indivs = [list(np.array(indiv.fitness.values)) for indiv in population]
       
        return [np.array(indiv.fitness.values) for indiv in pareto_front]
    
    def calculate_hypervolume(self, pareto_front) -> float:
        """ Normalize values and calculate the hypervolume indicator of the current pareto front """
        # Retrieve and calculate normalised pareto front set
        
        
        normalized_pareto_set = np.array([tuple([self.normalize(val=obj_v[i], 
                                                                LB=self.val_bounds[i][0], 
                                                                UB=self.val_bounds[i][1]) for i in range(self.NBOJ)]) for obj_v in pareto_front])        
        
        pareto_front = np.array([tuple(self.denormalize(val=obj_v[i]) for i in range(self.NBOJ)) for obj_v in pareto_front])

        hv = hypervolume(normalized_pareto_set, self.hv_reference_point)

        abs_hv = hypervolume(pareto_front*-1, self.hv_absolute_reference_point)
        #self.hv_tracking.append(hv)
        return hv, abs_hv


    def _RUN(self): 
        """Run the NSGA-III loop until the termination criterion is met"""
        """Initialization"""
        # Initialize creator class
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0,)*self.NBOJ)
        creator.create("Individual", np.ndarray , fitness=creator.FitnessMulti)
        
        print(f'Problem: {self.PROBLEM}')
        print(f'---start NSGA-III Run for {self.NGEN} generations and a population of size {self.POP_SIZE} distributing work over {self.MP} cores ---')
        #Start time
        Start_timer = time.time()

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

        #Generate initial population
        pop = toolbox.population(n=self.POP_SIZE)
       
        #Evaluate the individuals of the initial population with an invalid fitness (measure evaluation time)
        eval_start = time.time()
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        #print(invalid_ind)
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        avg_eval_time = (time.time() - eval_start) / len(invalid_ind)
        
        #Compile statistics about the population
        record = stats.compile(pop)
        self.logbook.record(gen=0, evals=len(invalid_ind), **record)
        if self.verbose:
            print(self.logbook.stream)
        
        #Calculate hypervolume of initial population
        pareto_front = self.retrieve_pareto_front(population=pop)
        hv, abs_hv = self.calculate_hypervolume(pareto_front=pareto_front)
        

        #Save generation to file
        save_gen = self.save_generation(gen=0, population=pop, avg_eval_time=avg_eval_time, gen_time=0, pareto = pareto_front, hv = hv, abs_hv= abs_hv)
        df = pd.DataFrame(save_gen)

        """Evolutionary process"""
        #Start the evolutionary process (measure total procesing time of evolution)
        for gen in range(1, self.NGEN+1):
            gen_start = time.time()

            #Create offspring
            offspring = algorithms.varAnd(pop, toolbox, self.CXPB, self.MUTPB)
        

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

            #Calculate hypervolume of population
            pareto_front = self.retrieve_pareto_front(population=pop)
            hv, abs_hv = self.calculate_hypervolume(pareto_front=pareto_front)

            """"Save results"""
            #Save generation to file
            if gen!= self.NGEN:
                save_gen = self.save_generation(gen=gen, population=pop, avg_eval_time=avg_eval_time, gen_time=gen_time, pareto = pareto_front, hv = hv, abs_hv= abs_hv)
                df = pd.concat([df, pd.DataFrame(save_gen)], ignore_index=True)
            
            #Final generation
            else:
                print(pareto_front)
                algorithm_execution_time = time.time() - Start_timer
                save_gen = self.save_generation(gen=gen, population=pop, avg_eval_time=avg_eval_time, gen_time=gen_time, pareto = pareto_front, hv = hv, abs_hv= abs_hv,
                                             final_pop=1,
                                             alg_exec_time=algorithm_execution_time) 
                df = pd.concat([df, pd.DataFrame(save_gen)], ignore_index=True)
                display(df)
        # Close the multiprocessing pool if used
        if self.MP > 0:
            pool.close()
            pool.join()
        return df

    def multiple_runs(self, problem_name, nr_of_runes, progressbar = False):
        """ Run the NSGA-III algorithm multiple times """
        for idx in tqdm(range(1, nr_of_runes+1)) if progressbar else range(1, nr_of_runes+1):
            random.seed(17+idx)
            performance = self._RUN()
            if self.save:
                self.save_run_to_file(performance, idx, problem_name)

if __name__ == '__main__':
    problem_names = ['zdt3']
    for i in problem_names:
        problem_name = i
        if problem_name.startswith('DF'):
            problem = ps.problems_CEC[problem_name]
        else:
            problem = ps.problems_DEAP[problem_name]

        generations = [100]
        for i in generations:
            nsga = NSGA_III(problem_name = problem_name, 
                            problem = problem, 
                            num_gen=i, 
                            pop_size=20, 
                            cross_prob=1.0, 
                            mut_prob=1.0, 
                            MP=0, 
                            verbose=False,
                            save = False)
        
            nsga.multiple_runs(problem_name = problem_name, nr_of_runes=1, progressbar=True)


