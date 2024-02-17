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

    def __init__(self, problem_name, problem, num_gen, pop_size, cross_prob, mut_prob, MP, verbose = True):
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
        #self.directory = self.check_results_directory()
        self.val_bounds = ps.problem_bounds[problem_name]['val']
        self.val_bounds_population = [[0,0] for _ in range(self.NBOJ)]
        self.val_bounds_pareto = [[0,0] for _ in range(self.NBOJ)]
        self.hv_reference_point = np.array([1.0]*self.NBOJ)
        self.hv_bounds = [1.0, 0.0]
        self.hv_bounds2 = [0.99, 1.0]
        self.hv_lowerbound = 1.0
        
        
        


    def check_results_directory(self):
        """ Check if there are already results in the NSGA-III file, if so ask for overwrite and if requested create new file """
        if len(os.listdir("Results/NSGA-III_HV")) > 0:
            selection = input("Existing files found, do you want to overwrite? [y/n]")
            if selection == 'y' or selection == 'yes' or selection == 'Y' or selection == 'YES':
                return 'NSGA-III_HV'
            elif selection == 'n' or selection == 'no' or selection == 'N' or selection == 'NO':
                folder_extension = input("Insert Folder Extension")
                os.mkdir(path=f'Results/NSGA-III_HV_{folder_extension}') 
                return f'NSGA-III_{folder_extension}'
        else:
            return 'NSGA-III_HV'
        
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
        file = open(f"Results/{f'{self.directory}/Problem_{problem_name}_Run_{run}_Gens_{self.NGEN}_POP_size_{self.POP_SIZE}'}.pkl", "wb")
        pickle.dump(performance, file)
        file.close()

    def normalize(self, val, LB, UB, clip=True):
        """ Apply (bounded) normalization on the given value using the given bounds (LB, UB) """
        if clip:
            return min(max((val - LB)/(UB - LB), 0.0), 1.0)
        else:
            return (val - LB)/(UB - LB)
        
    def normalize_lower_bound(self, val, LB, clip=True):
        """ Apply (bounded) normalization on the given value using the given bounds (LB, UB) """
        if clip:
            return max((val - LB), 0.0)
        else:
            return (val - LB)

    def retrieve_pareto_front(self, population, gen= None):
        """ Calculate and return the pareto optimal set """
        pareto_front = tools.sortNondominated(individuals=population, 
                                              k=self.POP_SIZE, 
                                              first_front_only=True)[0]
        
        if gen == 0:
            indivs = [list(np.array(indiv.fitness.values)) for indiv in pareto_front]
            for ind in indivs:
                for i in range(self.NBOJ):
                    if ind[i] >= self.val_bounds_pareto[i][1]:
                        self.val_bounds_pareto[i][1] = ind[i]

       
        return [np.array(indiv.fitness.values) for indiv in pareto_front]
    
    def calculate_hypervolume(self, pareto_front) -> float:
        """ Normalize values and calculate the hypervolume indicator of the current pareto front """
        # Retrieve and calculate normalised pareto front set
        normalized_pareto_set = np.array([tuple([self.normalize(val=obj_v[i], 
                                                                LB=self.val_bounds[i][0], 
                                                                UB=self.val_bounds[i][1]) for i in range(self.NBOJ)]) for obj_v in pareto_front])        
        normalized_pareto_set_population = np.array([tuple([self.normalize(val=obj_v[i], 
                                                                LB=self.val_bounds_population[i][0], 
                                                                UB=self.val_bounds_population[i][1]) for i in range(self.NBOJ)]) for obj_v in pareto_front])        
        
        normalized_pareto_set_pareto = np.array([tuple([self.normalize(val=obj_v[i], 
                                                                LB=self.val_bounds_pareto[i][0], 
                                                                UB=self.val_bounds_pareto[i][1]) for i in range(self.NBOJ)]) for obj_v in pareto_front])        
        
        
        hv = hypervolume(normalized_pareto_set, self.hv_reference_point)
        hv_population = hypervolume(normalized_pareto_set_population, self.hv_reference_point)
        hv_pareto = hypervolume(normalized_pareto_set_pareto, self.hv_reference_point)
        #self.hv_tracking.append(hv)
        return hv, hv_population, hv_pareto

    def binary_hv(self, hv_list):
        if hv_list[-1] >= hv_list[-2]:
            binary_hv = 1
        else:
            binary_hv = 0
        
        return binary_hv

    def firstderivative_hv(self, hv_list):
        if len(hv_list) <2:
            return 0
        else:
            #return ((hv_list[-1] - hv_list[-2])/ hv_list[-2]) #RELATIVE CHANGE IN Y WITH RESPECT TO PREVIOUS VALUE
            return (hv_list[-1] - hv_list[-2]) #FIRST DERIVATIVE
    
    def secondderivative_hv(self, firstder_hv_list):
        
        if len(firstder_hv_list) <3:
            return 0
        else:
            #return ((firstder_hv_list[-1] - firstder_hv_list[-2])/ firstder_hv_list[-2])
            return (firstder_hv_list[-1] - firstder_hv_list[-2])


    def _RUN(self, warmup=False): 
        """Run the NSGA-III loop until the termination criterion is met"""
        """Initialization"""
        # Initialize creator class
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0,)*self.NBOJ)
        creator.create("Individual", np.ndarray , fitness=creator.FitnessMulti)
        
        print(f'Problem: {self.PROBLEM}')
        print(f'---start NSGA-III Run for {self.NGEN} generations and a population of size {self.POP_SIZE} distributing work over {self.MP} cores ---')
        #Start time
        Start_timer = time.time()
        norm_hv_list = []
        norm_hv_non_clip_list = []
        hv_list = []
        firstder_hv_list = []
        secondder_hv_list = []
        norm_hv_pop_list = []
        norm_hv_pareto_list = []

        norm_hv_with_min_list = []

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
            for i in range(self.NBOJ):
                if fit[i] >= self.val_bounds_population[i][1]:
                    self.val_bounds_population[i][1] = fit[i]
            
            
        avg_eval_time = (time.time() - eval_start) / len(invalid_ind)
        
        
        #Compile statistics about the population
        if warmup == False:
            record = stats.compile(pop)
            self.logbook.record(gen=0, evals=len(invalid_ind), **record)
            if self.verbose:
                print(self.logbook.stream)
        
        #Calculate hypervolume of initial population
        pareto_front = self.retrieve_pareto_front(population=pop, gen = 0)
        hv, norm_hv_population, norm_hv_pareto = self.calculate_hypervolume(pareto_front=pareto_front)
        hv_list.append(hv)
        binary_hv = 1
        firstder_hv = self.firstderivative_hv(hv_list)
        firstder_hv_list.append(firstder_hv)    
        secondder_hv = self.secondderivative_hv(firstder_hv_list)
        secondder_hv_list.append(secondder_hv)
        norm_hv_pop_list.append(norm_hv_population)
        norm_hv_pareto_list.append(norm_hv_pareto)
        
        if hv < self.hv_lowerbound:
            self.hv_lowerbound = hv

        norm_hv_with_min = self.normalize_lower_bound(hv, self.hv_lowerbound, clip = True)
        norm_hv_with_min_list.append(norm_hv_with_min)

        # print('hv', hv)
        # print('hv_list', hv_list)
        # print('firstder_hv', firstder_hv)
        # print('firstder_hv_list', firstder_hv_list)
        # print('secondder_hv', secondder_hv)
    
        if warmup == False:
            norm_hv = self.normalize(hv, self.hv_bounds[0], self.hv_bounds[1], clip=True)
            norm_hv_non_clip = self.normalize(hv, self.hv_bounds2[0], self.hv_bounds2[1], clip=True)
            norm_hv_list.append(norm_hv)
            norm_hv_non_clip_list.append(norm_hv_non_clip)

            
            
        
        #Save generation to file
        if warmup == False:
            save_gen = self.save_generation(gen=0, population=pop, avg_eval_time=avg_eval_time, gen_time=0, pareto = pareto_front, hv = hv)
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
            if warmup == False:
                record = stats.compile(pop)
                self.logbook.record(gen=gen, evals=len(invalid_ind), **record)
                if self.verbose:
                    print(self.logbook.stream)

            gen_time = time.time() - gen_start

            #Calculate hypervolume of population
            pareto_front = self.retrieve_pareto_front(population=pop)
            hv, norm_hv_population, norm_hv_pareto = self.calculate_hypervolume(pareto_front=pareto_front)
            hv_list.append(hv)
            
            binary_hv = self.binary_hv(hv_list)
            firstder_hv = self.firstderivative_hv(hv_list)
            firstder_hv_list.append(firstder_hv)    
            secondder_hv = self.secondderivative_hv(firstder_hv_list)
            secondder_hv_list.append(secondder_hv)
            norm_hv_pop_list.append(norm_hv_population)
            norm_hv_pareto_list.append(norm_hv_pareto)
            # print('hv', hv)
            # print('hv_list', hv_list)
            # print('firstder_hv', firstder_hv)
            # print('firstder_hv_list', firstder_hv_list)
            # print('secondder_hv', secondder_hv)
            if hv < self.hv_lowerbound:
                self.hv_lowerbound = hv

            norm_hv_with_min = self.normalize_lower_bound(hv, self.hv_lowerbound, clip = True)
            norm_hv_with_min_list.append(norm_hv_with_min)
            
            if warmup == False:
                norm_hv = self.normalize(hv, self.hv_bounds[0], self.hv_bounds[1], clip=True)
                norm_hv_non_clip = self.normalize(hv, self.hv_bounds2[0], self.hv_bounds2[1], clip=True)
                norm_hv_list.append(norm_hv)
                norm_hv_non_clip_list.append(norm_hv_non_clip)

            
           


            """"Save results"""
            #Save generation to file
            if warmup ==False:
                if gen!= self.NGEN:
                    save_gen = self.save_generation(gen=gen, population=pop, avg_eval_time=avg_eval_time, gen_time=gen_time, pareto = pareto_front, hv = hv)
                    df = pd.concat([df, pd.DataFrame(save_gen)], ignore_index=True)
                
                #Final generation
                else:
                    algorithm_execution_time = time.time() - Start_timer
                    save_gen = self.save_generation(gen=gen, population=pop, avg_eval_time=avg_eval_time, gen_time=gen_time, pareto = pareto_front, hv = hv,
                                                final_pop=1,
                                                alg_exec_time=algorithm_execution_time) 
                    df = pd.concat([df, pd.DataFrame(save_gen)], ignore_index=True)
                    #display(df)
        # Close the multiprocessing pool if used
        if self.MP > 0:
            pool.close()
            pool.join()
        
        if warmup:
            return norm_hv_list, hv_list
        else:
            return df, norm_hv_list, hv_list, firstder_hv_list, secondder_hv_list, norm_hv_non_clip_list, norm_hv_with_min_list, norm_hv_pop_list, norm_hv_pareto_list


    def multiple_runs(self, problem_name, nr_of_runes, progressbar = False):
        """ Run the NSGA-III algorithm multiple times """
        for i in range(10):
            _, hv_list = self._RUN(warmup=True)
            self.hv_bounds[0] = min(self.hv_bounds[0], min(hv_list))
            self.hv_bounds[1] = max(self.hv_bounds[1], max(hv_list))
            self.hv_bounds2[0] = min(self.hv_bounds2[0], min(hv_list))
            self.hv_bounds2[1] = max(self.hv_bounds2[1], max(hv_list))
            #self.hv_bounds[1] = 1.0

            print(self.hv_bounds)
            self.val_bounds_pareto = [[0,0] for _ in range(self.NBOJ)]
            self.val_bounds_population = [[0,0] for _ in range(self.NBOJ)]

        for idx in tqdm(range(1, nr_of_runes+1)) if progressbar else range(1, nr_of_runes+1):
            performance, norm_hv_list, hv_list, firstder_hv_list, secondder_hv_list, norm_hv_non_clip_list, norm_hv_with_min_list, norm_hv_pop_list, norm_hv_pareto_list = self._RUN()
            #self.save_run_to_file(performance, idx, problem_name)
            self.val_bounds_pareto = [[0,0] for _ in range(self.NBOJ)]
            self.val_bounds_population = [[0,0] for _ in range(self.NBOJ)]
            plt.plot(hv_list, label = 'hv')
            plt.plot(norm_hv_list, label = 'norm hv')
            plt.plot(norm_hv_pop_list, label = 'hv population')
            plt.plot(norm_hv_pareto_list, label = 'hv pareto')
            plt.ylim(0,1)
            #plt.plot(norm_hv_non_clip_list, label = 'norm hv non clip')
            #plt.plot(norm_hv_with_min_list, label = 'norm hv with min')
            #plt.plot(firstder_hv_list, label = 'firstder hv')
            #plt.plot(secondder_hv_list, label = 'secondder hv')

            plt.legend()
            plt.show()



if __name__ == '__main__':
    problem_name = 'dtlz1'
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
                        verbose=False)
    
        nsga.multiple_runs(problem_name = problem_name, nr_of_runes=1, progressbar=True)


