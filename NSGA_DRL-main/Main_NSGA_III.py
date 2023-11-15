import matplotlib.pyplot as plt
import numpy as np
import time
import multiprocessing
import random
import os

from deap import base, creator, tools, algorithms

from Simulation_Model.Evaluate_Individual import Evaluate
from Simulation_Model import GF

# Initialize creator class
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
creator.create("Individual", list, typecode="d", fitness=creator.FitnessMulti)

class NSGA_III:
    """  
    Description: Holds all functionality of the NSGA-III Multi-Objective Evolutionary Optimization Algorithm
    Input:  - num_gen: The number of generations to be executed until termination
            - pop_size: The number of individuals simultanteously represented in an evolutionary population
            - cross_prob: Cross-over probability
            - mut_prob: Mutation probability
            - MP: if set to a value higher than 0, the given number of cores will be utilized
            - verbose: If True the performance of every population will be printed
    """
    def __init__(self, num_gen, pop_size, cross_prob, mut_prob, MP=0, verbose=True):
        self.NGEN = num_gen
        self.POP_SIZE = pop_size
        self.CXPB = cross_prob
        self.MUTPB = mut_prob
        self.IND_SIZE = 12
        self.NOBJ = 3
        self.BOUND_L, self.BOUND_U = 0.0, 1.0
        self.P = 12
        self.eval = Evaluate(simulation_time=40, storage_fallback=True, check_validity=False)
        self.MP = MP
        self.verbose = verbose
        self.directory = self.check_results_directory()
    
    def check_results_directory(self):
        """ Check if there are already results in the NSGA-II file, if so ask for overwrite and if requested create new file """
        if len(os.listdir("Results/NSGA-III")) > 0:
            selection = input("Existing result files found, do you want to overwrite? [y/n]")
            if selection == 'y' or selection == 'yes' or selection == 'Y' or selection == 'YES':
                return 'NSGA-III'
            elif selection == 'n' or selection == 'no' or selection == 'N' or selection == 'NO':
                folder_extension = input("Insert Folder Extension")
                os.mkdir(path=f'Results/NSGA-III_{folder_extension}') 
                return f'NSGA-III_{folder_extension}'
        else:
            return 'NSGA-III'
    
    def save_generation_to_file(self, gen, population, avg_eval_time, gen_time, final_pop=None, alg_exec_time=None):
        """ Save performance of generation to file """
        # Summarize performance in dictionary object and save to file
        performance_dict = {}
        performance_dict['pareto_front'], performance_dict['pareto_front_indivs'] = self.retrieve_pareto(population)
        performance_dict['avg_eval_time'] = avg_eval_time
        performance_dict['gen_time'] = gen_time
        performance_dict['avg_obj'] = self.logbook[gen]['avg']
        performance_dict['max_obj'] = self.logbook[gen]['max']
        performance_dict['min_obj'] = self.logbook[gen]['min']
        performance_dict['std_obj'] = self.logbook[gen]['std']
        performance_dict['population'] = [list(self.eval.transform_individual(indiv)) for indiv in population]
        if final_pop != None and alg_exec_time != None:
            performance_dict['algorithm_execution_time'] = alg_exec_time
        
        GF.save_to_file(performance_dict, f'{self.directory}/GEN_{gen}')
            
    def retrieve_pareto(self, population):
        """ Calculate and return the pareto optimal set """
        pareto_front = tools.sortNondominated(individuals=population, 
                                              k=self.POP_SIZE, 
                                              first_front_only=True)[0]
        indivs = [list(self.eval.transform_individual(indiv)) for indiv in population]
        return [np.array(indiv.fitness.values) for indiv in pareto_front], indivs
        
    def _RUN(self) -> None:
        """ Run the NSGA-III optimization loop until the termination criterion is met """ 
        print(f'--- Start NSGA-III Run for {self.NGEN} generations and a population of size {self.POP_SIZE} distributing work over {self.MP} cores ---') 
        # Start time measure for complete algorithm
        algorithm_start = time.time()
        
        # Set parameters for individual and population creation
        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.random)
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                         toolbox.attr_float, n=self.IND_SIZE)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        # Initialize reference points and create evolutionary phases in toolbox
        ref_points = tools.uniform_reference_points(self.NOBJ, self.P)
        toolbox.register("evaluate", self.eval.eval_individual)
        toolbox.register("select", tools.selNSGA3, ref_points=ref_points)
        toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=self.BOUND_L, up=self.BOUND_U, eta=30.0)
        toolbox.register("mutate", tools.mutPolynomialBounded, low=self.BOUND_L, up=self.BOUND_U, eta=20.0, indpb=1.0/self.IND_SIZE)
        
        # Create multiple workers on the set number of cores
        if self.MP > 0:
            pool = multiprocessing.Pool(processes=self.MP)
            toolbox.register("map", pool.map)

        # Initialize statistics object
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)
        
        # Initialize logbook
        self.logbook = tools.Logbook()
        self.logbook.header = "gen", "evals", "std", "min", "avg", "max"
        
        # Initialize population
        pop = toolbox.population(n=self.POP_SIZE)

        # Evaluate the fitness of the individuals with an invalid fitness value (+ measure evaluation time)
        eval_start = time.time()
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        avg_eval_time = (time.time() - eval_start) / len(invalid_ind)
                    
        # Compite the population statistics and print (performance) stream
        record = stats.compile(pop)

        self.logbook.record(gen=0, evals=len(invalid_ind), **record)
        if self.verbose: 
            print(self.logbook.stream)

        # Document and save performance to file
        self.save_generation_to_file(gen=0, 
                                     population=pop, 
                                     avg_eval_time=avg_eval_time, 
                                     gen_time='N\A')

        # Start generational process and measure total processing time of generation
        for gen in range(1, self.NGEN+1):
            gen_start = time.time()
            
            # Create offspring through selection, crossover and mutation applied using the given percentages
            offspring = algorithms.varAnd(pop, toolbox, self.CXPB, self.MUTPB)
            
            # Evaluate the individuals with an invalid fitness (+ measure evaluation time)
            eval_start = time.time()
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            avg_eval_time = (time.time() - eval_start) / len(invalid_ind)
        
            # Select the next generation population from parents and offspring, constained by the population size
            pop = toolbox.select(pop + offspring, self.POP_SIZE)
            
            # Compile statistics about the new population
            record = stats.compile(pop)
            self.logbook.record(gen=gen, evals=len(invalid_ind), **record)
            if self.verbose:
                print(self.logbook.stream)
                
            # Calculate processing time of this generation
            gen_time = time.time() - gen_start
            
            # Document and save performance to file
            if gen != self.NGEN:
                self.save_generation_to_file(gen=gen, 
                                             population=pop, 
                                             avg_eval_time=avg_eval_time, 
                                             gen_time=gen_time)
                
            # If final generation, also save the final population and algorithm execution time
            else:
                algorithm_execution_time = time.time() - algorithm_start
                
                self.save_generation_to_file(gen=gen, 
                                             population=pop, 
                                             avg_eval_time=avg_eval_time, 
                                             gen_time=gen_time,
                                             final_pop=[list(self.eval.transform_individual(indiv)) for indiv in pop],
                                             alg_exec_time=algorithm_execution_time)
            
        # Close the multiprocessing pool if used
        if self.MP > 0:
            pool.close()
            pool.join()

if __name__ == '__main__':
    nsga = NSGA_III(num_gen=5, pop_size=20, cross_prob=1.0, mut_prob=1.0, MP=12, verbose=True)
    
    nsga._RUN()
