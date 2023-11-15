import matplotlib.pyplot as plt
import numpy as np
import time
import multiprocessing
import random
import os
from copy import deepcopy
from tqdm import tqdm

from deap import base, creator, tools, algorithms
from deap.benchmarks.tools import hypervolume

from DDQN_Agent_Episodic_Reward import Agent
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
    def __init__(self, num_gen, pop_size, cross_prob, mut_prob, MP=0, verbose=True, learn_agent=False, load_agent=None):
        self.NGEN = num_gen
        self.POP_SIZE = pop_size
        self.CXPB = cross_prob
        self.MUTPB = mut_prob
        self.IND_SIZE = 12
        self.NOBJ = 3
        self.BOUND_L, self.BOUND_U = 0.0, 1.0
        self.P = 12
        
        self.agent = Agent(lr=1e-4,
                           gamma=0.99, 
                           actions=[[10.0, 50.0, 100.0],
                                    [0.01, 0.05, 0.10, 0.15, 0.2]], 
                           batch_size=32,
                           eps_dec_exp=0.99825,
                           input_dims=7,
                           replace=self.NGEN*100)  # Replacing every 5 episodes of 200 generations
        self.learn_agent = learn_agent
        if load_agent != None:
            self.agent.load_model(fname=f'{load_agent}.h5')
            self.agent.epsilon=0
        
        self.eval = Evaluate(simulation_time=40, storage_fallback=True, check_validity=False)
        self.MP = MP
        self.verbose = verbose
        
        self.stagnation_counter = 0
        self.val_bounds = [(0.0, 25.0), (172600, 1702735), (0.0, 890.5)]
        self.std_bounds = [(0.9, 45.0), (62400, 575220), (118, 628)]
        
        self.hv_reference = np.array([1.0] * self.NOBJ)
        self.hv_tracking = []
        self.hv_dict = {}
        self.track_policy = {}
        
        self.episode_performance = []
        self.episode_rewards = []
        self.track_epsilon = []
        
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
    
    def create_offspring(self, population, operator, use_agent=True) -> list:
        """ Create offspring from the current population (retrieved from DEAP varAnd module) """
        offspring = [deepcopy(ind) for ind in population]

        # For every parent pair request settings from RL agent and apply crossover and mutation
        for i in range(1, len(offspring), 2):
            
            if random.random() < self.CXPB:
                offspring[i - 1], offspring[i] = tools.cxSimulatedBinaryBounded(ind1=offspring[i - 1], 
                                                                                ind2=offspring[i], 
                                                                                eta=33.8,  #30
                                                                                low=self.BOUND_L, 
                                                                                up=self.BOUND_U)
                del offspring[i - 1].fitness.values, offspring[i].fitness.values
            
            if use_agent:
                for idx in [i-1, i]:
                    if random.random() < self.MUTPB:
                        offspring[idx], = tools.mutPolynomialBounded(individual=offspring[idx], 
                                                                    eta=operator[0], 
                                                                    low=self.BOUND_L, 
                                                                    up=self.BOUND_U, 
                                                                    indpb=operator[1])
                        del offspring[idx].fitness.values
            else:
                for idx in [i-1, i]:
                    if random.random() < self.MUTPB:
                        offspring[idx], = tools.mutPolynomialBounded(individual=offspring[idx], 
                                                                    eta=79.6,   #20
                                                                    low=self.BOUND_L, 
                                                                    up=self.BOUND_U, 
                                                                    indpb=0.193)  #1/96
                        del offspring[idx].fitness.values
            
        return offspring
    
    def normalize(self, val, LB, UB, clip=True):
        """ Apply (bounded) normalization on the given value using the given bounds (LB, UB) """
        if clip:
            return min(max((val - LB)/(UB - LB), 0.0), 1.0)
        else:
            return (val - LB)/(UB - LB)
    
    def retrieve_pareto_front(self, population, return_indivs=False) -> list:
        """ Calculate the pareto front obtained by the evolutionary algorithm """
        pareto_front = tools.sortNondominated(individuals=population, 
                                              k=self.POP_SIZE, 
                                              first_front_only=True)[0]
        if return_indivs:
            return [np.array(indiv.fitness.values) for indiv in pareto_front], [self.eval.transform_individual(indiv) for indiv in pareto_front]
        else:
            return [np.array(indiv.fitness.values) for indiv in pareto_front]
    
    def calculate_hypervolume(self, pareto_front) -> float:
        """ Normalize values and calculate the hypervolume indicator of the current pareto front """
        # Retrieve and calculate pareto front figures
        normalized_pareto_set = np.array([tuple([self.normalize(val=obj_v[i], 
                                                                LB=self.val_bounds[i][0], 
                                                                UB=self.val_bounds[i][1]) for i in range(self.NOBJ)]) for obj_v in pareto_front])        
        hv = hypervolume(normalized_pareto_set * -1, self.hv_reference)
        self.hv_tracking.append(hv)
        return hv
    
    def call_agent(self, gen, hv, pareto_size, population, offspring=[], state=[], action=None, prev_hv=None) -> tuple:
        """ Call for action selection by agent and manage accompanying transition storing and learning """
        if action == None:
            state = self.agent.create_state_representation(optim=self, 
                                                           gen_nr=gen,
                                                           hv=hv,
                                                           pareto_size=pareto_size) 
            
            return state, self.agent.choose_action(observation=state)
            
        else:
            state_ = self.agent.create_state_representation(optim=self, 
                                                            gen_nr=gen,
                                                            hv=hv,
                                                            pareto_size=pareto_size)
                        
            idx = self.agent.store_transition(state=state, 
                                              action=action, 
                                              new_state=state_)
            
            if self.learn_agent:
                self.agent.learn()
            
            state = state_
        
            return state, self.agent.choose_action(observation=state), idx
        
    def save_generation_to_file(self, gen, population, avg_eval_time, gen_time, pareto, final_pop=None, alg_exec_time=None):
        """ Save performance of generation to file """
        # Summarize performance in dictionary object and save to file
        performance_dict = {}
        performance_dict['pareto_front'], performance_dict['pareto_front_indivs'] = self.retrieve_pareto_front(population, return_indivs=True)
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
            
    def _RUN(self, use_agent=True) -> None:
        """ Run the NSGA-III optimization loop until the termination criterion is met """ 
        print(f'--- Start NSGA-III Run for {self.NGEN} generations and a population of size {self.POP_SIZE} distributing work over {self.MP} cores ---') 
        # Start time measure for complete algorithm
        # random.seed(10)
        algorithm_start = time.time()
        
        # Initialize tracking lists of the agents interactions
        track_states = []
        track_actions = []
        reward_idx_tracker = []
        
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
        
        # Calculate pareto front and hypervolume indicator
        pareto = self.retrieve_pareto_front(population=pop)           
        prev_hv = self.calculate_hypervolume(pareto_front=pareto)
        
        
        # Retrieve initial operator settings
        state, action = self.call_agent(gen=0, 
                                        hv=prev_hv, 
                                        pareto_size=len(pareto), 
                                        population=pop)
        operator_settings = self.agent.retrieve_operator(action=action)
        track_states.append(state)
        track_actions.append(operator_settings)

        # Document and save performance to file
        self.save_generation_to_file(gen=0, 
                                     population=pop, 
                                     avg_eval_time=avg_eval_time, 
                                     gen_time='N\A',
                                     pareto=pareto)

        # Start generational process and measure total processing time of generation
        for gen in range(1, self.NGEN+1):
            gen_start = time.time()
            
            # Create offspring through selection, crossover and mutation applied using the given percentages
            offspring = self.create_offspring(population=pop, 
                                              operator=operator_settings,
                                              use_agent=use_agent)  
            
            # Evaluate the individuals with an invalid fitness (+ measure evaluation time)
            eval_start = time.time()
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            avg_eval_time = (time.time() - eval_start) / len(invalid_ind)
        
            # Select the next generation population from parents and offspring, constained by the population size
            prev_pop = [deepcopy(ind) for ind in pop]
            pop = toolbox.select(pop + offspring, self.POP_SIZE)
            
            # Compile statistics about the new population
            record = stats.compile(pop)
            self.logbook.record(gen=gen, evals=len(invalid_ind), **record)
            if self.verbose:
                print(self.logbook.stream)
             
            # Update stagnation counter according to the change in hypervolume indicator
            pareto = self.retrieve_pareto_front(population=pop)           
            cur_hv = self.calculate_hypervolume(pareto_front=pareto)
            if cur_hv <= prev_hv:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
             
            # Request action from agent and save/learn from interaction
            state, action, reward_idx = self.call_agent(gen=gen, 
                                                        hv=cur_hv, 
                                                        pareto_size=len(pareto), 
                                                        population=prev_pop, 
                                                        offspring=offspring, 
                                                        state=state, 
                                                        action=action,
                                                        prev_hv=prev_hv)
            operator_settings = self.agent.retrieve_operator(action=action)
            track_states.append(state)
            track_actions.append(operator_settings)
            reward_idx_tracker.append(reward_idx)
            prev_hv = cur_hv    
                
            # Calculate processing time of this generation
            gen_time = time.time() - gen_start
            
            # Document and save performance to file
            if gen != self.NGEN:
                self.save_generation_to_file(gen=gen, 
                                             population=pop, 
                                             avg_eval_time=avg_eval_time, 
                                             gen_time=gen_time,
                                             pareto=pareto)
                
            # If final generation, also save the final population and algorithm execution time
            else:
                algorithm_execution_time = time.time() - algorithm_start
                
                self.save_generation_to_file(gen=gen, 
                                             population=pop, 
                                             avg_eval_time=avg_eval_time, 
                                             gen_time=gen_time,
                                             pareto=pareto,
                                             final_pop=[list(self.eval.transform_individual(indiv)) for indiv in pop],
                                             alg_exec_time=algorithm_execution_time)
            
        # Close the multiprocessing pool if used
        if self.MP > 0:
            pool.close()
            pool.join()
    
    def run_episodes(self, nr_of_episodes, progressbar=False):
        """ Run the set number of episodes on varying problem suites """
        # print('{:>10} | {:>15} | {:>15}'.format("Episode", "Epsilon", "Total Reward"))
        
        for idx in tqdm(range(1, nr_of_episodes+1)) if progressbar else range(1, nr_of_episodes+1):  
            _, actions, rewards, reward_idx = self._RUN()
            
            # Normalize and clip performance
            clipped_performance = max((sum(rewards)/self.NGEN), -0.5)
            self.agent.store_reward(performance=clipped_performance,
                                    indeces=reward_idx)
            
            # print('{:>10} | {:>15} | {:>15}'.format(idx, round(self.agent.epsilon,5), str(round(clipped_performance, 4))))
                    
            self.episode_rewards.append(rewards)
            self.episode_performance.append(clipped_performance)
            self.hv_dict[idx] = self.hv_tracking.copy()
            self.hv_tracking = []
            self.track_epsilon.append(self.agent.epsilon)
            self.track_policy[idx] = actions

            # Decay epsilon, to decrease exploration and increase exploitation
            self.agent.epsilon_decay_exponential(idx) 

if __name__ == '__main__':
    nsga = NSGA_III(num_gen=200, 
                    pop_size=20, 
                    cross_prob=1.0, 
                    mut_prob=1.0, 
                    MP=10, 
                    verbose=True, 
                    learn_agent=False, 
                    load_agent='DDQN_11-02-2022')
        
    nsga._RUN(use_agent=True)