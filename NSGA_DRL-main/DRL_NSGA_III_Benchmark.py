import numpy as np
import multiprocessing
import random
from copy import deepcopy
from scipy.spatial import distance
import matplotlib.pyplot as plt

from deap import base, creator, tools
from deap.benchmarks.tools import hypervolume

from DDQN_Agent_Episodic_Reward import Agent
import Problem_Suites as ps

from tqdm import tqdm
import pickle

# Initialize creator class
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
creator.create("Individual", list, typecode="d", fitness=creator.FitnessMulti)

class NSGA_III_Learning:
    """  
    Description: Holds all functionality of the NSGA-III Multi-Objective Evolutionary Optimization Algorithm
    Input:  - num_gen: The number of generations to be executed until termination
            - pop_size: The number of individuals simultanteously represented in an evolutionary population
            - cross_prob: Cross-over probability
            - mut_prob: Mutation probability
            - MP: if set to a value higher than 0, the given number of cores will be utilized
            - verbose: If True the performance of every population will be printed
    """
    def __init__(self, num_gen, pop_size, cross_prob, mut_prob, lr, MP=0, verbose=True, learn_agent=True, load_agent=None):
        self.NGEN = num_gen
        self.POP_SIZE = pop_size
        self.CXPB = cross_prob
        self.MUTPB = mut_prob
        self.NOBJ = 3
        self.P = 12
        self.BOUND_L, self.BOUND_U = 0.0, 1.0
        self.agent = Agent(lr=lr,  # 5e-4
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
            
        self.MP = MP
        self.verbose = verbose
        self.final_pop = []
        self.stagnation_counter = 0
        
        self.max_indiv_dist = distance.euclidean([0]*self.NOBJ, [1]*self.NOBJ)
        self.hv_reference = np.array([1.0] * self.NOBJ)
        self.hv_tracking = []
        self.hv_dict = {}
        self.track_policy = {}
        
        self.episode_performance = {problem: [] for problem in ps.problem_names}
        self.episode_rewards = {problem: [] for problem in ps.problem_names}
        self.track_epsilon = {problem: [] for problem in ps.problem_names}
        
    def create_offspring(self, population, operator, use_agent=True) -> list:
        """ Create offspring from the current population (retrieved from DEAP varAnd module) """
        offspring = [deepcopy(ind) for ind in population]

        # For every parent pair request settings from RL agent and apply crossover and mutation
        for i in range(1, len(offspring), 2):
            
            if random.random() < self.CXPB:
                offspring[i - 1], offspring[i] = tools.cxSimulatedBinaryBounded(ind1=offspring[i - 1], 
                                                                                ind2=offspring[i], 
                                                                                eta=30,
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
                                                                    eta=20, 
                                                                    low=self.BOUND_L, 
                                                                    up=self.BOUND_U, 
                                                                    indpb=1/96)
                        del offspring[idx].fitness.values
            
        return offspring
    
    def normalize(self, val, LB, UB, clip=True):
        """ Apply (bounded) normalization on the given value using the given bounds (LB, UB) """
        if clip:
            return min(max((val - LB)/(UB - LB), 0.0), 1.0)
        else:
            return (val - LB)/(UB - LB)
    
    def retrieve_pareto_front(self, population) -> list:
        """ Calculate the pareto front obtained by the evolutionary algorithm """
        pareto_front = tools.sortNondominated(individuals=population, 
                                              k=self.POP_SIZE, 
                                              first_front_only=True)[0]
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
            
            reward = self.agent.reward_functions(pop=population, 
                                                 off=offspring, 
                                                 prev_hv=prev_hv, 
                                                 new_hv=hv)
            
            idx = self.agent.store_transition(state=state, 
                                              action=action, 
                                              new_state=state_)
            
            if self.learn_agent:
                self.agent.learn()
            
            state = state_
        
            return reward, state, self.agent.choose_action(observation=state), idx
          
    def _RUN(self, problem_name, use_agent=True) -> None:
        """ Run the NSGA-III optimization loop until the termination criterion is met """ 
        print(f'--- Start NSGA-III Run for {self.NGEN} generations and a population of size {self.POP_SIZE} distributing work over {self.MP} cores ---') 
        self.val_bounds = ps.problem_bounds[problem_name]['val']
        self.std_bounds = ps.problem_bounds[problem_name]['std']
        random.seed(10)
        
        # Retrieve problem suite from archive
        problem = ps.problems[problem_name]
        IND_SIZE = ps.problem_dims[problem_name]
        
        # Initialize tracking lists of the agents interactions
        track_states = []
        track_actions = []
        track_rewards = []
        reward_idx_tracker = []
    
        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.random)
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                         toolbox.attr_float, n=IND_SIZE)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        ref_points = tools.uniform_reference_points(self.NOBJ, self.P)
        toolbox.register("evaluate", problem.evaluate, return_values_of=["F"])
        toolbox.register("select", tools.selNSGA3, ref_points=ref_points)
            
        # If requested, create multiple workers on the set number of cores
        if self.MP > 0:
            pool = multiprocessing.Pool(processes=self.MP)
            toolbox.register("map", pool.map)
        
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)
        
        self.logbook = tools.Logbook()
        self.logbook.header = "gen", "evals", "std", "min", "avg", "max"
        
        # Initialize and evaluate population
        #probeersels
        pop = toolbox.population(n=self.POP_SIZE)
        print(pop)        
        fitnesses = []
        for i in pop:
            hoi = toolbox.evaluate(i)
            print('hoi')
            
            fitnesses.append(toolbox.evaluate(i))
        fitnesses = map(toolbox.evaluate, pop)
        print('dit is', fitnesses)
        fitnesses = list(fitnesses)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        #old version
        #pop = toolbox.population(n=self.POP_SIZE)
        #fitnesses = toolbox.map(toolbox.evaluate, pop)
        
        #for ind, fit in zip(pop, fitnesses):
        #   ind.fitness.values = fit[0]
    
        record = stats.compile(pop)
        self.logbook.record(gen=0, evals=self.POP_SIZE, **record)
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

        # Start generational process
        for gen in range(1, self.NGEN+1): 
            # Create offspring through selection, crossover and mutation applied using the given percentages
            offspring = self.create_offspring(population=pop, 
                                              operator=operator_settings,
                                              use_agent=use_agent)            
             
            # Evaluate the individuals with an invalid fitness (+ measure evaluation time)
            fitnesses = toolbox.map(toolbox.evaluate, offspring)
            for ind, fit in zip(offspring, fitnesses):
                ind.fitness.values = fit[0]
            
            # Select the next generation population from parents and offspring, constained by the population size
            prev_pop = [deepcopy(ind) for ind in pop]
            pop = toolbox.select(pop + offspring, self.POP_SIZE)
            
            # Compile statistics about the new population
            record = stats.compile(pop)
            self.logbook.record(gen=gen, evals=self.POP_SIZE, **record)
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
            prev_reward, state, action, reward_idx = self.call_agent(gen=gen, 
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
            track_rewards.append(prev_reward)
            reward_idx_tracker.append(reward_idx)
            prev_hv = cur_hv
            
        # Close the multiprocessing pool if used
        if self.MP > 0:
            pool.close()
            pool.join()
            
        self.final_pop = pop
            
        return track_states[:-1], track_actions[:-1], track_rewards, reward_idx_tracker
    
    def run_episodes(self, nr_of_episodes, progressbar=False):
        """ Run the set number of episodes on varying problem suites """
        # print('{:>10} | {:>15} | {:>15}'.format("Episode", "Epsilon", "Total Reward"))
        
        for idx in tqdm(range(1, nr_of_episodes+1)) if progressbar else range(1, nr_of_episodes+1):  
            problem = 'dtlz2'
            _, actions, rewards, reward_idx = self._RUN(problem_name=problem)
            
            # Normalize and clip performance
            clipped_performance = max((sum(rewards)/self.NGEN), -0.5)
            self.agent.store_reward(performance=clipped_performance,
                                    indeces=reward_idx)
            
            print('{:>10} | {:>15} | {:>15}'.format(idx, round(self.agent.epsilon,5), str(round(clipped_performance, 4))))
            
            self.episode_rewards[problem].append(rewards)
            self.episode_performance[problem].append(sum(rewards))
            self.hv_dict[idx] = self.hv_tracking.copy()
            self.hv_tracking = []
            self.track_epsilon[problem].append(self.agent.epsilon)
            self.track_policy[idx] = actions

            # Decay epsilon, to decrease exploration and increase exploitation
            self.agent.epsilon_decay_exponential(idx) 


if __name__ == '__main__':
    
    nsga = NSGA_III_Learning(num_gen=200,  #200
                             pop_size=40, #20
                             cross_prob=1.0, 
                             mut_prob=1.0, 
                             lr=1e-4,
                             MP=0, 
                             verbose=False, 
                             learn_agent=True)
                        
        
    nsga.run_episodes(nr_of_episodes=4000,
                      progressbar=True)
    

