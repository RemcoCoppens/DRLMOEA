import numpy as np
import pandas as pd
import multiprocessing
import random
import time
import os
import array
from copy import deepcopy
from scipy.spatial import distance
import matplotlib.pyplot as plt

from deap import base, creator, tools, algorithms
from deap.benchmarks.tools import hypervolume

from Agent_Classes.DuelingDQN_Agent import Agent
import Benchmark_Problem_Suites as ps

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
        #self.P = 12
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
                           actions = [[10.0,50.0,100.0],[10.0, 50.0, 100.0],
                           [0.01, 0.05, 0.10, 0.15, 0.20]],
                           batch_size = 32,
                           input_size = 7
                           )
        self.learn_agent = learn_agent
        if load_agent != None:
            self.agent.load_model(fname = f'{load_agent}.h5')
            self.agent.epsilon = 0

        self.stagnation_counter = 0
        self.hv_reference_point = np.array([1.0]*self.NBOJ)
        self.hv_trace = []

        self.hv_dict = {}
        self.policy_dict = {}
        
        self.run_performance = []
        self.run_reward = []
        self.run_epsilon = []

        

    
    def check_results_directory(self):
        """ Check if there are already results in the NSGA-II file, if so ask for overwrite and if requested create new file """
        if len(os.listdir("Results/NSGA-III_Learning")) > 0:
            selection = input("Existing result files found, do you want to overwrite? [y/n]")
            if selection == 'y' or selection == 'yes' or selection == 'Y' or selection == 'YES':
                return 'NSGA-III_Learning'
            elif selection == 'n' or selection == 'no' or selection == 'N' or selection == 'NO':
                folder_extension = input("Insert Folder Extension")
                os.mkdir(path=f'Results/NSGA-III_Learning_{folder_extension}') 
                return f'NSGA-III_Learning_{folder_extension}'
        else:
            return 'NSGA-III_Learning'   
    
    
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
        self.hv_trace.append(hv)
        return hv

    def create_offspring(self, population, operator, use_agent = True) -> list:
        """Create offspring from the current population (retrieved from DEAP varAnd module)"""
        #Clone parental population to create offspring population
        offspring = [deepcopy(indiv) for indiv in population]
        
        
        #Apply crossover for pairs of consecutive individuals
        #COMMENT: KIJK NAAR DE VALUES VAN ETA, VOOR CROSSOVER EN MUTATION
        #COMMENT: STRUCTUUR HANGT AF VAN WAT WE DE AGENT WILLEN LATEN DOEN
        
        for i in range(1, len(offspring), 2):
            if use_agent:
                if random.random() < self.CXPB:
                    offspring[i-1], offspring[i] = tools.cxSimulatedBinaryBounded(ind1= offspring[i-1],
                                                                                  ind2= offspring[i],
                                                                                  eta=operator[0],
                                                                                  low=self.BOUND_L,
                                                                                  up=self.BOUND_U)
                    del offspring[i-1].fitness.values, offspring[i].fitness.values
            else:
                if random.random() < self.CXPB:
                    offspring[i-1], offspring[i] = tools.cxSimulatedBinaryBounded(ind1= offspring[i-1],
                                                                                ind2= offspring[i],
                                                                                eta=30, 
                                                                                low=self.BOUND_L,
                                                                                up=self.BOUND_U)
                    del offspring[i-1].fitness.values, offspring[i].fitness.values

        #Apply mutation for every individual
        for i in range(len(offspring)):
            if use_agent: 
                if random.random() < self.MUTPB:
                    offspring[i], = tools.mutPolynomialBounded(individual=offspring[i],
                                                                eta=operator[1],
                                                                low=self.BOUND_L,
                                                                up=self.BOUND_U,
                                                                indpb=operator[2])
                    del offspring[i].fitness.values
            else:
                if random.random() < self.MUTPB:
                    offspring[i], = tools.mutPolynomialBounded(individual=offspring[i],
                                                                eta=20,
                                                                low=self.BOUND_L,
                                                                up=self.BOUND_U,
                                                                indpb=1/self.NDIM)
                    del offspring[i].fitness.values

        #Return a list of varied individuals that are independent of their parents
        return offspring

    def call_agent (self, gen, hv, pareto_size, state = [], action = None) -> tuple:
        """ Call the agent to retrieve the next action """
        if action == None:
            state = self.agent.create_state_representation(optims = self,
                                                           gen = gen,
                                                           hv = hv,
                                                           pareto_size = pareto_size)

            return state, self.agent.choose_action(state)
            
        else:
            state_ = self.agent.create_state_representation(optims = self,
                                                              gen = gen,
                                                              hv = hv,
                                                              pareto_size = pareto_size)
            
            reward_idx = self.agent.store_transition(state = state, action = action, new_state = state_)

            if self.learn_agent:
                self.agent.learn()

            state = state_
        return state, self.action.choose_action(state), reward_idx
    
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
        reward_trace = []
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
        #eval_start = time.time()
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        #avg_eval_time = (time.time() - eval_start) / len(invalid_ind)
        
        #Compile statistics about the population
        record = stats.compile(pop)
        self.logbook.record(gen=0, evals=len(invalid_ind), **record)
        if self.verbose:
            print(self.logbook.stream)
        
        #Calculate hypervolume of initial population
        pareto_front = self.retrieve_pareto_front(population=pop)
        prev_hv = self.calculate_hypervolume(pareto_front=pareto_front)

        #Obtain initial operator selection from agent
        state, action= self.call_agent(gen=0, hv=hv, pareto_size=len(pareto_front))
        
        operator_settings = self.agent.retrieve_operator(action = action)
        #COMMENT: kijken of deze retrieve operator echt nodig is, ik zie niet zo snel waarom namelijk
        #Het lijkt me dat de action de juiste action al terugkeert
        states_trace.append(state)
        actions_trace.append(operator_settings)

        #Save generation to file
        #COMMENT: KIJKEN OF WE TIJDENS TRAINEN DIT OOK NOG WILLEN OPSLAAN, WORDT BEETJE TEVEEL VAN HET GOEDE DENK IK
        #save_gen = self.save_generation(gen=0, population=pop, avg_eval_time=avg_eval_time, gen_time=0, pareto = pareto_front, hv = hv)
        #df = pd.DataFrame(save_gen)

        """Evolutionary process"""
        #Start the evolutionary process (measure total procesing time of evolution)
        for gen in range(1, self.NGEN+1):
            gen_start = time.time()
            
            #Create offspring
            offspring = self.create_offspring(population=pop, operator=operator_settings, use_agent = use_agent)

            #Evaluate the individuals of the offspring with an invalid fitness (measure evaluation time)
            #eval_start = time.time()
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            #avg_eval_time = (time.time() - eval_start) / len(invalid_ind)

            #Select the next population from parent and offspring
            pop = toolbox.select(pop + offspring, self.POP_SIZE)

            #Compile statistics about the new population
            record = stats.compile(pop)
            self.logbook.record(gen=gen, evals=len(invalid_ind), **record)
            if self.verbose:
                print(self.logbook.stream)

            #Calculate hypervolume of population
            pareto_front = self.retrieve_pareto_front(population=pop)
            cur_hv = self.calculate_hypervolume(pareto_front=pareto_front)

            if cur_hv <= prev_hv:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0

            """ Agent interaction """
            #Obtain next operator selection from agent
            state, action, reward_idx = self.call_agent(gen=gen, hv=cur_hv, pareto_size=len(pareto_front), state = state, action = action)
            
            operator_settings = self.agent.retrieve_operator(action = action)
            states_trace.append(state)
            actions_trace.append(operator_settings)
            reward_idx_trace.append(reward_idx)
            prev_hv = cur_hv

            #Calculate processig time of this generation
            gen_time = time.time() - gen_start



            """"Save results"""
            # #Save generation to file
            # if gen!= self.NGEN:
            #     save_gen = self.save_generation(gen=gen, population=pop, avg_eval_time=avg_eval_time, gen_time=gen_time, pareto = pareto_front, hv = hv)
            #     df = pd.concat([df, pd.DataFrame(save_gen)], ignore_index=True)
            
            # #Final generation
            # else:
            #     algorithm_execution_time = time.time() - Start_timer
            #     save_gen = self.save_generation(gen=gen, population=pop, avg_eval_time=avg_eval_time, gen_time=gen_time, pareto = pareto_front, hv = hv,
            #                                  final_pop=1,
            #                                  alg_exec_time=algorithm_execution_time) 
            #     df = pd.concat([df, pd.DataFrame(save_gen)], ignore_index=True)
            #     display(df)
        # Close the multiprocessing pool if used
        if self.MP > 0:
            pool.close()
            pool.join()

        #return df
        return states_trace[:-1], actions_trace[:-1], reward_idx_trace

    def multiple_runs(self, problem_name, nr_of_runes, progressbar = False):
        """ Run the NSGA-III algorithm multiple times """
        for idx in tqdm(range(1, nr_of_runes+1)) if progressbar else range(1, nr_of_runes+1):
            _, actions, reward_idx = self._RUN()

            #normalise and clip performance
            
            self.save_run_to_file(performance, idx, problem_name)
        

if __name__ == '__main__':
    problem_name = 'dtlz2'
    if problem_name.startswith('DF'):
        problem = ps.problems_CEC[problem_name]
    else:
        problem = ps.problems_DEAP[problem_name]

    nsga = NSGA_III_Learning(problem_name = problem_name, 
                    problem = problem, num_gen=5, 
                    pop_size=20, 
                    cross_prob=1.0, 
                    mut_prob=1.0, 
                    MP=12, 
                    verbose=False,
                    learn_agent=True, 
                    load_agent= None)
    
    nsga.multiple_runs(problem_name = problem_name, nr_of_runes=5, progressbar=True)