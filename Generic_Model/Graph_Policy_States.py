import pickle 
import numpy as np 
import matplotlib.pyplot as plt	


def open_pickle(problem_name, directory, load_model):
    with open(f'Results/{directory}/Problem_{problem_name}_POP_size_20_{load_model}.pkl', 'rb') as f:
        dfs = []
        
        while True:
            try:
                dfs.append(pickle.load(f))
            except EOFError:
                break
    return dfs

def plotpolicy(dfs, runs, gens):
    crossover = [[] for gen in range(gens+1)]
    mutation = [[] for gen in range(gens+1)]
    indptmut = [[] for gen in range(gens+1)]

    avgcrossover = []
    avgmutation = []
    avgindptmut = []

    mediancrossover = []
    medianmutation = []
    medianindptmut = []
    
    for i in range(runs):
        policy = dfs[i]["action"]
        for i in range(gens+1):
            crossover[i].append(policy[i][0])
            mutation[i].append(policy[i][1])
            indptmut[i].append(policy[i][2])
    
    for i in range(gens+1):
        avgcrossover.append(np.mean(crossover[i]))
        avgmutation.append(np.mean(mutation[i]))
        avgindptmut.append(np.mean(indptmut[i]))
        mediancrossover.append(np.median(crossover[i]))
        medianmutation.append(np.median(mutation[i]))
        medianindptmut.append(np.median(indptmut[i]))
    avgindptmut = [i*100 for i in avgindptmut]
    medianindptmut = [i*100 for i in medianindptmut]

    plt.plot(avgcrossover, label='Crossover')
    plt.plot(avgmutation, label='Mutation')
    plt.plot(avgindptmut, label='Independent Mutation')
    #plt.plot(mediancrossover, label='Crossovermed')
    #plt.plot(medianmutation, label='Mutationmed')
    #plt.plot(medianindptmut, label='Independent Mutationmed')
    plt.xlabel('Generations')
    plt.ylabel('Probability')
    plt.title('Policy evaluation over consecutive generations')
    plt.legend()
    return




def plotstates(dfs, runs, gens, labels):
    states = [[] for i in range(len(labels))]
    avgstates = [[] for i in range(len(labels))]
    print(states)
    for j in range(len(labels)):
        states[j] = [[] for gen in range(gens+1)]
    print(states)
    for i in range(runs):
        state = dfs[i]["state"]
        for i in range(gens+1):
            for j in range(len(labels)):
                states[j][i].append(state[i][j])
    for i in range(len(labels)):
        for j in range(gens+1):
            avgstates[i].append(np.mean(states[i][j]))
    avgstates[8] = [i/10 for i in avgstates[8]]



    for i in range(len(labels)):
        plt.plot(avgstates[i], label=labels[i])
    plt.xlabel('Generations')
    plt.ylabel('Average normalised value')
    plt.title('Average state representation over consecutive generations')
    plt.legend()
    plt.show()

    return


problem_name = 'dtlz3'
directory = 'NSGA-III_DRL'
load_model = 'Bestmodel_30-01-2024_dtlz2'
statesnames = ['gen', 'stag_count', 'mean', 'min', 'std' , 'norm_hv', 'pareto_size', 'spacing', 'hole relative size']
runs = 500
gens = 100

dfs = open_pickle(problem_name, directory, load_model)


plotpolicy(dfs, runs, gens) 
plt.figure()
plotstates(dfs, runs, gens, statesnames)