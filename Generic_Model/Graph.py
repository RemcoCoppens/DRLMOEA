import pickle
import numpy as np
import matplotlib.pyplot as plt	


def open_pickle(problem_name, run, directory, gens, load_model):
    
    if load_model == None:
        file = open(f"Results/{f'{directory}/Problem_{problem_name}_Run_{run}_Gens_{gens}_POP_size_20'}.pkl", "rb")
    else:
        file = open(f"Results/{f'{directory}/Problem_{problem_name}_Run_{run}_POP_size_20_{load_model}'}.pkl", "rb")
    data = pickle.load(file)
    return data

def plot_hypervolume(problem_name, run, directory, gens, load_model):
    hypervolume_per_gen = [[] for gen in range(gens+1)]
    min = []
    max = []
    avg = []
    std = []

    for i in range(1,run+1):
        data = open_pickle(problem_name, i, directory, gens, load_model)
        hv = data["hypervolume"]
        
        for i in range(gens+1):
            hypervolume_per_gen[i].append(hv[i])

    for i in hypervolume_per_gen:
        min.append(np.min(i))
        max.append(np.max(i))
        avg.append(np.mean(i))
        std.append(np.std(i))

    plt.errorbar(np.arange(len(avg)), avg, yerr=std, fmt='-', color='red', ecolor='lightcoral', elinewidth=5, capthick=1)
    #plt.fill_between(avg, min, max, color='red', alpha=0.3, label='Bounds')
    plt.xlabel("Generation", )
    plt.ylabel("Hypervolume")
    plt.title(f"Problem {problem_name}")
    plt.ylim(0,1)
    plt.yticks(np.arange(0, 1.05, step = 0.1))
    #plt.xticks(np.arange(0, gens+1, step = 1))
    if load_model == None:
        plt.savefig(f'Results/{directory}/{problem_name}_Gens_{gens}.png')
    else:
        plt.savefig(f'Results/{directory}/{problem_name}_Gens_{gens}_{load_model}.png') 
    
    plt.show()
doc = plot_hypervolume(problem_name= "dtlz2", 
                       run = 10, 
                       directory = "NSGA-III_DRL", 
                       gens =100,
                       load_model = 'Lastmodel_dtlz2')