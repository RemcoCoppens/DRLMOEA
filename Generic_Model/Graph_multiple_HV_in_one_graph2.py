
import pickle
import numpy as np
import matplotlib.pyplot as plt	

def open_pickle2(problem_name, run, directory, gens, load_model = None):
    
    if load_model == None:
        file = open(f"Results/{f'{directory}/Problem_{problem_name}_Run_{run}_Gens_{gens}_POP_size_20'}.pkl", "rb")
    else:
        file = open(f"Results/{f'{directory}/Problem_{problem_name}_Run_{run}_POP_size_20_{load_model}'}.pkl", "rb")
    data = pickle.load(file)
    return data

def get_hypervolume_values2(problem_name, run, directory, gens, load_model = None):
    hypervolume_per_gen = [[] for gen in range(gens+1)]
    min = []
    max = []
    avg = []
    std = []

    for i in range(1,run+1):
        data = open_pickle2(problem_name, i, directory, gens, load_model)
        hv = data["hypervolume"]
        
        for i in range(gens+1):
            hypervolume_per_gen[i].append(hv[i])

    for i in hypervolume_per_gen:
        min.append(np.min(i))
        max.append(np.max(i))
        avg.append(np.mean(i))
        std.append(np.std(i))
    return min, max, avg, std


def open_pickle(problem_name, directory, load_model):
    if load_model == None:
        with open(f'Results/{directory}.pkl', 'rb') as f:
            dfs = []
            
            while True:
                try:
                    dfs.append(pickle.load(f))
                except EOFError:
                    break
    else:
        with open(f'Results/{directory}/Problem_{problem_name}_POP_size_20_{load_model}.pkl', 'rb') as f:
            dfs = []
            
            while True:
                try:
                    dfs.append(pickle.load(f))
                except EOFError:
                    break
    return dfs

def get_hypervolume_values(problem_name, run, directory, gens, load_model = None):
    hypervolume_per_gen = [[] for gen in range(gens+1)]
    min = []
    max = []
    avg = []
    std = []

    data = open_pickle(problem_name, directory, load_model)

    for i in range(run):
        hv = data[i]['hypervolume']

        for i in range(gens+1):
            hypervolume_per_gen[i].append(hv[i])

    for i in hypervolume_per_gen:
        min.append(np.min(i))
        max.append(np.max(i))
        avg.append(np.mean(i))
        std.append(np.std(i))
    return min, max, avg, std

def plot_hypervolume(problem_name, run, nsga,  directory, directory2, gens, model1 = None, model2 = None, model3 = None, model4 = None, model5 = None, model6 = None, model7 = None, model8 = None, model9 = None, model10 = None):
    if problem_name == "dtlz2":
        main_nsga = get_hypervolume_values2(problem_name, run, 'NSGA-III_NSGA-III_dtlz2_36variables_withseed' , gens)
        min, max, avg, std = main_nsga
        plt.errorbar(np.arange(len(avg)), avg, yerr=std, fmt='-', color='red', ecolor='lightcoral', elinewidth=5, capthick=1, label='NSGA-III')

    else:
        main_nsga = get_hypervolume_values(problem_name, run, nsga, gens)
        min, max, avg, std = main_nsga
        plt.errorbar(np.arange(len(avg)), avg, yerr=std, fmt='-', color='red', ecolor='lightcoral', elinewidth=5, capthick=1, label='NSGA-III')

    if model1 != None:
        model1_values = get_hypervolume_values(problem_name, run, directory, gens, model1)
        min, max, avg, std = model1_values
        plt.errorbar(np.arange(len(avg)), avg, yerr=std, fmt='-', color='blue', ecolor='lightblue', elinewidth=5, capthick=1, label=model1)
    if model2 != None:
        model2_values = get_hypervolume_values(problem_name, run, directory, gens, model2)
        min, max, avg, std = model2_values
        plt.errorbar(np.arange(len(avg)), avg, yerr=std, fmt='-', color='green', ecolor='lightgreen', elinewidth=5, capthick=1, label=model2)
    if model3 != None:
        model3_values = get_hypervolume_values(problem_name, run, directory2, gens, model3)
        min, max, avg, std = model3_values
        plt.errorbar(np.arange(len(avg)), avg, yerr=std, fmt='-', color='yellow', ecolor='lightyellow', elinewidth=5, capthick=1, label=model3)
    if model4 != None:
        model4_values = get_hypervolume_values(problem_name, run, directory2, gens, model4)
        min, max, avg, std = model4_values
        plt.errorbar(np.arange(len(avg)), avg, yerr=std, fmt='-', color='orange', ecolor='lightorange', elinewidth=5, capthick=1, label=model4)
   
    #plt.fill_between(avg, min, max, color='red', alpha=0.3, label='Bounds')
    plt.xlabel("Generation", )
    plt.ylabel("Hypervolume")
    plt.title(f"Problem {problem_name}")
    #plt.ylim(0,1)
    #plt.yticks(np.arange(0, 1.05, step = 0.1))
    #plt.xticks(np.arange(0, gens+1, step = 1))
    plt.legend(loc = 'lower right')

    if model2 == None:
        plt.savefig(f'Results/{directory}/{problem_name}_Gens_{gens}_{model1}.png')
    elif model3 == None:
        plt.savefig(f'Results/{directory}/{problem_name}_Gens_{gens}_{model1}_{model2}.png') 
    elif model4 == None:
        plt.savefig(f'Results/{directory}/{problem_name}_Gens_{gens}_{model1}_{model2}_{model3}.png')
    else:
        plt.savefig(f'Results/{directory}/{problem_name}_Gens_{gens}_{model1}_{model2}_{model3}_{model4}.png')
    
    plt.show()

problem_name = 'DF14'
doc = plot_hypervolume(problem_name= problem_name, 
                       run = 10, 
                       directory = "NSGA-III_DRL_Transferability_30-01-2024", 
                       directory2 = "NSGA-III_DRL",
                       gens =100,
                       nsga = f'NSGA-III/Problem_{problem_name}_POP_size_20',
                       model1 = 'Bestmodel_30-01-2024_dtlz2',
                       model2 = 'Lastmodel_30-01-2024_dtlz2',
                       model3 = None,
                       model4 = None
                       )

