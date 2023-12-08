import pickle
import numpy as np
import matplotlib.pyplot as plt	



def open_pickle(problem_name, run, directory):
    file = open(f"Results/{f'{directory}/Problem_{problem_name}_Run_{run}'}.pkl", "rb")
    data = pickle.load(file)
    return data






if __name__ == '__main__':
    problem_name = 'dtlz2'
    runs = 5
    directory = 'NSGA-III'
    hypervolume = []

    for i in range(1,runs+1):
        data = open_pickle(problem_name, i, directory)
        hypervolume.append(data['hypervolume'][0])

        print(hypervolume)
