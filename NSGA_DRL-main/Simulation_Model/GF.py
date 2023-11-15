""" Global Functions """
import pandas as pd
import numpy as np
import pickle

def dataframe_to_dictionary(fileName, sheetName=None):
    """ Read and transform dataframe into dictionary for faster lookup """
    # Read dataframe (specific sheet if given)
    df = pd.read_excel(f'Simulation_Model/Data/{fileName}.xlsx', sheetName) if sheetName != None else pd.read_excel(f'Simulation_Model/Data/{fileName}')
    
    # Initialize dictionary
    df_dict = {}
    
    # Loop over all dataframe rows    
    for idx, row in df.iterrows():
        
        # If from, to, dist columns present
        if len(df.columns) == 3:
            # Append 'to' location and distance to dictionary
            try:
                df_dict[row.From].update({row.To: row.Dist})  
                
            # If 'from' location not present as key in dict, create first
            except KeyError:
                df_dict[row.From] = {}
                df_dict[row.From].update({row.To: row.Dist})
        
        # If only location, distance columns present
        else:
            df_dict[row.From] = row.Dist
            
    # Return created dictionary
    return df_dict

def quantities_per_orderline(pallets, orderlines):
    """ Return a list of quantities for the given number of pallets and orderlines """
    # Calculate integer division and 'overshoot' of the integer division
    int_division = pallets // orderlines
    overshoot = pallets % orderlines
    
    # Return the calculated quantities
    return [int_division if idx >= overshoot else int_division + 1 for idx in range(orderlines)]

def flatten_list(list_to_flatten):
    """ Return a flattened list from the given list of lists """
    return [item for sublist in list_to_flatten for item in sublist]

def flatten_array(array_to_flatten):
    """ Return a flattened array from the given list of arrays """
    return np.array([item for sublist in array_to_flatten for item in sublist])

def merge_dictionaries(dict1, dict2):
    """ Append dictionary 2 to dictionary 1 """
    return (dict1.update(dict2))

def create_fractions(vals):
    """ Transform given values to fractions of the total """
    total_vals = sum(vals)
    return [v/total_vals for v in vals]

def arreq_in_list(arr, list_of_arrs):
    """ Test whether the given array is in the list of arrays """
    return next((True for elem in list_of_arrs if np.array_equal(elem, arr)), False)

def save_to_file(data, filename):
    """ Create a file using the given file name and save the given data in this pickled file """
    file = open(f"Results/{filename}.pkl", "wb")
    pickle.dump(data, file)
    file.close()

def load_data(filename):
    """ Load data from the given filename """
    file = open(f"Results/{filename}", "rb")
    data = pickle.load(file)
    file.close()
    return data