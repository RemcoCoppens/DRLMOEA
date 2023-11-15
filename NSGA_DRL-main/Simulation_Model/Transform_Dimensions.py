from Simulation_Model import GC, GF

import numpy as np
import operator

def fix_potential_bias(storage):
    """ Fix potential calculation bias due to rounding in lane assignment """
    # Fix potential minor calculation errors and ensure that the amount of dimensions per section is not exceeded
    for k in storage.keys():
        # Retrieve dimension amounts of section
        section_dims_vals = list(storage[k].values())
        
        # Check if the dimensions per section is exceeded
        if len(section_dims_vals) > GC.dims_per_section:
            
            # Take the number of highest values equal to the dimensions allowed per section
            chosen_idx = np.argsort(-np.array(section_dims_vals))[:GC.dims_per_section]        
            chosen_dims = np.array(list(storage[k].keys()))[chosen_idx]
    
            # Adjust the values of the section in the storage
            storage[k] = {str(dim): storage[k][dim] for dim in chosen_dims}
        
        # Calculate total percentage and divide other fractions to create correct total
        total_perc = sum(storage[k].values())
        for dim in storage[k].keys():
            storage[k][dim] = round(storage[k][dim] / total_perc, 3)

    # Return adjusted storage dictionary
    return storage

def create_storage_distribution_WDT(dims, section_widths, sorted_sec, dim_names, total_width):
    """ Create storage distribution given the set of dimensions and section widths """
    # Recalculate fractions and calculate width per dimension and argsort widths
    dims = dims / sum(dims)
    width_per_dim = dims * total_width
    sorted_idx = np.argsort(-width_per_dim)
    
    # Initialize section counter and storage dictionary
    sec_cntr = 0
    storage = {}
    
    # Initialize dimension index and retrieve width of largest dimension
    dim_idx = 0
    dim_remaining = width_per_dim[sorted_idx[dim_idx]]
    
    # Loop over sections in sorted section list (descending)
    for sec in sorted_sec:
    
        # Retrieve section width and initialize sections dictionary
        remaining_section_width = section_widths[sec]
        section_dict = {}
        
        # While width remaining, keep assigning storage
        while remaining_section_width > 0:
            # If remaining dimension width fits within remaining section width
            if dim_remaining < remaining_section_width:
                # Assign dimension and mutate remaining section width
                section_dict[dim_names[sorted_idx[dim_idx]]] = round(dim_remaining/section_widths[sec],3)
                remaining_section_width -= dim_remaining
                
                # Increment dimension counter
                dim_idx += 1
                
                # If the final dimension is exceeded, save section and break from loop
                if dim_idx >= len(sorted_idx):
                    storage[sec] = section_dict
                    break
                
                # If not, retrieve assigned dimension width of next dimension
                else:
                    dim_remaining = width_per_dim[sorted_idx[dim_idx]]
                
            # Else if the remaining dimension width is larger or equal to the remaining section width
            elif dim_remaining >= remaining_section_width:
                
                # Assign dimension partially and mutate remaining dimension and section width
                section_dict[dim_names[sorted_idx[dim_idx]]] = round(remaining_section_width/section_widths[sec],3)
                dim_remaining -= remaining_section_width
                remaining_section_width = 0
                
                # Save section dictionary to storage dictionary
                storage[sec] = section_dict
            
            # Check if the number of dimensions per section is exceeded
            if sec_cntr > GC.dims_per_section:
                return False, storage 
    
    # Fix potential minor calculation errors and ensure that the amount of dimensions per section is not exceeded
    fix_potential_bias(storage)
    
    
    # If valid distribution is created, return storage dictionary
    return True, storage

                

def trim_dims(dims, nr_of_sections):
    """ Trim dims before placement heuristic to match allowed dimensions per section """
    # Check if number of given dimensions exceeds the amount of different creation abilities
    if sum(dims>0) > (nr_of_sections * GC.dims_per_section):
        
        # Calculate the amount of dimensions exceeding the threshold
        amount_exceeding = sum(dims>0) - (nr_of_sections * GC.dims_per_section)
        
        # Remove smallest dimension(s) until amounts are valid
        for _ in range(amount_exceeding):
            valid_idx = np.where(dims > 0)[0]
            dims[valid_idx[dims[valid_idx].argmin()]] = 0
            
    # Return trimmed dimensions
    return dims


def B2B_lanes(section_widths, Sc, Sm, So, Mc, Mm, Mo, Lc, Lm, Lo):
    """ Calculate the amount of storage locations for each dimension """
    # Calculate total width and retrieve sorted section numbers based on their widths in descending order
    total_width = sum(section_widths.values())
    sorted_sec = [x[0] for x in sorted(section_widths.items(), key=operator.itemgetter(1), reverse=True)]
    
    # Convert fractions to numpy array and create dimension name list of strings
    dims = np.array([Sc, Sm, So, Mc, Mm, Mo, Lc, Lm, Lo])
    dim_names = ['Sc', 'Sm', 'So', 'Mc', 'Mm', 'Mo', 'Lc', 'Lm', 'Lo']
    
    # Trim dimensions to match the allowed number of dimensions per section
    dims = trim_dims(dims=dims, nr_of_sections=len(section_widths))
    
    # Keep trying to create a storage distribution until a valid distribution is created
    while True:
        valid, storage =  create_storage_distribution_WDT(dims=dims, 
                                                          section_widths=section_widths, 
                                                          sorted_sec=sorted_sec, 
                                                          dim_names=dim_names, 
                                                          total_width=total_width)
        # If not valid, remove smallest dimension
        if not valid:
            valid_idx = np.where(dims > 0)[0]
            dims[valid_idx[dims[valid_idx].argmin()]] = 0
            
        # If valid, break from loop and return distribution
        else:
            break
    
    # Return created dimension list
    return storage


def return_length(dim, depth, sqm):
    """ Given the width, depth and square meters to occupy, calculate the number of lanes that will be occupied """
    # Calculate the sqm of a single lane
    lane_sqm = GC.loc_width['SHT'][dim[0]] * (depth * GC.loc_length)
    
    # Calculate and return the rounded nr of lanes
    return round(sqm/lane_sqm, 0) * GC.loc_width['SHT'][dim[0]], round(sqm/lane_sqm, 0) * lane_sqm


def create_storage_distribution_SQM(dims, section_sqm, sqm_dict, sorted_sec, dim_names, total_sqm):
    """ Create storage distribution given the set of dimensions and section square meters """
    # Recalculate fractions and calculate sqm per dimension and argsort widths
    dims = dims / sum(dims)
    sqm_per_dim = dims * total_sqm
    sorted_idx = np.argsort(-sqm_per_dim)

    # Initialize section counter and storage dictionary
    sec_cntr = 0
    storage = {}
    
    # Initialize dimension index and retrieve width of largest dimension
    dim_idx = 0
    dim_remaining = sqm_per_dim[sorted_idx[dim_idx]]
    
    # Loop over sections in sorted section list (descending)
    for sec in sorted_sec:
        # Retrieve section width and initialize sections dictionary
        remaining_section_sqm = sqm_dict[sec]
        section_dict = {}
        
        # While width remaining, keep assigning storage
        while remaining_section_sqm > 0:
            
            # If remaining dimension width fits within remaining section width
            if dim_remaining < remaining_section_sqm:
                
                # Calcualte the lanes and square meters occupied by this dimension
                length_occupied, sqm_occupied = return_length(dim=dim_names[sorted_idx[dim_idx]], 
                                                              depth=section_sqm[sec][1], 
                                                              sqm=dim_remaining)
                
                # Append fractions to section dictionary
                section_dict[dim_names[sorted_idx[dim_idx]]] = round(sqm_occupied/sqm_dict[sec], 3)
                remaining_section_sqm -= dim_remaining
                
                # Increment dimension index counter
                dim_idx += 1
                
                # If the final dimension is exceeded, break from loop
                if dim_idx >= len(sorted_idx):
                    storage[sec] = section_dict
                    break
                
                # If not, retrieve assigned dimension square meters of next dimension
                else:
                    dim_remaining = sqm_per_dim[sorted_idx[dim_idx]]
                
            
            # Else if the remaining dimension width is larger or equal to the remaining section width
            elif dim_remaining >= remaining_section_sqm:
                
                # Assign remaining section square meters to current dimension
                section_dict[dim_names[sorted_idx[dim_idx]]] = round(remaining_section_sqm/sqm_dict[sec], 3)
                
                # Mutate dimension remaining variable and set remaining sections sqm to 0
                dim_remaining -= remaining_section_sqm
                remaining_section_sqm = 0
                
                # Save section dictionary to storage dictionary
                storage[sec] = section_dict
            
            # Check if the number of dimensions per section is exceeded
            if sec_cntr > GC.dims_per_section:
                return False, storage 

    # Fix potential minor calculation errors and ensure that the amount of dimensions per section is not exceeded
    fix_potential_bias(storage)
    
    # If valid distribution is created, return storage dictionary
    return True, storage          
            

def SHT_lanes(section_sqm, Sc, Sm, Mc, Mm, Mo, Lc, Lm):
    """ Calculate the amount of storage locations for each dimension """
    # Create sqm dictionary
    sqm_dict = {k: section_sqm[k][0] * (section_sqm[k][1] * GC.loc_length) for k in section_sqm.keys()}
    
    # Calculate the total square meter (sqm) of storage space and sort in descending order
    total_sqm = sum(sqm_dict.values())

    sorted_sec = [x[0] for x in sorted(sqm_dict.items(), key=operator.itemgetter(1), reverse=True)]
    
    # Convert fractions to numpy array and create dimension name list of strings
    dims = np.array([Sc, Sm, Mc, Mm, Mo, Lc, Lm])
    dim_names = ['Sc', 'Sm', 'Mc', 'Mm', 'Mo', 'Lc', 'Lm']
    
    # Trim dimensions to match the allowed number of dimensions per section
    dims = trim_dims(dims=dims, nr_of_sections=len(section_sqm))
    
    # Keep trying to create a storage distribution until a valid distribution is created
    while True:
        valid, storage =  create_storage_distribution_SQM(dims=dims, 
                                                          section_sqm=section_sqm, 
                                                          sqm_dict=sqm_dict, 
                                                          sorted_sec=sorted_sec, 
                                                          dim_names=dim_names, 
                                                          total_sqm=total_sqm)

        # If not valid, remove smallest dimension
        if not valid:
            valid_idx = np.where(dims > 0)[0]
            dims[valid_idx[dims[valid_idx].argmin()]] = 0
            
        # If valid, break from loop and return distribution
        else:
            break
    
    # Return created dimension list
    return storage


def BLK_lanes(section_sqm, S, L):
    """ Calculate the amount of storage locations for each dimension """
    # Create sqm dictionary
    sqm_dict = {k: section_sqm[k][0] * (section_sqm[k][1] * GC.loc_length) for k in section_sqm.keys()}
    
    # Calculate the total square meter (sqm) of storage space and sort in descending order
    total_sqm = sum(sqm_dict.values())

    sorted_sec = [x[0] for x in sorted(sqm_dict.items(), key=operator.itemgetter(1), reverse=True)]
    
    # Convert fractions to numpy array and create dimension name list of strings
    dims = np.array([S, L])
    dim_names = ['S', 'L']
    
    # Trim dimensions to match the allowed number of dimensions per section
    dims = trim_dims(dims=dims, nr_of_sections=len(section_sqm))
    
    # Keep trying to create a storage distribution until a valid distribution is created
    while True:
        valid, storage =  create_storage_distribution_SQM(dims=dims, 
                                                          section_sqm=section_sqm, 
                                                          sqm_dict=sqm_dict, 
                                                          sorted_sec=sorted_sec, 
                                                          dim_names=dim_names, 
                                                          total_sqm=total_sqm)

        # If not valid, remove smallest dimension
        if not valid:
            valid_idx = np.where(dims > 0)[0]
            dims[valid_idx[dims[valid_idx].argmin()]] = 0
            
        # If valid, break from loop and return distribution
        else:
            break
    
    return storage


def reshape_dictionary(dimensions):
    """ Reshape the given dimensions to a workable format for the simulation """
    # Initialize looping dictionary
    dim_dict = {}
    
    # Loop over all storage areas
    for sa in GC.storage_area_reference.keys():
        # Initialize storage area section collection list
        sa_sections = []
        
        # Loop over all sections belonging to this storage area
        for section_id in GC.storage_area_reference[sa]:
            # If section_id is 69, deterministically set to Sc: 1.0 (as space is too small to have variation)
            if section_id == 69:
                dims = {'Sc': 1.0}
            # If not 69, normally retrieve dimensions
            else:
                dims = dimensions[section_id]
            
            # Append dimensions to sa sections list
            sa_sections.append(dims)
        
        # Append storage area to looping dictionary
        dim_dict[sa] = sa_sections
    
    # Return dimensions dictionary
    return dim_dict
            