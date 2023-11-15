import numpy as np
import math
import copy

from Simulation_Model import GC
from Simulation_Model.Storages.Back2BackStorage import B2BStorageLane as B2BSL
from Simulation_Model.Storages.BlockStorage import BlockStorageLane as BLKSL
from Simulation_Model.Storages.ShuttleStorage import ShuttleStorageLane as SHTSL


class Storage_Area:
    """
    Description: Manages a storage area consisting of multiple storage sections
    Input:  - area_nr: The number indicating the storage area
            - layout: The layout of the storage area retrieved from 'Storage_Areas.xlsx'
    """
    def __init__(self, area_nr, layout, storage_dimensions, corner_points):
        self.nr = area_nr
        self.layout = layout
        self.storage_dimensions = storage_dimensions[area_nr]
        self.DL, self.DR, self.WL, self.WR = corner_points
        self.storage_sections, self.IO, self.dimension_dist, self.storage_type_lanes, self.storage_type_locations = self.create_storage_sections()        
        self.lanes_occupied, self.locations_occupied = {'B2B': 0, 'BLK': 0, 'SHT': 0}, {'B2B': 0, 'BLK': 0, 'SHT': 0}
    
    def __str__(self):
        """ Print storage usage of storage area """
        message = f'---- Storage Area: {self.nr} ----\n'
        message += f'    B2B: lanes: {self.lanes_occupied["B2B"]}/{self.storage_type_lanes["B2B"]}   Locations: {self.locations_occupied["B2B"]}/{self.storage_type_locations["B2B"]}\n'
        message += f'    BLK: lanes: {self.lanes_occupied["BLK"]}/{self.storage_type_lanes["BLK"]}   Locations: {self.locations_occupied["BLK"]}/{self.storage_type_locations["BLK"]}\n'
        message += f'    SHT: lanes: {self.lanes_occupied["SHT"]}/{self.storage_type_lanes["SHT"]}   Locations: {self.locations_occupied["SHT"]}/{self.storage_type_locations["SHT"]}\n'
        return message
        
    def layout_scan(self):
        """ Loop through storage area layout and disect given storage locations """
        # Initialize monitoring variable and loop over layout row by row
        rows = []
        for row_id, row in self.layout.iterrows():
            # Initialize monitoring lists and indicator values
            cells = []
            counting=False
            storage_type = None
            
            # Loop over all values in the row
            for idx, cell in enumerate(row):
                # If not counting storage locations
                if not counting:
                    # If cell is not a driving lane (D) or a blocked lane (B)
                    if cell != 'D' and cell != 'B':
                        # If storage is B2B just store index, do not count
                        if cell[0:3] == 'B2B':
                            cells.append((idx, idx, cell[0:3], cell[-1]))
                        
                        else:
                            # Document starting index and storage type and set counter to True
                            start_cell = idx
                            storage_type = cell
                            counting=True
                
                # If counting storage locations
                else:
                    # If cell is a driving lane (D) or a blocked lane (B)
                    if cell == 'D' or cell == 'B':
                        # Append recorded storage locations to list and set counter to False
                        cells.append((start_cell, idx - 1, storage_type[0:3], storage_type[-1]))
                        counting=False
                    
                    # If another type of storage location is present
                    elif cell != storage_type:
                        # Append recorded storage locations to list and restart counter
                        cells.append((start_cell, idx - 1, storage_type[0:3], storage_type[-1]))
                        start_cell = idx
                        storage_type = cell
            
            # If still counting when reaching final column, append values to list            
            if counting:
                cells.append((start_cell, idx, storage_type[0:3], storage_type[-1]))
            
            # Append values of row to row monitoring list before starting next row
            rows.append(cells)
        
        # Initialize monitoring lists for calculating the length (vertical)
        lengths = {}
        
        # Loop over rows, storage types and locations found
        for row in rows:
            for loc in row:
                # If dict key exists increment length by location width
                try:
                    lengths[loc[0:3]] += GC.loc_width[loc[-2]][loc[-1]]
                # If dict key does not exists create key and set value to be length by location width
                except KeyError:
                    lengths[loc[0:3]] = GC.loc_width[loc[-2]][loc[-1]]
        
        # Create sorted list of storage locations from left to right
        location_keys = [k for k in lengths.keys()]
        locations = [location_keys[idx] for idx in np.argsort([k[0] for k in lengths.keys()])]
        
        # Return lengths and location order
        return lengths, locations


    def calc_storage_dist(self, storage_dims):
        """ Use the storage layout function to generate overview of amount of storage location of each dimension """
        # Scan the given layout and retrieve lengths and ordered locations
        lengths, ordered_locs = self.layout_scan()
        
        # Initialize storage lanes dictionary
        storage_lanes = {}
        
        # Loop over all storage locations
        for idx, loc in enumerate(ordered_locs):
            # Retrieve storage type, available length and dimensions
            storage_type = loc[-1]
            total_length = lengths[loc]
            dims = storage_dims[idx]
            
            # Initialize storage lane section in dictionary
            storage_lanes[loc] = []
            
            # Loop over dimensions and assign storage area
            for d in dims.keys():
                # Calculate the amount of storage location lanes
                length_assigned = total_length * dims[d]
                
                # If B2B only allow even number of storages
                if storage_type == 'B2B':
                    storage_lanes[loc].append((d, math.floor(round(length_assigned / GC.loc_widths[storage_type][d]['width'], 1))//2 * 2))
                else:
                    storage_lanes[loc].append((d, math.floor(round(length_assigned / GC.loc_widths[storage_type][d]['width'], 1))))
                
        # Return storage distribution
        return storage_lanes


    def retrieve_driving_lane(self, left, right):
        """ Check whether the left or right lane is adjacent a driving lane """
        # Adjust left and right index
        adj_left = left - 1
        adj_right = right + 1
        
        # Check whether the adjusted left location falls within the boundaries
        if adj_left >= 0 and (self.layout[adj_left] == 'D').all(0):
            return adj_left, 0  # Left is second index
            
        # Check whether the adjusted right location falls within the boundaries
        elif adj_right < len(self.layout.columns) and (self.layout[adj_right] == 'D').all(0):
            return adj_right, 1  # Right is second index
        
        # If none is found, raise error
        else:
            raise ValueError('Both Left and Right are not driving lanes')

    def increment_ipd(self, current_width, next_width, in_path_dist):
        """ Increment the in path distance using half the current width and half the next width """
        return in_path_dist + (current_width/2) + (next_width/2)
    
    def create_storage_locations(self, storage_type, dims, depth, storage_start, section_nr):
        """ Create storage locations with in path dist (ipd) variable """
        # Initialize ipd as half of the driving lane and the rest looking at the first dimension
        ipd = GC.avg_driving/2 + storage_start * GC.dimensions[storage_type][dims[0]]['width'] + GC.dimensions[storage_type][dims[0]]['width'] / 2
        
        # Select correct storage type for section
        if storage_type == 'SHT':
            storage_lane = SHTSL
        elif storage_type == 'B2B':
            storage_lane = B2BSL
        elif storage_type == 'BLK':
            storage_lane = BLKSL
        else:
            raise ValueError(f'Storage type: {storage_type}, does not exist!')
        
        # Find number of lanes and initialize storage dictionary
        nr_of_dims = len(dims)
        storage_section = {'depth': [depth] * nr_of_dims,
                           'width': [],
                           'height': [],
                           'ipd': [],
                           'storage': []}
        
        # Loop over all dimensions and create storage locations
        for idx, dim in enumerate(dims):
            ln = storage_lane(lane_nr= idx,
                              area_nr=self.nr,
                              section_nr=section_nr,
                              depth=depth,
                              dimensions=GC.dimensions[storage_type][dim],
                              ipd=ipd)
            storage_section['width'].append(GC.dimensions[storage_type][dim]['width'])
            if storage_type != 'BLK':
                storage_section['height'].append(GC.dimensions[storage_type][dim]['height'])
            storage_section['ipd'].append(ipd)
            storage_section['storage'].append(ln)
            
            # Increment ipd if the last dimension has not been reached yet
            if idx < (nr_of_dims -1):
                ipd = self.increment_ipd(current_width=GC.dimensions[storage_type][dim]['width'], 
                                         next_width=GC.dimensions[storage_type][dims[idx+1]]['width'], 
                                         in_path_dist=ipd)
                
        # Return created dict of storage lanes
        return storage_section

    def add_dimensions(self, dim_dict, cntr, storage_type, dim_type, dim_lanes, section_nr):
        """ Append location to dimension dictionary correctly """
        # If block storage, no levels have to be added
        if storage_type == 'BLK':
            dim = GC.dimensions[storage_type][dim_type]
            _ = [dim_dict[storage_type][dim['width']].append((section_nr, lane_nr)) for lane_nr in range(cntr, cntr+dim_lanes)]
            
        # If B2B or shuttle storage, add different levels in dictionary
        else:
            # Retrieve width and height and append to dict correctly
            dim = GC.dimensions[storage_type][dim_type]
            _ = [[dim_dict[storage_type][dim['width']][h].append((section_nr, lane_nr, level_nr)) for level_nr, h in enumerate(dim['height'])] for lane_nr in range(cntr, cntr + dim_lanes)]
        
        # Increment counter and return dictionary and counter
        cntr += dim_lanes
        return cntr
    
    def create_storage_sections(self):
        """ Create actual storage locations constituting the storage area """
        # Calculate storage distances
        storage_dist = self.calc_storage_dist(self.storage_dimensions)
        
        # Initialize looping variable for storage location sections
        sa_sections = {}
        storage_type_lanes = {'B2B': 0, 'BLK': 0, 'SHT': 0}
        storage_type_locations = {'B2B': 0, 'BLK': 0, 'SHT': 0}
        IO_points = []
        dimensions_dict = copy.deepcopy(GC.dimension_dict_template)
        
        # Loop over all locations
        for section_nr, sec in enumerate(storage_dist.keys()):            
            # Retrieve In-/Out point at driving lane
            IO_point, idx = self.retrieve_driving_lane(left=sec[0], right=sec[1])
            
            # Retrieve lane for dist calculation
            lane_layout = self.layout[sec[idx]]
            
            # Loop over lane (in reverse if needed) and find first storage location
            storage_start = None
            for lane_nr, sl in enumerate(reversed(lane_layout)) if GC.reversed_assignment[self.nr] else enumerate(lane_layout):
                if sl != 'B':
                    storage_start = lane_nr
                    break
                elif lane_nr == (len(lane_layout) -1):
                    raise ValueError(f'No storage start found in Storage Area: {self.nr}, Column: {sec[idx]}')
            
            # Calculate depth of given section of storage locations
            depth = sec[1] - sec[0] + 1
            if depth == 1 and f'SA_{self.nr}' in GC.predef_depths.keys():
                depth = GC.predef_depths[f'SA_{self.nr}']
            
            # Retrieve dimensions for this sections of storage locations
            section_dims = []
            nr_of_storages = 0
            cntr = 0
            for dim in reversed(storage_dist[sec]) if GC.reversed_assignment[self.nr] else storage_dist[sec]:
                # Retrieve dimension type and nr of lanes
                dim_type, dim_lanes = dim
                
                # Create list representing storage locations
                section_dims += dim_lanes * [dim_type]
                
                # Count number of lanes for different products in type of storage
                nr_of_storages += dim_lanes * len(GC.dimensions[sec[2]][dim_type]['height']) if sec[2] != 'BLK' else dim_lanes
                
                # Add storage locations to dimensions dictionary and increment counter
                cntr = self.add_dimensions(dim_dict=dimensions_dict, 
                                           cntr=cntr, 
                                           storage_type=sec[2], 
                                           dim_type=dim_type, 
                                           dim_lanes=dim_lanes, 
                                           section_nr=section_nr)
            
            # Create storage locations starting from lane start
            storage_section = self.create_storage_locations(storage_type=sec[2],
                                                            dims=section_dims,
                                                            depth=depth,
                                                            storage_start=storage_start,
                                                            section_nr=section_nr)
            
            # Append the amount of storage lanes to the monitoring dictionary
            storage_type_lanes[sec[2]] += nr_of_storages
            if sec[2] == 'BLK':
                storage_type_locations[sec[2]] += (nr_of_storages * depth * GC.empty_stack_level_BLK)
            else:
                storage_type_locations[sec[2]] += (nr_of_storages * depth)
            
            # Append section to dictionary
            sa_sections[section_nr] = storage_section
            IO_points.append(IO_point)
        
        # Return created storage sections
        return sa_sections, IO_points, dimensions_dict, storage_type_lanes, storage_type_locations