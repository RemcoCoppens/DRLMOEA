import pandas as pd
import numpy as np
import math

from Simulation_Model import GC
from Simulation_Model.Storages.StorageArea import Storage_Area
from Simulation_Model.Storages.Consolidation import Consolidation
from Simulation_Model.Storages.Loc_Dist import Distances
from Simulation_Model.Storages.Docks import Truck_Docks

class Storage:
    """
    Description: Manages all storage areas comprising the full warehouse
    Input: - ExcelFile: Excel file containing all storage area layouts
           - storage_dimensions: dictionary containing accompanying dimensions defining these storage areas
           - ppa: The Product Placement Algorithm
           -simres: Simulation Results class
    """
    def __init__(self, ExcelFile, storage_dimensions, ppa, tda, simres):
        self.storage_dimensions =  storage_dimensions
        self.ppa = ppa
        self.tda = tda
        
        self.simres = simres
        self.areas = self.create_storage_areas(excel=ExcelFile, 
                                               dimensions=storage_dimensions)
        self.docks = self.create_truck_docks()
        self.consolidation = self.create_consolidation()
        self.connect_docks_and_cons()  # Connect docks and consolidation areas to one another
        
        self.dist = Distances(self)
        self.simres_storage_figures()
        self.connect_storage_to_tda()
        
        # Create truck queues for trucks that are unable to dock or get a consolidation area (or both)
        self.dock_queue = []
        self.truck_queue = []
        self.cons_queue = []
    
    def __str__(self):
        """ Print for every storage area the fill and occupation rate """
        occ_lanes, av_lanes = {'B2B':0, 'BLK':0, 'SHT':0}, {'B2B':0, 'BLK':0, 'SHT':0}
        occ_locs, av_locs = {'B2B':0, 'BLK':0, 'SHT':0}, {'B2B':0, 'BLK':0, 'SHT':0}
        
        message = f'----- Fill and Occupation Rate(s) -----\n'
        for area in self.areas.values():
            message += f'--> Storage Area {area.nr} <--\n'
            for st in area.storage_type_lanes.keys():
                if area.storage_type_locations[st] == 0:
                    pass
                else:
                    message += f'{st} - Occupation: {round((area.lanes_occupied[st]/area.storage_type_lanes[st])*100,0)}%, \t Fill: {round((area.locations_occupied[st]/area.storage_type_locations[st])*100,0)}% \n'
                    occ_lanes[st] += area.lanes_occupied[st]
                    av_lanes[st] += area.storage_type_lanes[st]
                    occ_locs[st] += area.locations_occupied[st]
                    av_locs[st] += area.storage_type_locations[st]
            message += '\n'
        message += f'\n --> Total Warehouse <--\n'
        for st in occ_lanes.keys():
            message += f'{st} - Occupation: {round((occ_lanes[st]/av_lanes[st])*100,0)}%, \t Fill: {round((occ_locs[st]/av_locs[st])*100,0)}%\n'
        return message
    
    def connect_storage_to_tda(self):
        """ Connect tda to storage class """
        self.tda.storage = self
        self.tda.dist = self.dist
        self.tda.storage_areas = self.areas
        self.tda.consolidation = self.consolidation
        self.tda.docks = self.docks
    
    def simres_storage_figures(self):
        """ Document the storage figures in the simulations results class """
        # Count the storage locations and lanes in all areas
        for area in self.areas.keys():
            
            # Loop over all storage types
            for storage_type in self.simres.nr_of_locations.keys():
                self.simres.nr_of_locations[storage_type] += self.areas[area].storage_type_locations[storage_type]
                self.simres.nr_of_lanes[storage_type] += self.areas[area].storage_type_lanes[storage_type]
    
    def truck_arrival_possible(self, event):
        """ Check if the truck is able to arrive """
        if event.type == GC.Inbound_Truck_Arrival or event.type == GC.Outbound_Truck_Arrival:
            # Check if both a consolidation lane and a dock are available
            if any([dock.available_dock_and_cons(truck_type=event.truck.type) for dock in self.docks.values()]):
                return True
            else:
                return False
            
        elif event.type == GC.Outbound_Truck_Order_Arrival:
            # Check if a consolidation lane is available
            if any([consolidation.available_cons_lane() for consolidation in self.consolidation.values()]):
                return True
            else:
                return False
    
    def check_truck_queues(self, joblist, OUTBD, time, dock=None, cons_lane=None):
        """ Upon calling, check dock queue, truck queue and then consolidation queue for trucks waiting """
        # If no trucks are waiting in either one of the queues, abort function and do nothing
        if (len(self.dock_queue) + len(self.truck_queue) + len(self.cons_queue)) == 0:
            return
        
        # If a dock is given that became available
        elif dock != None:
            # Check if there are trucks waiting for a dock at this location
            dock_waits = [truck for truck in self.dock_queue if truck.cons_assigned.area[-1] == dock.dock_set.id]
            if len(dock_waits) > 0:
                # Retrieve first truck from dock_waits and remove from queue and let artificially arrive
                waiting_truck = dock_waits.pop(0)
                self.dock_queue.remove(waiting_truck)
                waiting_truck.outbd_truck_arrival(dock=dock)
                
                # Check if all products are already in consolidation
                if waiting_truck.cons_ready != False:
                    # Create truck loading job in joblist
                    joblist.create_job(time=time, 
                                       job_type=GC.Job_Load_Truck, 
                                       truck=waiting_truck,
                                       cons_lane=waiting_truck.cons_assigned)
                return
            
            # Check if consolidation lane is also available connected to the given dock
            elif dock.dock_set.connected_consolidation.available_cons_lane() and len(self.truck_queue) > 0:
                # Retrieve first truck from truck queue and retrieve available consolidation lane
                waiting_truck = self.truck_queue.pop(0)
                cons_lane = dock.dock_set.connected_consolidation.return_available_lane()
                
                # Let the waiting truck (artificially) arrive
                waiting_truck.arrival(dock=dock, cons_lane=cons_lane)
                
                # If truck concerns inbound, create job for deloading the arrived truck
                if waiting_truck.type == 'INB':
                    joblist.create_job(time=time,
                                       job_type=GC.Job_Deload_Truck,
                                       truck=waiting_truck,
                                       cons_lane=cons_lane)
                
                # If truck concerns outbound
                else:
                    # Initialize truck order list
                    OUTBD.truck_order_arrival(time=time, truck=waiting_truck)
                    
                    # Create first product retrieval job
                    joblist.create_job(time=time,
                                       job_type=GC.Intermediate_Job_Storage2Cons,
                                       truck=waiting_truck,
                                       cons_lane=cons_lane)
                return
            
        # If consolidation lane is given to have become available and outbound orders are waiting for consolidation lane
        if cons_lane != None and len(self.cons_queue) > 0:
            # Retrieve first truck waiting, let artificially arrive and assign to consolidation lane
            waiting_truck_order = self.cons_queue.pop(0)
            OUTBD.truck_order_arrival(time=time, truck=waiting_truck_order)
            waiting_truck_order.outbd_order_arrival(cons_lane=cons_lane)
            
            # Initialize first product retrieval job
            joblist.create_job(time=time,
                               job_type=GC.Intermediate_Job_Storage2Cons,
                               truck=waiting_truck_order,
                               cons_lane=cons_lane)
            
        
    def create_storage_areas(self, excel, dimensions):
        """ Create Storage areas using the given layouts and dimensions """
        # Initialize storage area dictionary
        storage_areas = {}
        
        # Loop over all layouts and generate storage areas and dimensions matrix
        for SA in range(1, 18):
            sa = pd.read_excel('Simulation_Model/Data/Storage_Areas.xlsx', f'SA_{SA}', index_col=0)
            storage_areas[SA] = Storage_Area(area_nr=SA, 
                                             layout=sa, 
                                             storage_dimensions=dimensions,
                                             corner_points=GC.corner_list_full[SA-1])
        
        # Return created storage areas dictionary
        return storage_areas
    
    def create_truck_docks(self):
        """ Create truck docks using the number of truck docks in GC """
        return {hall: Truck_Docks(dock_id=hall, nr_of_docks=GC.number_of_docks[hall]) for hall in GC.number_of_docks.keys()}
    
    def create_consolidation(self):
        """ Create consolidation areas using the consolidation dimensions in GC """
        return {hall: Consolidation(consolidation_nr=hall, lanes=GC.cons_lanes[hall]) for hall in GC.cons_lanes.keys()}
    
    def connect_docks_and_cons(self):
        """ Connect truck docks to consolidation areas and consolidation areas to truck docks """
        # Loop over all storage halls (all having one set of consolidation areas and one set of docks)
        for hall in self.consolidation.keys():
            # Retrieve consolidation object and docks object
            cons = self.consolidation[hall]
            docks = self.docks[hall]
            
            # Set the references of both objects
            cons.connected_docks = docks
            docks.connected_consolidation = cons
            
            # Set consolidation area reference to all docks in the dock area
            for dock in docks.docks:
                dock.consolidation_area = cons
    
    def retrieve_storage_location(self, area, section, lane, level=None):
        """ Retrieve the storage location object from the storage class instance """
        if level == None:
            return self.areas[area].storage_sections[section]['storage'][lane]
        else:
            return self.areas[area].storage_sections[section]['storage'][lane].levels[level]    
    
    def storage_unoccupied(self, area, location):
        """ Return True if the area is available (/unoccupied) and False if it is occupied """
        # If Block storage
        if len(location) == 2:
            return True if self.retrieve_storage_location(area, location[0], location[1]).product == None else False
        # If B2B or Shuttle Storage
        else:
            return True if self.retrieve_storage_location(area, location[0], location[1], location[2]).product == None else False
    
    def find_closest_storage(self, cons, dist_2_crnr, sa, corner, storage_type, width, height):
        """ Find and return the closest unoccupied storage location object """ 
        # Retrieve locations in the storage area with similar dimensions
        # print(f'Storage: {storage_type}, Width: {width}, Height: {height}')
        if storage_type == 'BLK' and width == 1.4:
            locs = sa.dimension_dist[storage_type][1.6]
        elif storage_type == 'BLK':
            locs = sa.dimension_dist[storage_type][width]
        elif storage_type == 'SHT' and width == 1.2 and height == 2.3:
            locs = sa.dimension_dist[storage_type][1.4][height]
        elif storage_type == 'SHT' and width == 1.6 and height == 2.3:
            locs = sa.dimension_dist['BLK'][width]
        else:
            locs = sa.dimension_dist[storage_type][width][height]
        
        # If there are locations of the given size dimensions
        if len(locs) > 0:
        
            # Loop over all first locations of sections to find the closest
            locations, distances = [], []
            for loc in locs:
                if self.storage_unoccupied(area=sa.nr, location=loc):
                    # Append location to locations list
                    locations.append(loc)
                    
                    # If consolidation area falls within storage area under review, calculate consolidation to storage
                    if sa.nr in GC.cons_in_area[cons.nr[-1]].keys():
                        distances.append(self.dist.calc_cons2storage_inarea(cons=cons, sa=sa, loc=loc))
                        
                    # If not, calculate cornerpoint to storage
                    else:
                        distances.append(dist_2_crnr + self.dist.calc_crnr2storage(corner=corner, sa=sa, loc=loc))
            
            # # If available storage locations are found, return closest storage location
            if len(locations) > 0:
                
                return locations[np.argmin(np.array(distances))], min(distances)
        
        # If no (available) locations found, return None
        return None
    
    def search_corner(self, dist_2_crnr, cons, corner_point, storage_type, width, height, width2=None, height2=None, width3=None, height3=None, return_dist=False):
        """ Look through all adjacent storage areas to the given cornerpoint and return the closest found storage location """
        # Initialize monitoring lists
        found_storage = []
        distances = []
        
        # Loop over all adjacent corner points
        for sa_nr in GC.corner_point_lookup[corner_point].keys():  
            
            # Retrieve corner of the given corner-point in the area under consideration
            area_corner = GC.corner_point_lookup[corner_point][sa_nr]
                
            # Retrieve closest area storage
            closest_storage = self.find_closest_storage(cons=cons,
                                                        dist_2_crnr=dist_2_crnr,
                                                        sa=self.areas[sa_nr], 
                                                        corner=area_corner, 
                                                        storage_type=storage_type, 
                                                        width=width,
                                                        height=height)
            
            # If location is found, append closest storage and distance to monitoring lists
            if closest_storage != None:
                found_storage.append((sa_nr, closest_storage[0]))
                distances.append(closest_storage[1])
        
        # If secondary values for both width and height are given, compare these locations as well
        if width2 != None and height2 != None:
            # Loop over all adjacent corner points
            for sa_nr in GC.corner_point_lookup[corner_point].keys():               
                # Retrieve corner of the given corner-point in the area under consideration
                area_corner = GC.corner_point_lookup[corner_point][sa_nr]
                    
                # Retrieve closest area storage
                closest_storage = self.find_closest_storage(cons=cons, 
                                                            dist_2_crnr=dist_2_crnr,
                                                            sa=self.areas[sa_nr], 
                                                            corner=area_corner, 
                                                            storage_type=storage_type, 
                                                            width=width2,
                                                            height=height2)
                
                # If location is found, append closest storage and distance to monitoring lists
                if closest_storage != None:
                    found_storage.append((sa_nr, closest_storage[0]))
                    distances.append(closest_storage[1])
                    
        # If tertiary values for both width and height are given, compare these locations as well
        if width3 != None and height3 != None:
            # Loop over all adjacent corner points
            for sa_nr in GC.corner_point_lookup[corner_point].keys():               
                # Retrieve corner of the given corner-point in the area under consideration
                area_corner = GC.corner_point_lookup[corner_point][sa_nr]
                    
                # Retrieve closest area storage
                closest_storage = self.find_closest_storage(cons=cons, 
                                                            dist_2_crnr=dist_2_crnr,
                                                            sa=self.areas[sa_nr], 
                                                            corner=area_corner, 
                                                            storage_type=storage_type, 
                                                            width=width3,
                                                            height=height3)
                
                # If location is found, append closest storage and distance to monitoring lists
                if closest_storage != None:
                    found_storage.append((sa_nr, closest_storage[0]))
                    distances.append(closest_storage[1])
            
        # If any storage location is returned, find the closest one to the corner-point
        if len(found_storage) > 0:
            if return_dist:
                return found_storage[np.argmin(np.array(distances))], min(distances)
            else:
                return found_storage[np.argmin(np.array(distances))]
        
        # If no location found, return None
        else:
            if return_dist:
                return None, None
            else:
                return None            
      
    def closest_cons2storage(self, cons, product, storage_type, return_dist=False):
        """ Find the closest storage area having suitable dimensions and is unoccupied """
        # Retrieve distances from consolidation to all corner-points
        corner_points, distances = np.array(list(self.dist.cons2crnr[cons.nr].keys())), np.fromiter(self.dist.cons2crnr[cons.nr].values(), dtype=int)
        
        # Arrange corner points in ascending order based on distances
        corner_points = corner_points[np.argsort(distances)]
        distances.sort()
        
        # Retrieve product width and height for (first) storage location search
        width1 = product.width
        height1 = product.height
        width2 = None
        height2 = None
        width3 = None
        height3 = None
        
        # Look for available storage location until one is found or all places are searched
        upscale_cost = 1
        while True:
            # First check storage areas where the consolidation is located
            
            # Loop through sorted corner points
            for idx, cp in enumerate(corner_points):
                
                if return_dist:
                    # Search through all storage areas adjacent to the concerning corner-point
                    closest_loc, dist = self.search_corner(dist_2_crnr=distances[idx],
                                                           cons=cons,
                                                           corner_point=cp, 
                                                           storage_type=storage_type, 
                                                           width=width1, 
                                                           height=height1, 
                                                           width2=width2, 
                                                           height2=height2, 
                                                           width3=width3, 
                                                           height3=height3,
                                                           return_dist=True)
                    
                    # If location is found, break loop
                    if closest_loc is not None:
                        return closest_loc, dist 
                
                else:
                    # Search through all storage areas adjacent to the concerning corner-point
                    closest_loc = self.search_corner(dist_2_crnr=distances[idx],
                                                     cons=cons,
                                                     corner_point=cp, 
                                                     storage_type=storage_type, 
                                                     width=width1, 
                                                     height=height1, 
                                                     width2=width2, 
                                                     height2=height2, 
                                                     width3=width3, 
                                                     height3=height3)
                    
                    # If location is found, break loop
                    if closest_loc is not None:
                        return closest_loc   
            
            # Retrieve upscaling dimensions for the given storage type, current dimensions and the current upscale cost
            if storage_type == 'BLK':
                # Width M does not exist in block storage, make width L
                if product.width == GC.widths['M'] or product.width == GC.widths['L']:
                    width = GC.widths['L']
                else:
                    width = GC.widths['S']
                    
                dims =  GC.dimension_scale_up[storage_type][(width)][upscale_cost]
                
            # Handle shuttle storage   
            elif storage_type == 'SHT':
                # So does not exist in shuttle storage, make Mo
                if product.width == GC.widths['S'] and product.height == GC.heights['O']:
                    width, height = GC.widths['M'], GC.heights['O']
                
                    dims = GC.dimension_scale_up[storage_type][(width, height)][upscale_cost]
                    
                # Lo does not exist in shuttle storage, return empty dimensions sequence
                elif product.width == GC.widths['L'] and product.height == GC.heights['O']:
                    dims = []
                
                # All other dimensions exist, return normally
                else: 
                    dims = GC.dimension_scale_up[storage_type][(product.width, product.height)][upscale_cost]
                    
            # All dimensions exist in B2B storage, return normally      
            else:
                dims = GC.dimension_scale_up[storage_type][(product.width, product.height)][upscale_cost]
            
            # If no upscaling is possible no more, return None
            if len(dims) == 0:
                if return_dist:
                    return None, None
                else:
                    return None
            
            # If upscale dimensions are found
            else:
                upscale_cost += 1
                if len(dims) == 1: 
                    width1, height1 = (dims[0], None) if storage_type == 'BLK' else (dims[0][0], dims[0][1])
                    width2, height2 = (None, None)
                    width3, height3 = (None, None)
                elif len(dims) == 2: 
                    width1, height1 = (dims[0], None) if storage_type == 'BLK' else (dims[0][0], dims[0][1])
                    width2, height2 = (dims[1], None) if storage_type == 'BLK' else (dims[1][0], dims[1][1])
                    width3, height3 = (None, None)
                elif len(dims) == 3: 
                    width1, height1 = (dims[0], None) if storage_type == 'BLK' else (dims[0][0], dims[0][1])
                    width2, height2 = (dims[1], None) if storage_type == 'BLK' else (dims[1][0], dims[1][1])
                    width3, height3 = (dims[2], None) if storage_type == 'BLK' else (dims[2][0], dims[2][1])
                else:
                    raise ValueError('More than 3 upscale dimensions found!!')
            
    def store_product(self, cons, product, storage_type, time, return_loc=False, fraction=1):
        """ Store product on the closest location """
        # If product is already stored in (several) location(s)
        if len(product.stored) > 0:

            # Initialize monitoring lists, loop over locations and store distance from cons for all available locations
            loc_list = []
            dist_list = []
            for loc in set(product.stored):
                # Retrieve storage location
                loc_obj = self.retrieve_storage_location_simplified(loc = loc)
                
                # If location available, append location and distance to list
                if loc_obj.type == storage_type and loc_obj.storage_available():
                    loc_list.append(loc_obj)
                    dist_list.append(self.dist.cons2storage(cons=cons, sa=self.areas[loc[0]], loc=loc[1]))
            
            # If one or more location(s) are found, return closest location
            if len(loc_list) > 0:
                # Retrieve storage location object of the closest storage location
                storage_location = loc_list[np.argmin(np.array(dist_list))]
                
                # Store product on the retrieved storage location and document storage time
                storage_location.store_product(product, time)
                
                # Document storage
                self.simres.document_product_storage(storage_type=storage_type, product=product, storage_loc=storage_location)
                
                # Record placement in area location occupation and break out of function
                self.areas[storage_location.area].locations_occupied[storage_type] += 1
                
                # If location is desired return location, otherwise return nothing to break out of function
                if return_loc:
                    if storage_type == 'BLK':
                        location = (storage_location.area, (storage_location.section, storage_location.nr))
                    else:
                        location = (storage_location.area, (storage_location.section, storage_location.lane, storage_location.nr))
                    return location, min(dist_list)
                else:
                    return
     
        # If no storage locations are given or found available, retrieve closest suitable storage location
        if return_loc:
            loc, dist = self.closest_cons2storage(cons=cons, 
                                                  product=product, 
                                                  storage_type=storage_type,
                                                  return_dist=True)
        else:
            loc = self.closest_cons2storage(cons=cons, 
                                            product=product, 
                                            storage_type=storage_type)
        
        # If no locations are to be found, document product to sink
        if loc == None:
            self.simres.unplaceable_products += 1
            # If desired, return location
            if return_loc:
                return None, None
            # If no location is desired to be returned, just simply terminate function
            else:
                return
        
        # Retrieve storage location object
        storage_location = self.retrieve_storage_location_simplified(loc)
                    
        # Store product on the retrieved storage location and document storage time
        storage_location.store_product(product, time, fraction)
        
        # Document storage
        self.simres.document_product_storage(storage_type=storage_type, product=product, storage_loc=storage_location)
    
        # Record placement in area lane and location occupation
        self.areas[loc[0]].lanes_occupied[storage_type] += 1
        self.areas[loc[0]].locations_occupied[storage_type] += 1
        
        # Check if new lane concerns block storage and adjust the number of storage locations according to the stack level of the product
        if storage_location.type == 'BLK':
            # If stack level of product is lower than the predefined empty stack level
            if product.stack_level < GC.empty_stack_level_BLK:
                # Decrease number of storage locations in the storage area object
                self.areas[loc[0]].storage_type_locations[storage_type] -= ((GC.empty_stack_level_BLK - product.stack_level) * self.areas[loc[0]].storage_sections[loc[1][0]]['depth'][0])
            
            # If stack level of product is higher than the predefined empty stack level
            elif product.stack_level > GC.empty_stack_level_BLK:
                # Increase number of storage locations in the storage area object
                self.areas[loc[0]].storage_type_locations[storage_type] += ((product.stack_level - GC.empty_stack_level_BLK) * self.areas[loc[0]].storage_sections[loc[1][0]]['depth'][0])
        
        # If desired, return location
        if return_loc:
            return loc, dist
    
    def retrieve_storage_location_simplified(self, loc):
        """ Simplified quick version of previous function """
        # Retrieve location object for block storage (as it does not have levels)
        if len(loc[1]) == 2:
            storage_location = self.retrieve_storage_location(area=loc[0], 
                                                              section=loc[1][0], 
                                                              lane=loc[1][1])
            
        # Retrieve location object for B2B ad Shuttle storage (as they do have levels)
        else:
            storage_location = self.retrieve_storage_location(area=loc[0], 
                                                              section=loc[1][0], 
                                                              lane=loc[1][1], 
                                                              level=loc[1][2])
        return storage_location
      
    
    def retrieve_product(self, product, location, consolidation, fraction=False):
        """ Retrieve product from the given storage location """
        # If product is not stored in the given storage location, raise ValueError
        if location not in product.stored:
            raise ValueError("Requested product for retrieval, not present in the given storage location!")
        
        # Retrieve class object of the storage location
        storage_location = self.retrieve_storage_location_simplified(location)
        
        # Initialize distance monitoring variable
        total_distance = 0
        
        # If product fraction is taken
        if fraction:
            # If storage location concerns B2B storage
            if storage_location.type == 'B2B':
                # Retrieve distance from location to consolidation
                dist = self.dist.storage2cons(loc=location[1],
                                              sa=self.areas[location[0]], 
                                              cons=consolidation)
                total_distance += dist
                
                # And retrieve distance from consolidation back to location
                dist = self.dist.cons2storage(cons=consolidation, 
                                              sa=self.areas[location[0]], 
                                              loc=location[1])
                total_distance += dist
                
                # Mutate quantity stored at location, if fraction 0, remove storage location
                product.pallet_fractions[product.stored.index(location)] -= GC.Broken_Pallet_Size
                if product.pallet_fractions[product.stored.index(location)] == 0:
                    storage_location.retrieve_product(product)
                    self.areas[storage_location.area].locations_occupied[storage_location.type] -= 1
                    self.areas[storage_location.area].lanes_occupied[storage_location.type] -= 1
            
            # If storage location concerns either SHT or BLK storage
            else:
                # Retrieve placement time of product at the given location
                placement_time = product.placement_times[product.stored.index(location)]
                
                # Take full pallet from the storage location and mutate occupied storage locations
                storage_location.retrieve_product(product)
                self.areas[storage_location.area].locations_occupied[storage_location.type] -= 1
                                
                # If storage location product is set to none, location is empty and lane occupation should be mutated
                if storage_location.product == None:
                    self.areas[storage_location.area].lanes_occupied[storage_location.type] -= 1
                    
                    # If storage concerns block storage
                    if storage_location.type == 'BLK':
                        
                        # If the stack level of the product is not identical to the empty lane stack level of BLK
                        if product.stack_level != GC.empty_stack_level_BLK:
                            
                            # Reset number of storage locations to the empty stack level quantity
                            if product.stack_level < GC.empty_stack_level_BLK:
                                # Increase number of storage locations in the storage area object
                                self.areas[storage_location.area].storage_type_locations['BLK'] += ((GC.empty_stack_level_BLK - product.stack_level) * self.areas[storage_location.area].storage_sections[storage_location.section]['depth'][0])
                        
                            # If stack level of product is higher than the predefined empty stack level
                            elif product.stack_level > GC.empty_stack_level_BLK:
                                # Decrease number of storage locations in the storage area object
                                self.areas[storage_location.area].storage_type_locations['BLK'] -= ((product.stack_level - GC.empty_stack_level_BLK) * self.areas[storage_location.area].storage_sections[storage_location.section]['depth'][0])
                
            
                # Calculate dist from loc to cons
                dist = self.dist.storage2cons(loc=location[1],
                                              sa=self.areas[location[0]], 
                                              cons=consolidation)
                total_distance += dist
            
                # Store product in B2B (3/4 pallet)
                loc, dist = self.store_product(cons=consolidation, 
                                               product=product, 
                                               storage_type='B2B', 
                                               time=placement_time, 
                                               return_loc=True, 
                                               fraction=1-GC.Broken_Pallet_Size)
                
                # If product is unplaceable, return average execution time (already documented in simres)
                if loc == None:
                    dist = GC.avg_cons2storage_time
                
                total_distance += dist
        
        # If no pallet fraction required, take full pallet from storage location
        else:
            # Check if location has full pallet size, otherwise raise ValueError
            if product.pallet_fractions[product.stored.index(location)] < 1:
                raise ValueError('Full pallet requested, but not present in the given storage location!')
            
            # Retrieve product from storage location
            storage_location.retrieve_product(product)
            self.areas[storage_location.area].locations_occupied[storage_location.type] -= 1
            
            # If storage location product is set to none, location is empty and lane occupation should be mutated
            if storage_location.product == None:
                self.areas[storage_location.area].lanes_occupied[storage_location.type] -= 1
                
                # If storage concerns block storage
                if storage_location.type == 'BLK':
                    # If the stack level of the product is not identical to the empty lane stack level of BLK
                    if product.stack_level != GC.empty_stack_level_BLK:
                        # Reset number of storage locations to the empty stack level quantity
                        if product.stack_level < GC.empty_stack_level_BLK:
                            # Increase number of storage locations in the storage area object
                            self.areas[storage_location.area].storage_type_locations['BLK'] += ((GC.empty_stack_level_BLK - product.stack_level) * self.areas[storage_location.area].storage_sections[storage_location.section]['depth'][0])
                    
                        # If stack level of product is higher than the predefined empty stack level
                        elif product.stack_level > GC.empty_stack_level_BLK:
                            # Decrease number of storage locations in the storage area object
                            self.areas[storage_location.area].storage_type_locations['BLK'] -= ((product.stack_level - GC.empty_stack_level_BLK) * self.areas[storage_location.area].storage_sections[storage_location.section]['depth'][0])
            
            # Calculate distance back to consolidation and add to total distance monitoring variable            
            dist = self.dist.storage2cons(loc=location[1],
                                          sa=self.areas[location[0]], 
                                          cons=consolidation)
            total_distance += dist
            
        # Return total distance calculated
        return total_distance
    
        
    def initialize_storage(self, inbound, fallback):
        """ Initially fill the storage with products, according to the predefined percentages and PPA """
        # Calculate number of lanes needed to be filled for initialization
        lanes_to_fill_B2B = math.floor(self.simres.nr_of_lanes['B2B'] * GC.init_B2B)
        lanes_to_fill_BLK = math.floor(self.simres.nr_of_lanes['BLK'] * GC.init_BLK)
        lanes_to_fill_SHT = math.floor(self.simres.nr_of_lanes['SHT'] * GC.init_SHT)
        
        # Create workable dictionary to keep track of lane occupation
        storage_occupation = {'B2B': {'to_fill': lanes_to_fill_B2B,
                                      'filled': 0},
                              'SHT': {'to_fill': lanes_to_fill_SHT,
                                      'filled': 0},
                              'BLK': {'to_fill': lanes_to_fill_BLK,
                                      'filled': 0}}      
        
        # Initialize index counter, set random seed and retrieve 500 products and pallet amounts to start with
        idx=0
        np.random.seed(42)
        product_list = np.random.choice(a=inbound.product_percentages['Product_ID'],
                                        size=10000, 
                                        replace=True,
                                        p=inbound.product_percentages['Percentage'])
        
        pallets = np.random.choice(a=list(range(1, 21)),
                                   size=10000, 
                                   replace=True,
                                   p=inbound.pallet_storage_init)
        
        # Keep initializing storage untill all storage types are filled to the desired amount
        while storage_occupation['B2B']['filled'] < storage_occupation['B2B']['to_fill'] or storage_occupation['SHT']['filled'] < storage_occupation['SHT']['to_fill'] or storage_occupation['BLK']['filled'] < storage_occupation['BLK']['to_fill']:
            

            # Pull a product and pallet amount from generated lists
            product = inbound.products.catalogue[product_list[idx%10000]]
            amount = pallets[idx%10000]
            order_list = [product] * amount
            
            # Increment index
            idx += 1
            
            # Loop over products and place according to the PPA
            for product in order_list:
                
                # Retrieve storage type using the Product Placement Algorithm
                storage_type = self.ppa.get_storage_type(product)
                                
                # Only store the product if the storage location is not (yet) filled till the desired amount calculated
                if storage_occupation[storage_type]['filled'] < storage_occupation[storage_type]['to_fill']:
                                        
                    # Pick random consolidation (Speed Up)
                    consolidation = self.consolidation[np.random.choice(['A', 'B', 'C', 'D', 'H'])]
                                        
                    # Store products in the closest storage area looking from the consolidation area the truck is in
                    self.store_product(cons=consolidation, 
                                       product=product, 
                                       storage_type=storage_type, 
                                       time=0)
                
                # Fallback, if storage is filled, place products in the storage area that has the lowest % filling
                elif fallback:
                    # Pick random consolidation (Speed Up)
                    consolidation = self.consolidation[np.random.choice(['A', 'B', 'C', 'D', 'H'])]
                    
                    # Retrieve storage type with the lowest filling
                    storage_type_replacement = list(storage_occupation.keys())[np.argmax([storage_occupation[k]['to_fill'] - storage_occupation[k]['filled'] for k in storage_occupation.keys()])]

                    
                    # Store products in the closest storage area looking from the consolidation area the truck is in
                    self.store_product(cons=consolidation, 
                                       product=product, 
                                       storage_type=storage_type_replacement, 
                                       time=0)
            
            # Update lane occupation amounts
            for storage_type in storage_occupation.keys():
                                
                storage_occupation[storage_type]['filled'] = sum([storage_area.lanes_occupied[storage_type] for storage_area in self.areas.values()])
            
            # If length of product list is exceeded, generate new list
            if idx % 10000 == 0:
                product_list = np.random.choice(a=inbound.product_percentages['Product_ID'],
                                        size=10000, 
                                        replace=True,
                                        p=inbound.product_percentages['Percentage'])
        
                pallets = np.random.choice(a=list(range(1, 21)),
                                           size=10000, 
                                           replace=True,
                                           p=inbound.pallet_storage_init)
                
            
        # Make snapshot of storage occupation
        self.simres.document_storage_stats(time=0, 
                                           storage_areas=self.areas)
        