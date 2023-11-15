import numpy as np
import random

from Simulation_Model import GC

class Product_Placement_Algorithm:
    """  
    Description: Decide upon product placement, following the defined set of rules
    Input:  - Rule_Order: The order in which the rules are executed
            - Product_Catalogue: The product class recording all products and their movements
    """
    def __init__(self, rule_order, hist_outb, pallet_stored, stack_level1, pallet_height, stack_level2):
        self.order = rule_order
        self.hist_outb = hist_outb
        self.pallet_stored = pallet_stored
        self.stack_level1 = stack_level1
        self.pallet_height = pallet_height
        self.stack_level2 = stack_level2
    
    def get_storage_type(self, product):
        """ Loop over Product Placement Algorithm (PPA) rules in the predefined order """
        for rule in self.order:
            # If rule 1 should be executed
            if rule == 1:
                # print(f'Rule 1: Historic Outbound: {product.historic_outb} < {self.hist_outb} and Storage: {product.quantity_stored()} < {self.pallet_stored}')
                if product.historic_outb < self.hist_outb and product.quantity_stored() < self.pallet_stored:
                    return 'B2B'
                
            # If rule 2 should be executed
            elif rule == 2:
                # print(f'Rule 2: Stack Level: {product.stack_level} >= {self.stack_level1}') 
                if product.stack_level >= self.stack_level1:
                    return 'BLK'
                
            # If rule 3 should be executed
            elif rule == 3:
                # print(f'Rule 3: Height: {product.height} <= {self.pallet_height} and Stack Level: {product.stack_level} < {self.stack_level2}') 
                if product.height > self.pallet_height and product.stack_level >= self.stack_level2:
                    return 'BLK'
            
            # If rule 4 should be executed
            elif rule == 4:
                return 'SHT'
    
class Truck_Docking_Algorithm:
    """
    Description: Decide which dock will be used to (un)load a given truck
    Input:  - products: the product catalogue containing all products
            - storage: the storage class containing all needed information sources
    """
    def __init__(self, products):
        self.storage = None
        self.dist = None
        self.cat = products.catalogue
        self.storage_areas = None
        self.consolidation = None
        self.docks = None
    
    def get_truck_assignment(self, truck_orderlist, truck_type, truck_retrieval_locations=None, order_arrival=False):
        """ Retrieve the most preferable, available truck dock and consolidation area """
        # If not initialisation, retrieve all available dock - consolidation area combinations for further review
        if truck_type != 'INIT':
            # If only order arrival is required, only look at consolidation lanes
            if order_arrival:
                consolidation_review = [self.consolidation[hall] for hall in self.docks.keys() if self.storage.consolidation[hall].available_cons_lane()]
            
            # If not order arrival (but truck arrival) check for consolidation and dock combination
            else:
                consolidation_review = [self.consolidation[hall] for hall in self.docks.keys() if self.docks[hall].available_dock_and_cons(truck_type=truck_type)]

            # # If no consolidation-dock combination is available raise ValueError
            # if len(consolidation_review) == 0:
            #     print({hall: [lane.nr for lane in self.consolidation[hall].lanes if lane.truck_assigned == None] for hall in self.docks.keys()})
            #     print({hall: [dock.id for dock in self.docks[hall].docks if dock.truck_assigned == None] for hall in self.docks.keys()})
            #     raise ValueError('No Consolidation-Dock could be found available for docking the concerning truck')
        
        # If initialization, review all consolidation areas
        else:
            consolidation_review = [self.consolidation[hall] for hall in self.docks.keys()]

        # If inbound truck, retrieve all products of the truck's order list that are already stored
        if truck_type == 'INB' or truck_type == 'INIT':
            order_list = [p for p in truck_orderlist if len(p.stored) > 0]
            
            # If product is already stored
            if len(order_list) > 0:
                # Initialize total distance monitoring variable
                total_dists = []
                
                # Retrieve set of order list to remove duplicates, as quantity does not have to be taken into account
                order_set = set(order_list)
                
                # Loop over all consolidation areas under review
                for cons in consolidation_review:
                    # Initialize looing cumulative distance sum
                    total_dist = 0
                
                    # Loop over all products in the order set
                    for product in order_set:
                        # Retrieve set and distances to the consolidation area of (all) product location(s) of the given product
                        storage_distances = [self.dist.cons2storage(cons=cons, sa=self.storage_areas[p[0]], loc=p[1]) for p in set(product.stored)]
                
                        # Increment total_dist with the minimal distance found
                        total_dist += min(storage_distances)
                    
                    # Add total dist to the monitoring variable for later review
                    total_dists.append(total_dist)
                
                # Retrieve consolidation object of with the lowest total distance
                chosen_consolidation = consolidation_review[np.argmin(np.array(total_dists))]
            
            else:
                # If product is not already stored return random consolidation area
                chosen_consolidation = list(consolidation_review)[0]
            
            # If intialization, return consolidation area
            if truck_type == 'INIT':
                return chosen_consolidation
            
        # If outbound truck, look through all products and their locations
        else:
            # Retrieve storage locations to retrieve products from
            storage_locations = [loc for loc in truck_retrieval_locations if loc != 'N/A']
            
            # Initialize total distance monitoring variable
            total_dists = []
            
            # Loop over all consolidation areas under review
            for cons in consolidation_review:            
                # Retrieve set and distances to the consolidation area of (all) product location(s) of the given product
                total_dists.append(sum([self.dist.cons2storage(cons=cons, sa=self.storage_areas[p[0]], loc=p[1]) for p in storage_locations]))
                
            # Retrieve consolidation object of with the lowest total distance
            chosen_consolidation = consolidation_review[np.argmin(np.array(total_dists))]
            
            # If outbound order arrival, just return chosen consolidation
            if order_arrival:
                return chosen_consolidation
        
        # Retrieve dock connected to the chosen consolidation area
        chosen_dock = [dock for dock in chosen_consolidation.connected_docks.docks if dock.truck_assigned == None][GC.dock_assignment[chosen_consolidation.nr[-1]]]
         
        # Return the chosen dock
        return chosen_dock
            
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
