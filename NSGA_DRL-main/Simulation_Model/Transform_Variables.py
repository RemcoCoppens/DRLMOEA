import numpy as np

from Simulation_Model import GC, GF
from Simulation_Model.Transform_Dimensions import B2B_lanes, SHT_lanes, BLK_lanes, reshape_dictionary

class Transform:
    """
    Description: Holds all information/functionality to make individual compatible with the simulation
    """
    def __init__(self):
        # Set PPA value Limitations
        self.range_historic_outbound = (100, 500)
        self.range_pallet_stored = (2, 20)
        self.range_stack_level = (1, 6)
        self.pallet_heights = (1.13, 1.66, 1.93, 2.30)
        
        # Set Resource amount limitations
        self.range_forklift = (1, 20)
        self.range_reachtruck = (1, 20)
        self.range_reachtruckplus = (1, 20)
    
    def transform_PPA_rule_order(self, values):
        """ Transform given values to a compatible format for a simulation run """
        return list(np.argsort(np.array(values)) + 1)
            
    def transform_PPA_values(self, values):
        """ Transform given values to a compatible format for a simulation run """
        # Transform the given continuous [0, 1] values to values on the set integer range
        hist_outb = int(self.range_historic_outbound[0] + values[0] * (self.range_historic_outbound[1] - self.range_historic_outbound[0]))
        pallet_stored = int(self.range_pallet_stored[0] + values[1] * (self.range_pallet_stored[1] - self.range_pallet_stored[0]))
        stack_level1 = int(self.range_stack_level[0] + values[2] * (self.range_stack_level[1] - self.range_stack_level[0]))
        stack_level2 = int(self.range_stack_level[0] + values[4] * (self.range_stack_level[1] - self.range_stack_level[0]))
        
        # Transform the given continuous [0, 1] value to one of the categories given
        pallet_height = self.pallet_heights[int(values[3] * 3)]
        
        # Return the PPA values in the correct order
        return [hist_outb, pallet_stored, stack_level1, pallet_height, stack_level2]
    
    def transform_resource_values(self, values):
        """ Transform given values to a compatible format for a simulation run """
        # Transform the given continuous [0, 1] values to values on the set integer range
        forklifts = int(self.range_forklift[0] + values[0] * (self.range_forklift[1] - self.range_forklift[0]))
        reachtrucks = int(self.range_reachtruck[0] + values[1] * (self.range_reachtruck[1] - self.range_reachtruck[0]))
        reachtruckplus = int(self.range_reachtruckplus[0] + values[2] * (self.range_reachtruckplus[1] - self.range_reachtruckplus[0]))
        
        # Return the number of resources
        return [forklifts, reachtrucks, reachtruckplus]
 
    def transform_warehouse_dimensions(self, values):
        """ Transform given values to a compatible format for a simulation run """
        # Disect the values into separate halls
        halls = {'A': list(values[0:18]), 
                 'B': list(values[18:34]), 
                 'C': list(values[34:50]), 
                 'D': list(values[50:68]), 
                 'H': list(values[68:84])}
        
        # Initialize looping dictionary 
        hall_dims = {}
        
        # Iterate over all halls and retrieve dimension dictionary
        for h in halls.keys():
            # Retrieve values
            vals = halls[h].copy()
            
            # If hall contains back-to-back storage
            if GC.hall_storage_type[h][GC.B2B]:
                # Retrieve and remove values concerning back-to-back storage
                b2b_vals = vals[:GC.vals_B2B]
                del vals[:GC.vals_B2B]
                
                # Normalize the values given for b2b dimensions
                b2b_vals = GF.create_fractions(b2b_vals)
                
                # Retrieve dimension distribution
                b2b_dims = B2B_lanes(section_widths=GC.section_widths[h]['B2B'], 
                                     Sc=b2b_vals[0], Sm=b2b_vals[1], So=b2b_vals[2], Mc=b2b_vals[3], Mm=b2b_vals[4], 
                                     Mo=b2b_vals[5], Lc=b2b_vals[6], Lm=b2b_vals[7], Lo=b2b_vals[8])
                
                # Append retrieved dimensions dictionary to monitoring hall dictionary
                GF.merge_dictionaries(hall_dims, b2b_dims)
            
            # If hall contains shuttle storage    
            if GC.hall_storage_type[h][GC.SHT]:
                # Retrieve and remove values concerning back-to-back storage
                sht_vals = vals[:GC.vals_SHT]
                del vals[:GC.vals_SHT]
                
                # Normalize the values given for b2b dimensions
                sht_vals = GF.create_fractions(sht_vals)
                
                # Retrieve dimension distribution
                sht_dims = SHT_lanes(section_sqm=GC.section_widths[h]['SHT'], 
                                     Sc=sht_vals[0], Sm=sht_vals[1], Mc=sht_vals[2], Mm=sht_vals[3], 
                                     Mo=sht_vals[4], Lc=sht_vals[5], Lm=sht_vals[6])
                
                # Append retrieved dimensions dictionary to monitoring hall dictionary
                GF.merge_dictionaries(hall_dims, sht_dims)
            
            # If hall contains block storage
            if GC.hall_storage_type[h][GC.BLK]:
                # Retrieve and remove values concerning back-to-back storage
                blk_vals = vals[:GC.vals_BLK]
                del vals[:GC.vals_BLK]
                
                # Normalize the values given for b2b dimensions
                blk_vals = GF.create_fractions(blk_vals)
                
                # Retrieve dimension distribution
                blk_dims = BLK_lanes(section_sqm=GC.section_widths[h]['BLK'], 
                                     S=blk_vals[0], L=blk_vals[1])
                
                # Append retrieved dimensions dictionary to monitoring hall dictionary
                GF.merge_dictionaries(hall_dims, blk_dims)
        
        # Finally transform the hall dimensions into separate storage area dimensions
        sa_dims = reshape_dictionary(dimensions=hall_dims)
        
        # Return the created dimensions
        return sa_dims
            