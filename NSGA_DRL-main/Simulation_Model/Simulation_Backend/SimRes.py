import numpy as np

from Simulation_Model import GC

class Simulation_Results:
    """ 
    Description: Documents all simuation results during a simulation run
    """
    def __init__(self):
        self.sim_time = None
        
        # TRUCK VARIABLES
        self.outbound_time = []
        self.inbound_truck_departure = []
        
        # Storage variables
        self.snapshot_times = []
        self.occupied_locations = {'SHT': [],
                                   'B2B': [],
                                   'BLK': []}
        self.occupied_lanes = {'SHT': [],
                               'B2B': [],
                               'BLK': []}
        
        self.nr_of_lanes = {'SHT': 0, 'B2B': 0, 'BLK': 0}
        self.nr_of_locations = {'SHT': 0, 'B2B': 0, 'BLK': 0}
        self.BLK_locations = []
        
        self.unplaceable_products = 0
    
    def __str__(self):
        """ Print simulation results """
        if len(self.outbound_time) > 0:
            message = f'Outbound Truck Performance: \n'
            message += '--------------------------- \n'
            total_outbd_trucks = len(self.outbound_time)
            message += f'Perfect: {round(len([t for t in self.outbound_time if t < GC.OB_Truck_OK])/total_outbd_trucks, 3)}%    {len([t for t in self.outbound_time if t < GC.OB_Truck_OK])}/{total_outbd_trucks}\n'
            message += f'OK: {round(len([t for t in self.outbound_time if GC.OB_Truck_OK < t < GC.OB_Truck_Late])/total_outbd_trucks, 3)}%    {len([t for t in self.outbound_time if GC.OB_Truck_OK < t < GC.OB_Truck_Late])}/{total_outbd_trucks} \n'
            message += f'Late: {round(len([t for t in self.outbound_time if t >= GC.OB_Truck_Late])/total_outbd_trucks, 3)}%    {len([t for t in self.outbound_time if t >= GC.OB_Truck_Late])}/{total_outbd_trucks} \n'
            message += f'=========================================================================== \n'
            message += f'Average outbound processing time: {round(np.mean(self.outbound_time),3)} \n'
            message += f'Shortest outbound processing time: {round(min(self.outbound_time),3)} \n'
            message += f'Longest outbound processing time: {round(max(self.outbound_time),3)} \n'
            return message
        
        else:
            return f'No outbound trucks are completed yet, thus no results can be shown.'
    
    def print_stats(self, resource_fleet):
        """ Print all simulation statistics """
        print('\n\n\n\n\n\n')
        print(self)
        print('\n ====================================== \n')
        print('\n Resource Utilization: \n ---------------------------')
        self.retrieve_resource_utilization(resource_fleet)
        print('\n ====================================== \n')
        print('\n Storage Utilization: \n ---------------------------')
        self.retrieve_vulgraad_bezettingsgraad()
        print('\n ====================================== \n')
        print('\n Dimension Scale Ups: \n ---------------------------')
        for storage_type in self.dimension_scale_up.keys():
            print(f'{storage_type}: {self.dimension_scale_up[storage_type]}')
        print(f'\nNumber of unplaceable products: {self.unplaceable_products} \n')
    
    def retrieve_resource_utilization(self, resource_fleet):
        """ Show the resource utilization of all resources in the fleet """
        # Loop over all resource types in the fleet
        for typ in resource_fleet.keys():
            # Retrieve resource type and loop over all resources of this type
            resource_type = resource_fleet[typ][0].type
            for res in resource_fleet[typ]:
                # Calculate occupation fraction and print stats
                occ_fraction = res.time_occupied / self.sim_time
                print(f'{resource_type}_{res.nr}:  Busy: {round(occ_fraction,2)}%,  Idle: {round(1 - occ_fraction,2)}%')
            print('-' * 40)  
            
    def retrieve_vulgraad_bezettingsgraad(self):
        """ Calculate and return the vulgraad and bezettingsgraad """
        # Retrieve all inter measure times and add final timestep to complete the simulation
        inter_measure_times = [self.snapshot_times[idx] - self.snapshot_times[idx - 1] for idx in range(1, len(self.snapshot_times))]
        inter_measure_times.append(self.sim_time - self.snapshot_times[-1])
        
        # Calculate weighted average of occupied lanes and locations
        avg_occupied_lanes = {st : sum([inter_measure_times[idx] * measure for (idx, measure) in enumerate(self.occupied_lanes[st])]) / self.sim_time for st in self.occupied_lanes.keys()}
        avg_occupied_locations = {st : sum([inter_measure_times[idx] * measure for (idx, measure) in enumerate(self.occupied_locations[st])]) / self.sim_time for st in self.occupied_locations.keys()}
        avg_blk_locs = sum([inter_measure_times[idx] * nr_of_locs for (idx, nr_of_locs) in enumerate(self.BLK_locations)]) / self.sim_time    

        # Calculate vulgraad and bezettingsgraad using the calculated averages and total number of storage locations/lanes
        bezettingsgraad = {st: avg_occupied_lanes[st] / self.nr_of_lanes[st] for st in self.nr_of_lanes.keys()}
        vulgraad = {st: avg_occupied_locations[st] / avg_blk_locs if st == 'BLK' else avg_occupied_locations[st] / self.nr_of_locations[st] for st in self.nr_of_locations.keys()}
        
        # Print values for bezetting and vulgraad
        for st in bezettingsgraad.keys():
            print(f"---------- {st} Storage ----------")
            print(f'Vulgraad: {round(vulgraad[st],2)} \t Bezettingsgraad: {round(bezettingsgraad[st],2)}\n')

    def document_product_storage(self, storage_type, product, storage_loc):
        """ Document product storage """
        pass
    
    def document_storage_stats(self, time, storage_areas):
        """ Document the Vul-/Bezettingsgraad of all storage areas """
        # Set timestamps of measurements and initialize counting variable for BLK storage
        self.snapshot_times.append(time)
        BLK_locs = 0
                
        # Initialize counting dictionaries and loop over all storage areas
        full_locations = {'SHT': 0, 'B2B': 0, 'BLK': 0}
        full_lanes = {'SHT': 0, 'B2B': 0, 'BLK': 0}
        for a in storage_areas.keys():
            # Retrieve storage area object
            area = storage_areas[a]
            
            # Add number of BLK locations to counter
            BLK_locs += area.storage_type_locations['BLK']
            
            # Loop over all storage types and add the number of full locations and lanes
            for storage_type in area.locations_occupied.keys():
                full_locations[storage_type] += area.locations_occupied[storage_type]
                full_lanes[storage_type] += area.lanes_occupied[storage_type]
        
        # Append BLK locations to timestamp data
        self.BLK_locations.append(BLK_locs)
        
        # Append full locations and lanes to the class object lists
        for storage_type in area.locations_occupied.keys():
            self.occupied_locations[storage_type].append(full_locations[storage_type])
            self.occupied_lanes[storage_type].append(full_lanes[storage_type])
            
    def document_outbound_departure(self, truck):
        """ Document departure of an outbound truck, reviewing its performance """
        # Retrieve time it took from arrival to departure and add to total list
        processing_time = truck.departure_time - truck.arrival_time
        self.outbound_time.append(processing_time)
    
    def document_inbound_departure(self, truck):
        """ Document departure of an inbound truck """
        pass
#        self.inbound_truck_departure.append(truck)       
        
