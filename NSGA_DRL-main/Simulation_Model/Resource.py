from Simulation_Model import GC
from Simulation_Model.Resources.Fork_Lift import Fork_Lift
from Simulation_Model.Resources.Reach_Truck import Reach_Truck
from Simulation_Model.Resources.Reach_Truck_Plus import Reach_Truck_Plus


class Resource_Fleet:
    """
    Description: Creates and manages functionality of all resources of the designed resource fleet
    Input:  - nr_of_fork_lifts: The number of fork lifts to be used
            - nr_of_reach_trucks: The number of reach trucks to be used
            - nr_of_reach_trucks_plus: The number of reach trucks + (mol) to be used
            - storage: Storage class containing all functionalities of the warehouse
    """
    def __init__(self, nr_of_fork_lifts, nr_of_reach_trucks, nr_of_reach_trucks_plus, storage):
        self.fleet = {GC.Forklift: [Fork_Lift(fork_lift_nr=idx) for idx in range(0, nr_of_fork_lifts)], 
                      GC.Reachtruck: [Reach_Truck(reach_truck_nr=idx) for idx in range(0, nr_of_reach_trucks)], 
                      GC.Reachtruckplus: [Reach_Truck_Plus(reach_truck_nr=idx) for idx in range(0, nr_of_reach_trucks_plus)]}
        self.storage = storage
    
    def resource_available(self, resource_type):
        """ Return True if a resource of the given type is available """
        return not all([res.occupied for res in self.fleet[resource_type]])
    
    def retrieve_closest_available_resource(self, resource_type, consolidation=None, location=None):
        """ Retrieve an available resource of the given resource type """
        # Retrieve all unoccupied resources of the given resource type
        available_resources = [r for r in self.fleet[resource_type] if r.occupied == False]
        
        # Return the first available resource or none if there aren't any
        if len(available_resources) > 0: 
            # Return closest resource to consolidation and its distance
            return self.return_closest_resource(resources=available_resources,
                                                consolidation=consolidation,
                                                location=location)
            
        # If no resources are found, return none    
        else: 
            return None, None
    
    def return_resource_to_loc(self, resource, consolidation=None, location=None):
        """ Retrieve distance of resource to travel to consolidation area or storage location """
        # If the locations a resource is desired concerns a consolidation area
        if consolidation != None:
            # If the resource is currently at a consolidation area
            if resource.location[:4] == 'CONS':
                # If resource is already at consolidation, return resource object and 0 distance
                if resource.location == consolidation.nr:
                    return 0
                
                # If not already at consolidation, retrieve distance to consolidation and append to monitoring list
                else:
                    # Calculate and return distance between resource location (consolidation area) and the destination (another consolidation)
                    return self.storage.dist.retrieve_loc2cons(corner_point=resource.location,
                                                               cons=consolidation)
                
            # If the resource is currently at a storage location
            else:
                # Retrieve storage area from the location
                sa = self.storage.areas[resource.location[0]]
                
                # Return distance from storage location to consolidation and append to monitoring list
                return self.storage.dist.storage2cons(loc=resource.location[1], 
                                                      sa=sa, 
                                                      cons=consolidation)
        
        # If the location a resource is desired concerns a storage location
        elif location != None:
            # If the resource is currently at a consolidation area
            if resource.location[:4] == 'CONS':
                # Retrieve consolidation class object
                res_cons = self.storage.consolidation[resource.location[-1]]
                
                # Return distance from consolidation to storage location and append to monitoring list
                return self.storage.dist.cons2storage(cons=res_cons, 
                                                      sa=self.storage.areas[location[0]],
                                                      loc=location[1])
            
            # If the resource is currently at a storage location
            else:
                # Return distance from storage location to storage location and append to monitoring list
                return self.storage.dist.retrieve_loc2loc_dist(loc1=resource.location, 
                                                               loc2=location)
                    
        # If both consolidation and location are None, raise error
        else:
            raise ValueError('Both Consolidation and Location are set to None, no location for retrieval')
    
    
    def return_closest_resource(self, resources, consolidation, location):
        """ Retrieve distance to travel to consolidation area or storage location """
        # Set monitoring list
        distances = []
        
        # If the locations a resource is desired concerns a consolidation area
        if consolidation != None:
            # Loop over all resources found by the retrieval function
            for res in resources:
                # If the resource is currently at a consolidation area
                if res.location[:4] == 'CONS':
                    # If resource is already at consolidation, return resource object and 0 distance
                    if res.location == consolidation.nr:
                        return res, 0
                    
                    # If not already at consolidation, retrieve distance to consolidation and append to monitoring list
                    else:
                        # Calculate distance between resource location (consolidation area) and the destination (another consolidation)
                        dist = self.storage.dist.retrieve_loc2cons(corner_point=res.location,
                                                                   cons=consolidation)
                        distances.append(dist)
                    
                # If the resource is currently at a storage location
                else:
                    # Retrieve storage area from the location
                    sa = self.storage.areas[res.location[0]]
                    
                    # Retrieve distance from storage location to consolidation and append to monitoring list
                    dist = self.storage.dist.storage2cons(loc=res.location[1], 
                                                          sa=sa, 
                                                          cons=consolidation)
                    
                    distances.append(dist)
        
        # If the location a resource is desired concerns a storage location
        elif location != None:
            # Loop over all resources found by the retrieval function
            for res in resources:
                # If the resource is currently at a consolidation area
                if res.location[:4] == 'CONS':
                    # Retrieve consolidation class object
                    res_cons = self.storage.consolidation[res.location[-1]]
                    
                    # Retrieve distance from consolidation to storage location and append to monitoring list
                    dist = self.storage.dist.cons2storage(cons=res_cons, 
                                                          sa=self.storage.areas[location[0]],
                                                          loc=location[1])
                    distances.append(dist)
                
                # If the resource is currently at a storage location
                else:
                    # Retrieve distance from storage location to storage location and append to monitoring list
                    dist = self.storage.dist.retrieve_loc2loc_dist(loc1=res.location, 
                                                                   loc2=location)
                    distances.append(dist)
                    
        # If both consolidation and location are None, raise error
        else:
            raise ValueError('Both Consolidation and Location are set to None, no location for retrieval')
        
#        # Return closest resource object and the distance to travel
#        return resources[np.argmin(distances)], min(distances)
        return resources[0], distances[0]
    
    
    def manage_resource_task_completion(self, joblist, resource, current_time):
        """ Manage resource completion correctly """        
        # Call next job function from joblist
        next_job = joblist.next(resource)
                
        # If job found to be executed by resource, start job
        if next_job != None:
            joblist.job_execution(job=next_job, 
                                  time=current_time, 
                                  resource=resource)
            
        # If no job found to be executed, set resource occupation to false
        else:
            resource.occupied = False
