import numpy as np

from Simulation_Model import GC
from Simulation_Model.Simulation_Backend.Event import Event

class Joblist:
    """
    Description: Manage all the jobs that have to be done based on a (priority) deadline
    Input:  - resource_fleet: Resource class containing all resources
            - fes: Future Event Set, used to simulate the environment
            - storage: Storage class containing all information regarding the warehouse
    """
    def __init__(self, resource_fleet, fes, storage):
        self.resources = resource_fleet
        self.fes = fes
        self.storage = storage
        self.dist = storage.dist
        self.jobs = []
        self.job_cntr = 0
    
    def __str__(self):
        """ Print current joblist sorted based on deadline """
        message = f'The Joblist currently contains {len(self.jobs)} jobs'
        if len(self.jobs) > 0:
            sorted_jobs = np.array(self.jobs)[np.argsort(np.array([job.deadline for job in self.jobs]))]
            message += ', namely: \n'
            for job in sorted_jobs:
                message += 'Job: {:>20} | Deadline: {:>10} \n'.format(GC.JOB_NAMES[job.type],  round(job.deadline,4))
        
        return message
    
    def is_empty(self):
        """ Check if there are no jobs left to execute """
        return len(self.jobs) == 0
    
    def next(self, resource):
        """ Return the next event """
        # If joblist is empty, return no next job
        if self.is_empty():
            return None
        
        else:        
            # Retrieve list of jobs executable by resource
            executable_jobs = [job for job in self.jobs if job.skill_needed[resource.skill_index] == 1]
            
            # Check if there are executable jobs for the given resource
            if len(executable_jobs) > 0:
                # Retrieve deadlines of the executable jobs
                deadlines = np.array([job.deadline for job in executable_jobs])
                
                # Find the job with the earliest deadline
                nxt_job = executable_jobs[np.argmin(deadlines)]
                
                # Remove job from joblist and return job object
                self.jobs.remove(nxt_job)
                return nxt_job
            
            # If no jobs are left to be executed by the given resource, return no next job
            else:
                return None
    
    def create_job(self, time, job_type, truck, cons_lane, location=None, deadline=None):
        """ Create a job of the given type  """
        # If job type is consolidation to storage
        if job_type == GC.Intermediate_Job_Cons2Storage:
            # Retrieve product to store
            cons_lane = cons_lane
            product = cons_lane.stored[-1]
            
            # Retrieve storage type using PPA
            storage_type = self.storage.ppa.get_storage_type(product)
            
            # Adjust job type to the correct type
            if storage_type == 'B2B':
                job_type = GC.Job_Cons_2_B2B
            elif storage_type == 'SHT':
                job_type = GC.Job_Cons_2_SHT
            else:
                job_type = GC.Job_Cons_2_BLK
                
            # Set job deadline to given deadline (initial time + 2 hrs.)
            job_deadline = deadline
        
        # If job is storage to consolidation
        elif job_type == GC.Intermediate_Job_Storage2Cons:
            # Retrieve product to take from storage and its oldest storage location
            product, quantity = truck.orderlist_W[0]
            location = truck.retrieval_locations[0]
            
            # If product was not found, make job to SHT as this has separate execution
            if product == 'N/A':
                job_type = GC.Job_SHT_2_Cons
            
            # If product was found, retrieve storage type
            else:
                # Adjust job type to the correct type 
                storage_location = self.storage.retrieve_storage_location_simplified(loc=location)
                if storage_location.type == 'B2B':
                    job_type = GC.Job_B2B_2_Cons
                elif storage_location.type == 'SHT':
                    job_type = GC.Job_SHT_2_Cons
                else:
                    job_type = GC.Job_BLK_2_Cons
                
            # Set job deadline to truck arrival time
            job_deadline = truck.arrival_time
        
        # If job is either deloading or loading a truck
        elif job_type == GC.Job_Deload_Truck or job_type == GC.Job_Load_Truck:
            # Set job deadline to truck arrival time
            job_deadline = truck.arrival_time
        
        # Create job object
        new_job = Job(job_nr=self.job_cntr, 
                      job_type=job_type, 
                      creation_time=time, 
                      deadline=job_deadline,
                      truck=truck,
                      cons_lane=cons_lane,
                      location=location)
        
        # Increment job counter
        self.job_cntr += 1
        
        # Check if job can be executed immediately
        self.job_execution(job=new_job, time=time)
        
        # If no resource is found available for immediate exeuction, add job to joblist
        if new_job.start_execution == None:
            self.jobs.append(new_job)
    
    def check_resources_for_execution(self, job):
        """ Check if a resource is available for execution of the newly created job """
        # Loop through the resource types in the predefined allocation order        
        for resource_type in GC.resource_allocation_order:
            
            # Check if the resource type can execute the given job
            if job.skill_needed[GC.resource_skill_idx[resource_type]] == 1:
                
                # If resource found, break from loop and return resource type
                if self.resources.resource_available(resource_type=resource_type):
                    return resource_type
                
        # If no available resource is found, return None
        return None
    
    def job_execution(self, job, time, resource=None):
        """ Check if the job can be executed immediately """
        # If not given, check if a resource is available for the job and retrieve which type to use
        if resource == None:
            resource_type = self.check_resources_for_execution(job=job)
        
        # If resource is available
        if resource != None or resource_type != None:
            
            # If not given, retrieve resource object to execute job and the total job execution time
            if resource != None:
                resource, execution_time = self.calculate_job_execution_time(job=job,
                                                                             current_time=time,
                                                                             resource=resource)
            else:
                resource, execution_time = self.calculate_job_execution_time(job=job,
                                                                             current_time=time,
                                                                             resource_type=resource_type)
            
            # Assign resource to job, start execution and document execution time
            job.assigned_resource = resource           
            job.start_execution = time
            
            # Set resource occupation and document occupation time (if not in warm up time)
            resource.occupied = True
            if time > GC.warmup_time:
                resource.add_occupation_time(time = execution_time)
            
            # Create completion event
            if job.type == GC.Job_Deload_Truck:
                new_event = Event(typ=GC.Deload_Truck_Complete, 
                                  cur_time=time,
                                  time=time+execution_time,
                                  truck=job.truck, 
                                  cons_lane=job.cons_lane,
                                  job=job)
                
            elif job.type == GC.Job_Cons_2_B2B or job.type == GC.Job_Cons_2_SHT or job.type == GC.Job_Cons_2_BLK:
                new_event = Event(typ=GC.Product2Storage_Complete,
                                  cur_time=time,
                                  time=time+execution_time,
                                  truck=job.truck, 
                                  cons_lane=job.cons_lane,
                                  job=job)
            
            elif job.type == GC.Job_B2B_2_Cons or job.type == GC.Job_SHT_2_Cons or job.type == GC.Job_BLK_2_Cons:
                new_event = Event(typ=GC.Product2Consolidation_Complete, 
                                  cur_time=time,
                                  time=time+execution_time,
                                  truck=job.truck, 
                                  cons_lane=job.cons_lane,
                                  job=job)
                
            elif job.type == GC.Job_Load_Truck:
                new_event = Event(typ=GC.Load_Truck_Complete, 
                                  cur_time=time,
                                  time=time+execution_time,
                                  truck=job.truck, 
                                  cons_lane=job.cons_lane,
                                  job=job)
                
            else:
                raise ValueError('Job type {job.type} not recognized!!')   
            
            # Add newly created event to future event set (FES)
            self.fes.add(new_event)    
            
            
    def calculate_job_execution_time(self, job, current_time, resource_type=None, resource=None):
        """ Calculate the time it takes to complete a given job """        
        # If job consists of deloading truck
        if job.type == GC.Job_Deload_Truck: 
            # Retrieve truck, dock and consolidation object from job object
            truck = job.truck
            consolidation = truck.cons_assigned.consolidation_object
            
            # If resource is given, only calculate distance to location
            if resource != None:
                dist2cons = self.resources.return_resource_to_loc(resource=resource, 
                                                                  consolidation=consolidation)
            
            # Retrieve closest available resource if only resource type is given
            elif resource_type != None:
                resource, dist2cons = self.resources.retrieve_closest_available_resource(resource_type=resource_type,
                                                                                         consolidation=consolidation)
            
            # If both resource and resource type are none, raise ValueError
            else:
                raise ValueError("No Resource is given for job!")
            
            # Retrieve distance to travel from dock to consolidation area
            dist_dock2cons = self.dist.retrieve_dock2cons(dock=truck.dock_assigned.id)
                        
            # Calculate total distance to travel and total execution time
            total_distance = dist2cons + (truck.pallets * 2 - 1) * dist_dock2cons
            total_time = total_distance / resource.speed + truck.pallets * GC.truck_deload_load_time
            
            # Update resource location beforehand
            resource.location = consolidation.nr
            
            # Return resource and the job execution time
            return resource, total_time

        # If job implies taking products from consolidation to storage (B2B or BLK)
        elif job.type == GC.Job_Cons_2_B2B or job.type == GC.Job_Cons_2_BLK:
            # Retrieve truck, dock and consolidation object from job object
            cons_lane = job.cons_lane
            consolidation = self.storage.consolidation[cons_lane.area[-1]]
            
            # If resource is given, only calculate distance to location
            if resource != None:
                dist2cons = self.resources.return_resource_to_loc(resource=resource, 
                                                                  consolidation=consolidation)
            
            # Retrieve closest available resource if only resource type is given
            elif resource_type != None:
                resource, dist2cons = self.resources.retrieve_closest_available_resource(resource_type=resource_type,
                                                                                         consolidation=consolidation)
            # If both resource and resource type are none, raise ValueError
            else:
                raise ValueError("No Resource is given for job!")
            
            # Retrieve storage type from the job type
            if job.type == GC.Job_Cons_2_B2B:
                storage_type = 'B2B'
            else:
                storage_type = 'BLK'
            
            # Take last product in consolidation lane list to store (as this will be the first product in line)
            product = cons_lane.stored[0]
            
            # Store product and return location and distance to travel
            storage_location, distance = self.storage.store_product(cons=consolidation,
                                                                    product=product,
                                                                    storage_type=storage_type,
                                                                    time=current_time,
                                                                    return_loc=True)
            
            # If product is unplaceable, return average execution time (already documented in simres)
            if storage_location == None:
                return resource, GC.avg_cons2storage_time
            
            # Calculate total distance to travel and total execution time
            total_distance = dist2cons + distance
            total_time = total_distance / resource.speed + 2 * GC.pick_drop_time
            
            # Update resource location beforehand
            resource.location = storage_location
            
            # Return resource and the job execution time
            return resource, total_time
        
        
        # If job implies taking products from consolidation to storage (SHT) 
        elif job.type == GC.Job_Cons_2_SHT:
            # Retrieve truck, dock and consolidation object from job object
            cons_lane = job.cons_lane
            consolidation = self.storage.consolidation[cons_lane.area[-1]]
            
            # Initialize distance to mol to be 0
            dist2mol = 0
            
            # If resource not given, pick first available
            if resource == None:
                resource = [rtp for rtp in self.resources.fleet[GC.Reachtruckplus] if rtp.occupied==False][0]
            
            # If reachtruck+ is not holding its mol
            if not resource.holding_mol:
                # If reachtruck+ and mol are still in the same location, take mol (with no distance)
                if resource.location == resource.mol.location:
                    resource.holding_mol = True
                    
                # If reachtruck+ and mol are not in the same location, drive towards mol and retrieve it
                else:
                    # If reachtruck+ is at a consolidation area
                    if resource.location[:4] == 'CONS':
                        dist2mol = self.storage.dist.cons2storage(cons=self.storage.consolidation[resource.location[-1]],
                                                                  sa=self.storage.areas[resource.mol.location[0]],
                                                                  loc=resource.mol.location[1])
                        
                    # If reachtruck+ is at a storage location
                    else:
                        dist2mol = self.dist.retrieve_loc2loc_dist(loc1=resource.location,
                                                                   loc2=resource.mol.location)
                    
                    # Set resource holding mol indicator to True
                    resource.holding_mol = True
            
            # Take last product in consolidation lane list to store (as this will be the first product in line)
            product = cons_lane.stored[0]

            
            # Store product and return location and distance to travel
            storage_location, cons2storage = self.storage.store_product(cons=consolidation,
                                                                        product=product,
                                                                        storage_type='SHT',
                                                                        time=current_time,
                                                                        return_loc=True)
            
            # If product is unplaceable, return average execution time (already documented in simres)
            if storage_location == None:
                return resource, GC.avg_cons2storage_time
            
            # Calculate distance from storage location to consolidation
            storage2cons = self.dist.storage2cons(loc=storage_location[1], 
                                                  sa=self.storage.areas[storage_location[0]], 
                                                  cons=consolidation)
            
            # If resource was holding mol, calculate distance from resource location to storage location
            if resource.holding_mol:
                mol2storage = self.resources.return_resource_to_loc(resource=resource, 
                                                                    consolidation=consolidation)
                
            # If resource was not holding mol, calculate distance from mol location to storage location           
            else:
                mol2storage = self.dist.retrieve_loc2loc_dist(loc1=resource.mol.location,
                                                              loc2=storage_location)
                        
            # Calculate total distance to travel and total execution time
            total_distance = dist2mol + mol2storage + storage2cons + cons2storage
            total_time = total_distance / resource.speed + 2 * GC.pick_drop_time
            
            # Update resource (reach truck and mol) location beforehand and set holding mol to False
            resource.location, resource.mol.location = storage_location, storage_location
            resource.holding_mol = False
            
            # Return resource and the job execution time
            return resource, total_time
        
        # If job implies taking products from storage to consolidation (B2B or BLK)
        elif job.type == GC.Job_B2B_2_Cons or job.type == GC.Job_BLK_2_Cons:
            # Retrieve truck and consolidation object from job object
            truck = job.truck
            cons_lane = truck.cons_assigned
            consolidation = cons_lane.consolidation_object

            # Take first product from the (Working) orderlist
            (product, quantity), location = truck.pop_product_and_location()
            
            # If product no longer in working order list, set status to not reserved
            if product not in truck.prod_list:
                product.not_reserved = True
            
            # If resource is given, only calculate distance to location
            if resource != None:
                dist2loc = self.resources.return_resource_to_loc(resource=resource, 
                                                                 location=location)
            
            # Retrieve closest available resource if only resource type is given
            elif resource_type != None:
                resource, dist2loc = self.resources.retrieve_closest_available_resource(resource_type=resource_type,
                                                                                        location=location)
            
            # If both resource and resource type are none, raise ValueError
            else:
                raise ValueError("No Resource is given for job!")
            
            # Retrieve product and document retrieval distance
            distance = self.storage.retrieve_product(product=product, 
                                                     location=location, 
                                                     consolidation=consolidation,
                                                     fraction=(quantity % 1) > 0)
            
            # Calculate total distance to travel and total execution time
            total_distance = dist2loc + distance
            total_time = total_distance / resource.speed + 2 * GC.pick_drop_time            
            
            # Place product in consolidation lane
            cons_lane.place_product(product=product)
            
            # Update resource location beforehand
            resource.location = consolidation.nr
            
            # Return resource and the job execution time
            return resource, total_time
        
        # If job implies taking products from storage to consolidation (SHT)
        elif job.type == GC.Job_SHT_2_Cons:
            # Retrieve truck and consolidation object from job object
            truck = job.truck
            cons_lane = truck.cons_assigned
            consolidation = cons_lane.consolidation_object
            
            # Take first product from the (Working) orderlist
            (product, quantity), location = truck.pop_product_and_location()
            
            # If resource not given, pick first available
            if resource == None:
                resource = [rtp for rtp in self.resources.fleet[GC.Reachtruckplus] if rtp.occupied==False][0]
            
            # If product was not found, take average retrieval time
            if product == 'N/A':
                return resource, GC.avg_cons2storage_time
            
            # If product was found, act normally
            else:
                
                # If product no longer in working order list, set status to not reserved
                if product not in truck.prod_list:
                    product.not_reserved = True
                
                # Initialize distance to mol to be 0
                dist2mol = 0
                
                # If reachtruck+ is not holding its mol
                if not resource.holding_mol:
                    # If reachtruck+ and mol are still in the same location, take mol (with no distance)
                    if resource.location == resource.mol.location:
                        resource.holding_mol = True
                        
                    # If reachtruck+ and mol are not in the same location, drive towards mol and retrieve it
                    else:
                        # If reachtruck+ is at a consolidation area
                        if resource.location[:4] == 'CONS':
                            dist2mol = self.storage.dist.cons2storage(cons=self.storage.consolidation[resource.location[-1]],
                                                                      sa=self.storage.areas[resource.mol.location[0]],
                                                                      loc=resource.mol.location[1])
                            
                        # If reachtruck+ is at a storage location
                        else:
                            dist2mol = self.dist.retrieve_loc2loc_dist(loc1=resource.location,
                                                                       loc2=resource.mol.location)
                        
                        # Set resource holding mol indicator to True
                        resource.holding_mol = True
                
                # If resource was holding mol, calculate distance from resource location to storage location
                if resource.holding_mol:
                    mol2storage = self.resources.return_resource_to_loc(resource=resource, 
                                                                        location=location)
                    
                # If resource was not holding mol, calculate distance from mol location to storage location           
                else:
                    mol2storage = self.dist.retrieve_loc2loc_dist(loc1=resource.mol.location,
                                                                  loc2=location)
                
                # Leave mol at storage location
                resource.mol.location = location
                
                # Retrieve product and distance from storage location to consolidation
                storage2cons = self.storage.retrieve_product(product=product, 
                                                             location=location, 
                                                             consolidation=consolidation,
                                                             fraction=(quantity % 1) > 0)
                            
                # Calculate total distance to travel and total execution time (+ 10 sec for retrieval by mol)
                total_distance = dist2mol + mol2storage + storage2cons
                if dist2mol == 0 and mol2storage == 0:
                    total_time = total_distance / resource.speed + 2 * GC.pick_drop_time + GC.mol_retrieval_time
                else:
                    total_time = total_distance / resource.speed + 2 * GC.pick_drop_time + GC.mol_retrieval_time + 2 * GC.pick_drop_time
                
                # Place product in consolidation lane
                cons_lane.place_product(product=product)
                
                # Update resource (reach truck and mol) location beforehand and set holding mol to False
                resource.location = consolidation.nr
                resource.holding_mol = False
            
            return resource, total_time
        
        
        # If job implies loading an outbound truck
        elif job.type == GC.Job_Load_Truck:
            # Retrieve truck, dock and consolidation object from job object
            truck = job.truck
            cons_lane = truck.cons_assigned
            consolidation = cons_lane.consolidation_object
            
            # If resource is given, only calculate distance to location
            if resource != None:
                dist2cons = self.resources.return_resource_to_loc(resource=resource, 
                                                                  consolidation=consolidation)
            
            # Retrieve closest available resource if only resource type is given
            elif resource_type != None:
                resource, dist2cons = self.resources.retrieve_closest_available_resource(resource_type=resource_type,
                                                                                         consolidation=consolidation)
            
            # If both resource and resource type are none, raise ValueError
            else:
                raise ValueError("No Resource is given for job!")
            
            # Retrieve distance to travel from dock to consolidation area
            dist_dock2cons = self.dist.retrieve_dock2cons(dock=truck.dock_assigned.id)
            
            # Calculate total distance to travel and total execution time
            total_distance = dist2cons + (truck.pallets * 2 - 1) * dist_dock2cons
                        
            if truck.orderlines <= GC.orderlines_DL:
                total_time = GC.distance_direct_load / resource.speed + truck.pallets * GC.truck_deload_load_time_DL
            else:
                total_time = total_distance / resource.speed + truck.pallets * GC.truck_deload_load_time
                    
            # Update resource location beforehand
            resource.location = consolidation.nr
            
            # Return resource and the job execution time
            return resource, total_time
        
        else:
            raise ValueError("Job (type = {job.type}) passed through does not correspond to an existing job type")
    

class Job:
    """
    Description: Holds all information of a single job to be executed
    Input:  - job_nr: Index of the job (cumulative counter)
            - job_type: Type of job
            - creation_time: The time at which the job is created
            - deadline: Time jobs needs to be finished (determines priority)
            - truck: The truck object which the job concerns, to retrieve needed reference (if applicable)
            - cons_lane: The consolidation lane object which the job concerns, to retrieve needed reference (if applicable)
            - location: The location which is applicable for the job
    """
    def __init__(self, job_nr, job_type, creation_time, deadline, truck, cons_lane, location=None):
        self.nr = job_nr
        self.type = job_type
        self.creation_time = creation_time
        self.deadline = deadline
        self.truck = truck
        self.cons_lane = cons_lane
        self.location = location
        self.skill_needed = GC.job_needed_skill[self.type]
        self.assigned_resource = None
        self.start_execution = None
        self.end_execution = None
    
    def __str__(self):
        """ Print details of the job """
        if self.cons_lane != None:
            return f'JOB_{self.nr}, Type: {GC.JOB_NAMES[self.type]}, cons_lane: {self.cons_lane.area}_{self.cons_lane.nr}'
        else:
            return f'JOB_{self.nr}, Type: {GC.JOB_NAMES[self.type]}'




