from copy import deepcopy

from Simulation_Model import GC
from Simulation_Model.Algorithms import Product_Placement_Algorithm, Truck_Docking_Algorithm
from Simulation_Model.Storage import Storage
from Simulation_Model.Products import ProductCatalogue
from Simulation_Model.Resource import Resource_Fleet
from Simulation_Model.Joblist import Joblist
from Simulation_Model.Trucks import Inbound_Trucks, Outbound_Trucks
from Simulation_Model.Simulation_Backend.FES import FES
from Simulation_Model.Simulation_Backend.Event import Event
from Simulation_Model.Simulation_Backend.SimRes import Simulation_Results


class Simulate:
    """ 
    Description: Holds all functionality of the entire simulation 
    Input:  - sim_time: The simulation time for which truck arrivals are generated (simulation will complete all trucks initialized)
            - ppa: Rule order and values of the Product Placement Algorithm (PPA)
            - dims: The dimensions of the warehouse storage locations
            - res: The number of resources per resource type
            - storage_init_fallback: If set to True, the storage initialization will shifts storage type if the initial placement is already filled
            - truck_init: Future Event Set initial events, setting randomness of truck stuck over all consecutive runs
            - storage_init: Contains the storage area dictionary of the warehouse, can be used to maintain an identical storage initialization
            - products_init: Contains the product catalogue, NOTE: to be used in combination with storage init as they share product locations!
    """
    def __init__(self, sim_time, ppa, res, dims, storage_init_fallback=False, truck_init=None, storage_init=None, products_init=None):
        self.simulation_time = sim_time
        self.fes = FES()
        self.simres = Simulation_Results()
        
        self.products = ProductCatalogue(ExcelFile='Simulation_Model/Data/Product_List.xlsx')
        
        self.tda = Truck_Docking_Algorithm(products=self.products)
        self.INBD = Inbound_Trucks(ExcelFile='Simulation_Model/Data/Truck_Arrival.xlsx', products=self.products)
        self.OUTBD = Outbound_Trucks(ExcelFile='Simulation_Model/Data/Truck_Arrival.xlsx', products=self.products)
        self.snapshot_counter = 0
        
        if truck_init != None:
            self.fes.events = deepcopy(truck_init)
        else:
            self.truck_seed = self.initialize_truck_events()
        
        self.ppa = Product_Placement_Algorithm(rule_order=ppa[0],
                                               hist_outb=ppa[1][0],
                                               pallet_stored=ppa[1][1],
                                               stack_level1=ppa[1][2],
                                               pallet_height=ppa[1][3],
                                               stack_level2=ppa[1][4])
        
        self.storage = Storage(ExcelFile='Storage_Areas.xlsx',
                                   storage_dimensions=dims,
                                   ppa=self.ppa,
                                   tda=self.tda,
                                   simres=self.simres)
        
        # If initial storage and product catalogue is given
        if storage_init != None and products_init != None:
            self.storage.areas = deepcopy(storage_init)
            self.products.catalogue = deepcopy(products_init)
        
        # Else fill empty created product catalogue through storage initialization
        else:
            # Initialize storage in warehouse before simulation starts
            self.storage.initialize_storage(inbound=self.INBD, 
                                            fallback=storage_init_fallback)
            
            # Save storage and products seed
            self.storage_seed = deepcopy(self.storage.areas)
            self.products_seed = deepcopy(self.products.catalogue)
        
        self.resources = Resource_Fleet(nr_of_fork_lifts=res[0],
                                        nr_of_reach_trucks=res[1],  
                                        nr_of_reach_trucks_plus=res[2], 
                                        storage=self.storage)
        
        self.joblist = Joblist(resource_fleet=self.resources,
                               fes=self.fes,
                               storage=self.storage)
    
    def initialize_truck_events(self):
        """ Initialize all truck events and save random seeds for subsequent runs """
        # Initialize in- and outbound truck arrival events
        self.INBD.initialize_truck_arrivals(fes=self.fes, simulation_time=self.simulation_time)
        self.OUTBD.initialize_truck_arrivals(fes=self.fes, simulation_time=self.simulation_time)

        # Return FES to lock random seed for subsequent runs
        return deepcopy(self.fes.events)
    
    def terminate_simulation(self):
        """ Upon reaching the set simulation time, this function will correctly terminate all running processes. """
        # Retrieve full simulation time
        full_sim_time = (self.simulation_time + GC.warmup_time)
        
        # Loop through all docks of the system
        for hall in self.storage.docks.keys():
            for dock in self.storage.docks[hall].docks:
                
                # If dock contains assigned truck, and this truck concerns outbound
                if dock.truck_assigned != None and dock.truck_assigned.type == 'OUTB':
                    
                    # Check if the allowed departure time is exceeded, if yes document too late
                    truck = dock.truck_assigned
        
                    # IF truck is alrady too late, let truck artificially depart at simulation termination time
                    if truck.arrival_time + 2 <= full_sim_time:
                        truck.departure_time = full_sim_time
                        self.storage.simres.document_outbound_departure(truck)
        
        # Loop through all queues and check for late outbound trucks
        for truck in (self.storage.truck_queue + self.storage.dock_queue + self.storage.cons_queue):
            if truck.type == 'OUTB' and truck.arrival_time + 2 <= full_sim_time:
                
                # Artificially let truck depart at simulation termination time
                truck.departure_time = full_sim_time
                self.storage.simres.document_outbound_departure(truck)
    
    def RUN(self):
        """ Evaluate performance of the given configuration """
        # Retrieve first event from Future Event Set (FES) and update time
        event = self.fes.next()
        time = event.time
        
        # Keep running until termination criteria is reached
        while time <= (self.simulation_time + GC.warmup_time): 
            
            if event.type == GC.Inbound_Truck_Arrival:
                # Retrieve truck object from event class
                truck = event.truck
                
                # Transform product IDs into actual product objects
                truck.orderlist = [self.products.catalogue[prod] for prod in truck.orderlist]               
                
                # Check if a dock and consolidation combination is available
                if self.storage.truck_arrival_possible(event=event):
                    # Retrieve dock from Truck Docking Algorithm (TDA) and consolidation lane
                    dock = self.tda.get_truck_assignment(truck_orderlist=truck.orderlist,
                                                         truck_type=truck.type)
                    cons_lane = dock.dock_set.connected_consolidation.return_available_lane()
                    
                    # Dock truck and reserve consolidation area
                    truck.arrival(dock=dock, cons_lane=cons_lane)
                    
                    # Create job for truck deloading
                    self.joblist.create_job(time=time,
                                            job_type=GC.Job_Deload_Truck,
                                            truck=truck,
                                            cons_lane=cons_lane) 
                    
                # If not possible, place truck in truck queue
                else:
                    self.storage.truck_queue.append(truck)
                
                
            elif event.type == GC.Deload_Truck_Complete:
                # Retrieve truck and consolidation lane object
                truck = event.truck
                cons_lane = event.cons_lane
                
                # Transport all product from truck to consolidation area lane
                cons_lane.place_products(product_list=truck.orderlist)
                
                # Manage truck departure and check truck queues
                truck.departure(time=time, 
                                simres=self.simres)
                self.storage.check_truck_queues(joblist=self.joblist, 
                                                OUTBD=self.OUTBD, 
                                                time=time, 
                                                dock=truck.dock_assigned)
                
                # Calculate quality check execution time, create completion event and add to fes
                quality_check_execution_time = truck.pallets * GC.quality_check_time
                new_event = Event(typ=GC.Quality_Check_Done,
                                  cur_time=time,
                                  time=time + quality_check_execution_time, 
                                  cons_lane=cons_lane)
                self.fes.add(event=new_event)
            
                # Manage resource task completion correctly
                self.resources.manage_resource_task_completion(joblist=self.joblist,
                                                               resource=event.job.assigned_resource,
                                                               current_time=time)  
                
                
                
            elif event.type == GC.Quality_Check_Done:
                # Retrieve consolidation lane in which quality check is completed
                cons_lane = event.cons_lane
                
                # Create first cons to storage job and add to joblist
                self.joblist.create_job(time=time, 
                                        job_type=GC.Intermediate_Job_Cons2Storage,
                                        truck=None,
                                        cons_lane=cons_lane,
                                        deadline=time + GC.job_cons2storage_deadline_increment)
                
                
            elif event.type == GC.Product2Storage_Complete:
                # Retrieve consolidation lane in which quality check is completed
                cons_lane = event.cons_lane
                
                # Take first product from consolidation lane
                _ = cons_lane.take_product()
                
                # Check if consolidation lane still has products to be taken to storage
                if len(cons_lane.stored) > 0:
                    # Retrieve deadline of previous job
                    deadline_previous_job = event.job.deadline
                    
                    # Create new product to storage job
                    self.joblist.create_job(time=time, 
                                            job_type=GC.Intermediate_Job_Cons2Storage,
                                            truck=None,
                                            cons_lane=cons_lane,
                                            deadline=deadline_previous_job)
                    
                # If consolidation lane empty, check if another truck is waiting for a consolidation lane
                else:
                    self.storage.check_truck_queues(joblist=self.joblist, 
                                                    OUTBD=self.OUTBD, 
                                                    time=time, 
                                                    cons_lane=cons_lane)
                    
                # Manage resource task completion correctly
                self.resources.manage_resource_task_completion(joblist=self.joblist,
                                                               resource=event.job.assigned_resource,
                                                               current_time=time) 
                
            elif event.type == GC.Outbound_Truck_Order_Arrival:
                # Retrieve truck from event class object
                truck = event.truck
                
                # Check if a consolidation lane is available
                if self.storage.truck_arrival_possible(event=event):
                    # Create truck order list through the truck arrival function
                    self.OUTBD.truck_order_arrival(time=time, 
                                                   truck=truck)
                    
                    # Retrieve dock from Truck Docking Algorithm (TDA)
                    cons = self.tda.get_truck_assignment(truck_orderlist=truck.orderlist,
                                                         truck_type=truck.type,
                                                         truck_retrieval_locations=truck.retrieval_locations,
                                                         order_arrival=True)
                    
                    # Dock truck through the truck arrival event (this will NOT set arrival time!)
                    cons_lane = cons.return_available_lane()
                    truck.outbd_order_arrival(cons_lane=cons_lane)
                    
                    # Initialize first product to consolidation retrieval event
                    self.joblist.create_job(time=time, 
                                            job_type=GC.Intermediate_Job_Storage2Cons, 
                                            truck=truck,
                                            cons_lane=cons_lane)  
                
                # If not possible, place truck in consolidation queue
                else:
                    self.storage.cons_queue.append(truck)
                
            elif event.type == GC.Outbound_Truck_Arrival:
                # Retrieve truck from event class object
                truck = event.truck
                
                # Check if truck is already in consolidation queue
                if truck in self.storage.cons_queue:
                    # Take truck out of consolidation queue
                    self.storage.cons_queue.remove(truck)
                    
                    # Place truck in the first place of the truck queue
                    self.storage.truck_queue.insert(0, truck)
                
                # If truck concerns rush order
                elif truck.rush_order:
                                        
                    # Check if a dock and consolidation combination is available
                    if self.storage.truck_arrival_possible(event=event):
                        
                        # Create truck order list through the truck arrival function
                        self.OUTBD.truck_order_arrival(time=time, 
                                                       truck=truck)
                        
                        # Retrieve dock from Truck Docking Algorithm (TDA)
                        dock = self.tda.get_truck_assignment(truck_orderlist=truck.orderlist,
                                                             truck_type=truck.type,
                                                             truck_retrieval_locations=truck.retrieval_locations)
                        cons_lane = dock.consolidation_area.return_available_lane()
                        
                        # Dock truck through the truck arrival event (this will NOT set arrival time!)
                        truck.arrival(dock=dock, cons_lane=cons_lane)
                        
                        # Initialize first product to consolidation retrieval event
                        self.joblist.create_job(time=time, 
                                                job_type=GC.Intermediate_Job_Storage2Cons, 
                                                truck=truck,
                                                cons_lane=cons_lane)
                    
                    # If not possible, place truck in truck queue
                    else:
                        self.storage.truck_queue.append(truck)
                
                # If truck does not concern a rush order
                else:
                    # Retrieve dock for truck arrival
                    cons_lane = truck.outbd_truck_arrival()
                    
                    # If no dock is found (cons_lane == None), place truck in dock queue
                    if cons_lane == None:
                        self.storage.dock_queue.append(truck)
                
                    # If truck does not concern rush order, check if all product are already in the designated consolidation lane
                    elif truck.cons_ready:
                        # Create truck loading job in joblist
                        self.joblist.create_job(time=time, 
                                           job_type=GC.Job_Load_Truck, 
                                           truck=truck,
                                           cons_lane=cons_lane)
            
            
            
            elif event.type == GC.Product2Consolidation_Complete:
                # Retrieve truck object from event class object
                truck = event.truck
                cons_lane = event.cons_lane
                
                # If more products have to be picked to complete product to consolidation action
                if len(truck.orderlist_W) > 0:
                    self.joblist.create_job(time=time,
                                       job_type=GC.Intermediate_Job_Storage2Cons,
                                       truck=truck,
                                       cons_lane=cons_lane)
                
                # If all products are picked
                else:
                    # Set consolidation ready variable to current time in truck object
                    truck.cons_ready = time
                    
                    # If truck has already arrived
                    if truck.arrival_time < time and truck not in self.storage.dock_queue:
                        self.joblist.create_job(time=time, 
                                           job_type=GC.Job_Load_Truck, 
                                           truck=truck,
                                           cons_lane=cons_lane)
               
                # Manage resource task completion correctly
                self.resources.manage_resource_task_completion(joblist=self.joblist,
                                                               resource=event.job.assigned_resource,
                                                               current_time=time)  
                            
            elif event.type == GC.Load_Truck_Complete:
                # Retrieve truck object from event class object
                truck = event.truck
                cons_lane = event.cons_lane
                
                # Transport all products from consolidation lane to truck
                cons_lane.take_products()
                
                # Document truck departure
                truck.departure(time=time,
                                simres=self.simres)
                self.storage.check_truck_queues(joblist=self.joblist, 
                                                OUTBD=self.OUTBD, 
                                                time=time, 
                                                dock=truck.dock_assigned,
                                                cons_lane=cons_lane)
                
                # Manage resource task completion correctly
                self.resources.manage_resource_task_completion(joblist=self.joblist,
                                                               resource=event.job.assigned_resource,
                                                               current_time=time) 
            
            # Make a snapshot of the storage occupation every 5 hours
            if time > self.snapshot_counter * 5:
                self.simres.document_storage_stats(time=time, 
                                                   storage_areas=self.storage.areas)
                self.snapshot_counter += 1
            
            # If fes is empty, stop simulation
            if self.fes.is_empty() and self.joblist.is_empty():
                break
            
            # Retrieve next event from FES and update time
            event = self.fes.next()
            time = event.time
        
        # Handle simulation termination, through correctly closing all running processes
        self.terminate_simulation()
        
        # add total simulation time to simulation results
        self.simres.sim_time = time
