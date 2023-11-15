import numpy as np
import pickle

from Simulation_Model.Simulate import Simulate
from Simulation_Model.Transform_Variables import Transform
from Simulation_Model import GC

class Evaluate:
    """ 
    Description: Hold all information/functionality regarding the evaluation of an individual 
    Input:  - simulation_time: Simulation time used to evaluate the configuration performance
            - storage_fallback: If True storage initialization will be executed through the fallback mechanism
            - check_validity: If True, check individual's rule order to see if its 'valid' (1 before 4 and 2 or 3 before 4)
    """
    def __init__(self, simulation_time, storage_fallback=True, check_validity=True, use_truck_seed=True):
        self.simulation_time=simulation_time
        self.transform=Transform()
        self.storage_fallback = storage_fallback
        self.check_validity = check_validity
        self.use_truck_seed = use_truck_seed
        
        if use_truck_seed:
            self.truck_seed = self.load_truck_seed()
        else:
            self.truck_seed = None
    
    def load_truck_seed(self):
        """ Load the truck seed from the available pickled file """
        # Retrieve correct file name
        if self.simulation_time == 40:
            seed_name = f"Random_Seeds/Nobleo_OUTB_seed.pkl"
        else:
            seed_name = f"Random_Seeds/Truck_Seed_{self.simulation_time}.pkl"
        
        # Open seed file, load truck events and close file
        seed_file = open(seed_name, "rb")
        truck_events = pickle.load(seed_file)
        seed_file.close()
        
        # Return the retrieved truck events
        return truck_events
    
    def individual_valid(self, individual):
        """ Evaluate the validity of the individual """
        return individual[0] < individual[3] and individual[1] < individual[3] or individual[0] < individual[3] and individual[2] < individual[3]
    
    def run_simulation(self, config, storage_init=None, products_init=None):
        """ Create simulation environment and run simulation """
        # Initialize simulation model and run configuration evaluation
        sim = Simulate(sim_time=self.simulation_time, 
                       ppa=config[0], res=config[1], dims=config[2], 
                       storage_init_fallback=self.storage_fallback,
                       truck_init=self.truck_seed, 
                       storage_init=None, 
                       products_init=None)
        sim.RUN()
        
        # Retrieve all information required
        outbound_performance = sum([t if t > GC.OB_Truck_Late else 0.5 * t for t in sim.simres.outbound_time if t > GC.OB_Truck_OK])
        resource_cost = GC.resource_cost['forklift'] * config[1][0] + GC.resource_cost['reachtruck'] * config[1][1] + GC.resource_cost['reachtruck+'] * config[1][2]
        unplaceable_products = sim.simres.unplaceable_products
        
        # Return simulation performance
        return outbound_performance, resource_cost, unplaceable_products
       
    def transform_individual(self, individual):
        """ Transform the individual to a format that the simulation can work with """
        # Retrieve PPA rule order from individual and transform to simulation compatible format
        ppa_rule_order = self.transform.transform_PPA_rule_order(values=individual[0:4])
        
        # Retrieve PPA values from individual and transform to simulation compatible format
        ppa_values = self.transform.transform_PPA_values(values=individual[4:9])
        
        # Retrieve resource values from individual and tranform to simulation compatible format
        resource_values = self.transform.transform_resource_values(values=individual[9:12])
    
        # Retrieve dimension values from indvidual and transform to simulation compatible format
        dimension_values = self.transform.transform_warehouse_dimensions(values=individual[12:96])
        
        return [[ppa_rule_order, ppa_values], resource_values, dimension_values]
    
    def eval_individual(self, individual):
        """ Evaluate the performance of a single system configuration """
        # Transform individual to a format compatible with the simulation model
        transformed_individual = self.transform_individual(individual=individual)
       
        # If individual is invalid then return tuple of infinite values
        if self.check_validity and not self.individual_valid(individual):
            performance = (9999999, 9999999, 9999999)

        # Else evaluate individual accordingly
        else:
            performance = self.run_simulation(config=transformed_individual)

        # Return the performance of the individual
        return performance