from Simulation_Model import GC

class Reach_Truck_Plus:
    """
    Description: Manages all functionalities of a single reach truck + (mol)
    Input: reach_truck_nr: The index of the reach truck
    """
    def __init__(self, reach_truck_nr):
        self.nr = reach_truck_nr
        self.type = 'reachtruck+'
        self.occupied = False
        self.time_occupied = 0
        self.location = GC.resource_initial_location
        self.holding_mol = True
        self.mol = Mol(assigned_reach_truck=self)
        self.skill_index = GC.resource_skill_idx[GC.Reachtruckplus]
        self.speed = GC.resource_speed[GC.Reachtruckplus]

    def add_occupation_time(self, time):
        """ Add time occupied to the resource time counter """
        self.time_occupied += time
  
class Mol:
    """ 
    Description: Manages all functionalities of a mol (small AGV) that is assigned to a reach truck for Shuttle storage
    Input:  assigned_reach_truck: The reach truck to which it is assigned
    """
    def __init__(self, assigned_reach_truck):
        self.assigned_reach_truck = assigned_reach_truck
        self.location = GC.resource_initial_location
        self.placement_time = None
        
        
        
