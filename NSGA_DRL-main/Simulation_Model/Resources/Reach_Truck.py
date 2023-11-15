from Simulation_Model import GC

class Reach_Truck:
    """
    Description: Manages all functionalities of a single reach truck
    Input: reach_truck_nr: The index of the reach truck
    """
    def __init__(self, reach_truck_nr):
        self.nr = reach_truck_nr
        self.type = 'reachtruck'
        self.occupied = False
        self.time_occupied = 0
        self.location = GC.resource_initial_location
        self.skill_index = GC.resource_skill_idx[GC.Reachtruck]
        self.speed = GC.resource_speed[GC.Reachtruck]
        
    def add_occupation_time(self, time):
        """ Add time occupied to the resource time counter """
        self.time_occupied += time
        