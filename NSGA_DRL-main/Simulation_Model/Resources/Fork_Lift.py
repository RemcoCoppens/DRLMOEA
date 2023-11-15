from Simulation_Model import GC

class Fork_Lift:
    """
    Description: Manages all functionalities of a single fork lift
    Input: fork_lift_nr: The index of the fork lift
    """
    def __init__(self, fork_lift_nr):
        self.nr = fork_lift_nr
        self.type = 'forklift'
        self.occupied = False
        self.time_occupied = 0
        self.location = GC.resource_initial_location
        self.skill_index = GC.resource_skill_idx[GC.Forklift]
        self.speed = GC.resource_speed[GC.Forklift]
    
    def add_occupation_time(self, time):
        """ Add time occupied to the resource time counter """
        self.time_occupied += time
        