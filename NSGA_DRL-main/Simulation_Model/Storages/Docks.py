from Simulation_Model import GC

class Truck_Docks:
    """
    Description: Manage all functionality of a set of docks for one hall
    Input:  - dock_id: The id of the docks
            - nr_of_docks: The number of docks present
    """
    def __init__(self, dock_id, nr_of_docks):
        self.id = dock_id
        self.connected_consolidation = None
        self.dock_amounts = {'INB': {'docked': 0, 'allowed': GC.max_inbound[dock_id]},
                             'OUTB': {'docked': 0, 'allowed': GC.max_outbound[dock_id]}}     
        self.docks = self.create_docks(nr_of_docks=nr_of_docks)
        
    def create_docks(self, nr_of_docks):
        """ Create a single dock object to manage all functionalities of a single dock """
        return [Dock(dock_id=f'{self.id}{idx}', dock_set=self) for idx in range(1, nr_of_docks + 1)]
    
    def available_dock_and_cons(self, truck_type):
        """ Return if both a dock and a consolidation area are available for the given truck """
        # First check if there are any docks and consolidation areas available
        available_docks = [dock for dock in self.docks if dock.truck_assigned == None]
        available_cons = [cons for cons in self.connected_consolidation.lanes if cons.available]
        
        # If both a dock and consolidation is available
        if len(available_docks) > 0 and len(available_cons) > 0:
                       
            # Check if the truck is allowed to dock, looking at the maximal allowed docking restrictions
            if self.dock_amounts[truck_type]['docked'] < self.dock_amounts[truck_type]['allowed']:
                
                # Return True if an available dock is found
                return True  
            
        # If no available dock is found, return False
        return False
    
    def return_available_dock(self):
        """ Return available dock with the most preferable index """
        return [dock for dock in self.docks if dock.truck_assigned == None][GC.dock_assignment[self.id]]
    
    def dock_available(self):
        """ Return if there are any docks available """
        return any([dock.truck_assigned == None for dock in self.docks])
        

class Dock:
    """ 
    Description: Manage all functionality of a single dock
    Input:  - dock_id: The id of the dock (letter and number)
            - dock_set: The class object of the set of docks of the given hall
    """
    def __init__(self, dock_id, dock_set):
        self.id = dock_id
        self.dock_set = dock_set
        self.truck_assigned = None
        self.cons_lane_assigned = None
        
    def __str__(self):
        """ Print dock information """
        message = f'----- Dock {self.id} -----'
        if self.truck_assigned != None:
            message += f'\n\t - Truck Assigned: {self.truck_assigned.type}_{self.truck_assigned.nr}'
        else:
            message += f'\n\t - Truck Assigned: None'
        
        if self.cons_lane_assigned != None:
            message += f'\n\t - Consolidation Lane Assigned: {self.cons_lane_assigned.area}_{self.cons_lane_assigned.nr}'
        else:
            message += f'\n\t - Consolidation Lane Assigned: None'
        return message
        