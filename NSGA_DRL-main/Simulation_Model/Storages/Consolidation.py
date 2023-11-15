from Simulation_Model import GC

class Consolidation:
    """ 
    Description: Manages all consolidation areas for the various truck docks 
    Input: - consolidation_nr: The index of the consolidation area
           - cons_lanes: The number of consolidation lanes
    """
    def __init__(self, consolidation_nr, lanes):
        self.nr = f'CONS_{consolidation_nr}'
        self.lanes = self.create_consolidation_lanes(cons_lanes=lanes)
        self.in_area = GC.cons_in_area[consolidation_nr]
        self.connected_docks = None

    def create_consolidation_lanes(self, cons_lanes):
        """ Generate consolidation lanes """
        return [Consolidation_lane(cons_lane_nr=idx, cons_area=self) for idx in range(cons_lanes)] 

    def available_cons_lane(self):
        """ Return if there is a consolidation lane available """
        return any([cons_lane.available for cons_lane in self.lanes])

    def return_available_lane(self):
        """ Return the object of the first available consolidation lane """
        return [lane for lane in self.lanes if lane.available][0]

class Consolidation_lane:
    """ 
    Description: Manage a single consolidation lane
    Input: - cons_lane_nr: The index of the consolidation lane
           - cons_area: The class object of the consolidation area of which the lane is a part
    """
    def __init__(self, cons_lane_nr, cons_area):
        self.nr = cons_lane_nr
        self.area = cons_area.nr
        self.consolidation_object = cons_area
        self.available = True
        self.truck_arrival = None
        self.truck_assigned = None
        self.stored = []
        
    def __str__(self):
        """ Print station of the consolidation lane """
        if self.truck_assigned: 
            truck = f'{self.truck_assigned.type}_{self.truck_assigned.nr}'
            if self.truck_assigned.type == 'OUTB' and not self.truck_assigned.rush_order:
                dock = 'T.B.D.'
            else:
                dock = f'{self.truck_assigned.dock_assigned.id}'
        else: 
            truck = None
            dock = None
        
        return f'----- {self.area}_{self.nr} ----- \n \t - Truck: {truck}, \n\t - Dock: {dock}, \n\t - Stored: {len(self.stored)}, \n\t - Available: {self.available}'
        
    def place_products(self, product_list):
        """ Place a list of products in the consolidation lane """
        _ = [self.place_product(product=product) for product in product_list]
        
    def place_product(self, product):
        """ Place a single product in the consolidation lane """
        self.stored.append(product)
    
    def take_products(self):
        """ Take all products from consolidation lane and set availability to True """
        self.stored = []
        self.available = True
        self.truck_assigned = None
    
    def take_product(self):
        """ Pop and return the last placed product out of the consolidation area (LIFO) """
        # Take first product from list of stored products
        product = self.stored.pop(-1)
        
        # If after taking product consolidation lane is empty, set availability to True
        if len(self.stored) == 0:
            self.available = True
            self.truck_assigned = None
        
        # Return taken product
        return product