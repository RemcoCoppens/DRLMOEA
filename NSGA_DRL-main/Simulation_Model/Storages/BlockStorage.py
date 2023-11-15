
class BlockStorageLane:
    """ 
    Description: Manage a lane of Block Storage Locations 
    Input:  - lane_nr: index of the Shuttle storage lane
            - area_nr: index of the area in which the storage is located
            - dimensions: the dimensions (width and heights) of the storage locations
            - ipd: In Path Distance (IPD) measured from central driving lane
            - depth: the number of storage locations in one lane 
    """
    def __init__(self, lane_nr, area_nr, section_nr, dimensions, ipd, depth):
        self.nr = lane_nr
        self.area = area_nr
        self.section = section_nr
        self.depth = depth
        self.width = dimensions['width']
        self.product = None
        self.full = False
        self.storage = [0] * depth
        self.type = 'BLK'
        
    def store_product(self, product, time, fraction=1):
        """ Store product(s) in the given section and lane """
        # Retrieve correct storage location and store product
        self.place_product(product)
        
        # Document product placement
        product.record_storage(loc=(self.area, (self.section, self.nr)), 
                               time=time,
                               fraction=fraction)
    
    def retrieve_product(self, product):
        """ Take product(s) in the given section and lane """
        # Retrieve correct storage location and retrieve product 
        self.take_product(product)
        
        # Document product retrieval
        product.record_retrieval(loc=(self.area, (self.section, self.nr)))
        
    def place_product(self, product):
        """ Store product in the given lane """
        # If storage is empty
        if self.product == None:
            # Set product indicator to product name and place product on the last place
            self.product = product 
            self.storage[self.depth-1] += 1
        
        # Raise error if another product is already stored in the given lane
        elif self.product != None and self.product.id != product.id:
            raise ValueError(f'[BLK] Cannot store product {product.id} in {self.area, (self.section, self.nr)} - Another product already present')
        
        # If storage not empty and correct product is to be stored
        else:
            # Raise error if storage is already full
            if self.full:
                raise ValueError(f'[BLK] Cannot store product {product.id} in {self.area, (self.section, self.nr)} - Storage already full')
            
            # Search for first available spot (starting in the back)
            for i in range(self.depth-1, -1, -1):
                
                # Check if lane has reached products stack level                
                if self.storage[i] < product.stack_level:
                    
                    # If stack level is not reached, store product and break loop                    
                    self.storage[i] += 1
                    break
        
            # Check if storage is full after product placement
            if self.storage[0] == product.stack_level: 
                self.full = True
        
    def take_product(self, product):
        """ Retrieve product from the given lane """
        # Raise error if no product is placed in the given location
        if self.product == None:
            raise ValueError(f'[BLK] Cannot retrieve product {product.id} from {self.area, (self.section, self.nr)} - No product stored here')
        
        # Raise error if another product is stored in the given location
        elif self.product.id != product.id:
            raise ValueError(f'[BLK] Cannot retrieve product {product.id} from {self.area, (self.section, self.nr)} - Another product stored here')
        
        # If storage is full, retrieve from first position and change full indicator
        if self.full:
            self.storage[0] -= 1
            self.full = False
        
        # If storage is not full
        else:
            # Search from first product (starting at the front)
            for i in range(0, self.depth):
                if self.storage[i] > 0:
                    
                    # Retrieve product and break loop 
                    self.storage[i] -= 1
                    
                    # If storage is empty after retrieval, set product to none and return empty lane signal
                    if sum(self.storage) == 0:
                        self.product = None
                    break

    def storage_available(self):
        """ Return indication if a product can be stored in this storage location """
        return not self.full