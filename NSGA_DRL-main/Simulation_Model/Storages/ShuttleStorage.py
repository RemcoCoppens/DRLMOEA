
class ShuttleStorageLane:
    """
    Description: Manage a lane of Shuttle Storage Locations consisting of 2 or 3 levels
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
        self.width = dimensions['width']
        self.heights = dimensions['height']
        self.ipd = ipd
        self.levels = self.create_storage_levels(width=self.width,
                                                 heights=self.heights,
                                                 depth=depth)
    
    def create_storage_levels(self, width, heights, depth):
        """ Create the actual storage levels in a storage lane """
        return {level: ShuttleStorageLevel(level_nr=level,
                                           width=width,
                                           height=height,
                                           depth=depth,
                                           area_nr=self.area, 
                                           section_nr=self.section, 
                                           lane_nr=self.nr) for level, height in enumerate(heights)}

class ShuttleStorageLevel:
    """
    Description: Manage a level of Shuttle Storage Locations
    Input:  - level_nr: index of the B2B storage level
            - width: the width of a storage location in this lane
            - height: the height of a storage location in this lane
            - depth: the number of storage locations in one lane
    """    
    def __init__(self, level_nr, area_nr, section_nr, lane_nr, width, height, depth):
        self.nr = level_nr
        self.area = area_nr
        self.section = section_nr
        self.lane = lane_nr
        self.width = width
        self.height = height
        self.depth = depth
        self.product = None
        self.full = False
        self.storage = 0
        self.type = 'SHT'
    
    def store_product(self, product, time, fraction=1):
        """ Store product on the given level """
        # If storage is empty
        if self.product == None:
            # Set product indicator to product name and place product in storage
            self.product = product
            self.storage += 1
            
            # Document product placement
            product.record_storage(loc=(self.area, (self.section, self.lane, self.nr)), 
                                   time=time, 
                                   fraction=fraction)
        
        # Raise error if another product is already stored on the given level
        elif self.product.id != product.id:
            raise ValueError(f'[SHT] Cannot store product {product.id} in {self.area, (self.section, self.lane, self.nr)} - Another product already present')
        
        # If storge not empty and correct product is to be stored
        else:
            # Raise error if storage is already full
            if self.full:
                raise ValueError(f'[SHT] Cannot store product {product.id} in {self.area, (self.section, self.lane, self.nr)} - Storage already full')
            
            # If inventory space is available
            else:
                self.storage += 1
                
                # Document product placement
                product.record_storage(loc=(self.area, (self.section, self.lane, self.nr)), time=time)
                
                # Check if storage is full after product placement
                if self.storage == self.depth:
                    self.full = True
    
    def retrieve_product(self, product):
        """ Retrieve product from the given level """
        # Raise error if no product is present in the storage location
        if self.product == None:
            raise ValueError(f'[SHT] Cannot retrieve product {product.id} from {self.area, (self.section, self.lane, self.nr)} - No product stored here')
        
        # Raise error if another product is present in the storage location
        elif self.product.id != product.id:
            raise ValueError(f'[SHT] Cannot retrieve product {product.id} from {self.area, (self.section, self.lane, self.nr)} - Another product stored here')
                            
        # If storage contains the correct product
        else:
            # If storage is full, remove full indication
            if self.full:
                self.full = False
            
            # Take product from storage
            self.storage -= 1
            
            # Document product retrieval
            product.record_retrieval(loc=(self.area, (self.section, self.lane, self.nr)))
            
            # If storage is empty after retrieval reset product indicator
            if self.storage == 0:
                self.product = None
                
    def storage_available(self):
        """ Return indication if a product can be stored in this storage location """
        return not self.full
