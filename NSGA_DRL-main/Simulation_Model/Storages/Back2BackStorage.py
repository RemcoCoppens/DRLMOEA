
class B2BStorageLane:
    """
    Description: Manage a lane of B2B Storage Locations consisting of depth 1 and 2 or 3 levels
    Input:  - lane_nr: index of the Shuttle storage lane
            - area_nr: index of the area in which the storage is located
            - dimensions: the dimensions (width and heights) of the storage locations
            - ipd: In Path Distance (IPD) measured from central driving lane
            - depth: the number of storage locations in one lane 
    """
    def __init__(self, lane_nr, area_nr, section_nr, dimensions, ipd, depth=1):
        self.nr = lane_nr
        self.area = area_nr
        self.section = section_nr
        self.width = dimensions['width']
        self.heights = dimensions['height']
        self.ipd = ipd
        self.levels = self.create_storage_levels(width=self.width,
                                                 heights=self.heights)
        
    def create_storage_levels(self, width, heights):
        """ Create the actual storage levels in a storage lane """
        return {level: B2BStorageLevel(level_nr=level,
                                       width=width,
                                       height=height,
                                       area_nr=self.area, 
                                       section_nr=self.section, 
                                       lane_nr=self.nr) for level, height in enumerate(heights)}
    
class B2BStorageLevel:
    """
    Description: Manage a level of B2B Storage Locations
    Input:  - level_nr: index of the B2B storage level
            - width: the width of a storage location in this lane
            - height: the height of a storage location in this lane
    """
    def __init__(self, level_nr, area_nr, section_nr, lane_nr, width, height):
        self.nr = level_nr
        self.area = area_nr
        self.section = section_nr
        self.lane = lane_nr
        self.width = width
        self.height = height
        self.product = None
        self.full = False
        self.type = 'B2B'
    
    def store_product(self, product, time, fraction=1):
        """ Store product on the given level """
        # If storage is empty
        if self.product == None:
            # Set product indicator to product name
            self.product = product
            self.full = True
            
            # Document product placement in product and in storage area
            product.record_storage(loc=(self.area, (self.section, self.lane, self.nr)), 
                                   time=time, 
                                   fraction=fraction)
    
        # Raise error if other product is placed in given location
        elif self.product.id != product.id:
            raise ValueError(f'[B2B] Cannot store product {product.id} in {self.area, (self.section, self.lane, self.nr)} - Another product already present')
        
        # Raise error if place is already filled by identical product
        else:
            raise ValueError(f'[B2B] Cannot store product {product.id} in {self.area, (self.section, self.lane, self.nr)} - Identical product already present')
    
    def retrieve_product(self, product):
        """ Retrieve product from the given level """
        # If storage contains the correct product
        if self.product.id == product.id:
            # Reset product indicator to None, indicating available storage
            self.product = None
            self.full = False
            
            # Document product retrieval
            product.record_retrieval(loc=(self.area, (self.section, self.lane, self.nr)))
            
        # Raise error if no product is present in the storage location
        elif self.product == None:
            raise ValueError(f'[B2B] Cannot retrieve product {product.id} from {self.area, (self.section, self.lane, self.nr)} - No product stored here')
        
        # Raise error if another product is present in the storage location
        else:
            raise ValueError(f'[B2B] Cannot retrieve product {product.id} from {self.area, (self.section, self.lane, self.nr)} - Another product stored here')
    
    def storage_available(self):
        """ Return indication if a product can be stored in this storage location """
        return True if self.product == None else False
