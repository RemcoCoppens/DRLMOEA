import pandas as pd
import numpy as np

class ProductCatalogue:
    """ 
    Description: Holds all product class instances in searchable dictionary
    Input:  - ExcelFile: Name of the Excel file containing all product information
    """
    def __init__(self, ExcelFile):
        self.catalogue = self.create_catalogue(df=pd.read_excel(ExcelFile))
        
    def __str__(self):
        """ Print summary of amount of products stored """
        products_stored = [prod for prod in self.catalogue.values() if len(prod.stored) > 0]
        message = f'Currently there are {len(products_stored)} products stored in the warehouse.'
#        if len(products_stored) > 0:
#            for prod in products_stored:
#                message += '\n \t - Product {:>6}:  Different Locations: {:>3},  Oldest placement: {:>2}.'.format(prod.id, len(set(prod.stored)), prod.placement_times[0])
        return message        
    
    def reset_catalogue(self):
        """ Reset all products for new simulation run """
        for product in self.catalogue.values():
            product.stored = []
            product.pallet_fractions = []
            product.placement_times = []
            product.not_reserved = True
    
    def create_catalogue(self, df):
        """ Create product catalogue using the given dataframe of products """
        df['Product_ID'] = df['Product_ID'].map(str) # Transform IDs from floats to STR
        return {prod.Product_ID: Product(product_id=prod.Product_ID,
                                         width=prod.Width,
                                         height=prod.Height,
                                         stack_level=prod.Stack_Level,
                                         historic=prod.Historic_Outbound) for idx, prod in df.iterrows()}
   
    def return_oldest_products(self, qtys):
        """ Return the location of the N oldest products stored in the warehouse with the requested quantities present """       
        # Loop through quantities and select products
        selected_products = []
        for qty in qtys:
            # Retrieve viable product IDs for selection
            products = np.array([self.catalogue[k] for k in self.catalogue.keys() if sum(self.catalogue[k].pallet_fractions) >= qty 
                                 and self.catalogue[k].not_reserved 
                                 and self.catalogue[k] not in selected_products])
            
            # If no products are found, set N/A, this will later be translated to average travel distance
            if len(products) == 0:
                selected_products.append('N/A')
            
            # If product is found
            else:
                # Retrieve oldest placement dates of all product IDs
                oldest_placement_dates = np.array([min(prod.placement_times) for prod in products])
                
                # Retrieve oldest product ID
                oldest_product = products[np.argmin(oldest_placement_dates)]
                            
                # Append the oldest product to the monitoring list
                selected_products.append(oldest_product)
                                
        # Return selected products
        return selected_products
    
class Product:
    """
    Description: Hold all information concerning a single product
    Input:  - Product_nr: The index of the product
            - width: The width of the product
            - height: The height of the product
            - stack_level: The number of pallets the product can be stacked to in block storage
            - historic: The historic outbound to be used in the Product Placement Algorithm (PPA)
    """
    def __init__(self, product_id, width, height, stack_level, historic):
        self.id = product_id
        self.width = width
        self.height = height
        self.stack_level = stack_level
        self.historic_outb = historic
        self.stored = []
        self.pallet_fractions = []
        self.placement_times = []
        self.not_reserved = True
    
    def __str__(self):
        """ Print summary of information about the product """
        message = f'---- Product {self.id} ----\n'
        message += f'   width: {self.width}\n'
        message += f'   height: {self.height}\n'
        message += f'   stack_level: {self.stack_level}\n'
        message += f'   historic outbound: {self.historic_outb}\n'
        message += f'   currently stored: {sum(self.pallet_fractions)}\n'
        if sum(self.pallet_fractions) > 0:
            message += f'   oldest product: {min(self.placement_times)}'
        return message
    
    def record_storage(self, loc, time, fraction=1):
        """ Record placement of product and documents time of placement """
        self.stored.append(loc)
        self.placement_times.append(time)
        self.pallet_fractions.append(fraction)
        
    def record_retrieval(self, loc):
        """ Record retrieval of product, take time from placement times and set product available for other Outbound Trucks """
        # Retrieve index of the location in the stored list
        idx = self.stored.index(loc)
        
        # Remove product from storage and remove its placement time and pallet fraction
        _ = self.stored.pop(idx)
        _ = self.placement_times.pop(idx)
        _ = self.pallet_fractions.pop(idx)
            
    def return_oldest_product_location(self, look_list, pf_look_list, pt_look_list, broken_pallet=False):
        """ Return the location of the oldest products stored """
        # If half a pallet is required, return oldest half pallet location
        if broken_pallet:
            # Retrieve storage locatations containing broken pallets
            broken_pallets = [loc for (idx, loc) in enumerate(look_list) if pf_look_list[idx] < 1]
            
            # Broken pallet is found, return the oldest
            if len(broken_pallets) > 0:
                # Return all placement dates of found storage locations and return the storage location with the oldest product
                placement_dates = np.array([pt_look_list[look_list.index(loc)] for loc in broken_pallets])
                return broken_pallets[np.argmin(placement_dates)]
        try:   
            # If either full pallet is required or no broken pallet is found, return oldest (not already selected) product
            return [loc for (idx, loc) in enumerate(look_list) if pf_look_list[idx] == 1][0]
        except IndexError:
            print(f'Looklist: {look_list} Stored: {self.stored}, pallet fracs: {self.pallet_fractions}')
            
    def quantity_stored(self):
        """ Return the total amount of products stored """
        return sum(self.pallet_fractions)
    