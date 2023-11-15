import pandas as pd
import numpy as np
import random
import math
from bisect import bisect_right

from Simulation_Model import GC, GF
from Simulation_Model.Simulation_Backend.Event import Event


class Inbound_Trucks:
    """ 
    Description: Creates and documents all inbound trucks
    Input:  - ExcelFile: Name of the Excel file containing inbound truck information
    """
    def __init__(self, ExcelFile, products):
        self.products = products
        self.percentage_df = pd.read_excel(ExcelFile, 'Inbound')
        self.product_dist_df = pd.read_excel(ExcelFile, 'Inbound_Dist')
        self.product_dist_df['Product_ID'] = self.product_dist_df['Product_ID'].map(str)  # Transform IDs to strings
        self.pallet_storage_init = self.percentage_df.orderlines.tolist()
        self.pallets_percentages = self.extract_pallets_percentages()
        self.orderlines_percentages = self.extract_orderlines_percentages()
        self.product_percentages =  self.extract_product_percentages()
        self.truck_counter = 0
    
    def extract_pallets_percentages(self):
        """ Retrieve number of pallet percentages from the given file and transform into useable format """
        return np.cumsum(self.percentage_df.orderlines.tolist())
    
    def extract_orderlines_percentages(self):
        """ Retrieve number of orderlines percentages from the given file and transform into useable format """
        # Retrieve percentages and fix percentage bug exceeding 100%
        percentages = [[x/sum(row[2:].dropna().tolist()) for x in row[2:].dropna().tolist()] for _, row in self.percentage_df.iterrows()]
        
        # Create dictionary with cumulative sums of percentages
        return {idx + 1: np.cumsum(row) for idx, row in enumerate(percentages)}
        
    def extract_product_percentages(self):
        """ Retrieve product percentages from the given file and transform into useable format """
        # Aggregate duplicate products (different packaging) and calculate total percentage to round variation
        agg_products = self.product_dist_df[['Product_ID', 'Percentage']].groupby('Product_ID').agg('sum')
        total_perc = sum(agg_products.Percentage.tolist())
        
        # Return dictionary containing both cumulative probabilities and product IDs
        return {'Product_ID': agg_products.index.tolist(), 'Percentage': [x/total_perc for x in agg_products.Percentage.tolist()]}
    
    def truck_arrival(self, time, storage_initialization=False, random_seed=None):
        """ Create truck object for new inbound truck arrival """
        # If storage initialization, set random seed to given index
        if storage_initialization:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # Generate number of pallets using the inbound pallet percentages
        nr_of_pallets = bisect_right(self.pallets_percentages, random.random()) + 1

        # Generate the number of orderlines using the inbound orderlines percentages
        nr_of_orderlines = bisect_right(self.orderlines_percentages[nr_of_pallets], random.random()) + 1

        # Retrieve product IDs equal to the number of orderlines, pulled from product distribution
        product_list = np.random.choice(a=self.product_percentages['Product_ID'],
                                        size=nr_of_orderlines, 
                                        replace=False,
                                        p=self.product_percentages['Percentage'])

        # Decide on quantities per product
        quantities = GF.quantities_per_orderline(pallets=nr_of_pallets, 
                                                 orderlines=nr_of_orderlines)
        
        # If storage initialization, do not create trucks but return the products and quantities directly
        if storage_initialization:
            return GF.flatten_list([[self.products.catalogue[prod]] * quantities[idx] for idx, prod in enumerate(product_list)])
        
        # Create truck object, append to truck list and increment inbound truck counter
        new_inbound_truck = Truck(truck_nr=self.truck_counter, 
                                  truck_type='INB',
                                  arrival_time=time,
                                  orderlist=GF.flatten_list([[prod] * quantities[idx] for idx, prod in enumerate(product_list)]),
                                  nr_of_pallets=nr_of_pallets,
                                  nr_of_orderlines=nr_of_orderlines)
        self.truck_counter += 1
        
        # Return newly created inbound truck
        return new_inbound_truck
    
    def initialize_truck_arrivals(self, fes, simulation_time):
        """ Initialize all inbound truck arrivals for the complete simulation run """
        # Calculate time between inbound trucks and total amount of trucks in simulation time
        interarrival_time = GC.Truck_Timestep / GC.IB_Trucks_Per_Timestep
        total_inbound_trucks = int(((simulation_time + GC.warmup_time) // interarrival_time) + 1)  # one extra as start from time 0
        
        # Create new inbound trucks for full simulation duration and add to fes
        for truck_idx in range(total_inbound_trucks):
            arrival_time = truck_idx * interarrival_time
            new_inbound_truck = self.truck_arrival(time=arrival_time)
            truck_arrival_event = Event(typ=GC.Inbound_Truck_Arrival,
                                        cur_time=0,
                                        time=arrival_time, 
                                        truck=new_inbound_truck)
            fes.add(truck_arrival_event)
    
           
class Outbound_Trucks:
    """ 
    Description: Creates and documents all outbound trucks
    Input:  - ExcelFile: Name of the Excel file containing outbound truck information
    """
    def __init__(self, ExcelFile, products):
        self.products = products
        self.percentage_df = pd.read_excel(ExcelFile, 'Outbound')
        self.pallets_percentages = self.extract_pallets_percentages()
        self.orderlines_percentages = self.extract_orderlines_percentages()
        self.truck_counter = 0
        self.timestep_counter = 0
    
    def extract_pallets_percentages(self):
        """ Retrieve number of pallet percentages from file and transform into useable format """
        return np.cumsum(self.percentage_df.orderlines.tolist())
    
    def extract_orderlines_percentages(self):
        """ Retrieve number of orderlines percentages from file and transform into useable format """
        # Retrieve percentages and fix percentage bug exceeding 100%
        percentages = [[x/sum(row[2:].dropna().tolist()) for x in row[2:].dropna().tolist()] for _, row in self.percentage_df.iterrows()]
        
        # Create dictionary with cumulative sums of percentages
        return {idx + 1: np.cumsum(row) for idx, row in enumerate(percentages)}
        
    def truck_order_arrival(self, time, truck):
        """ Create truck object for new inbound truck arrival """        
        # Calculate quantities of the nr of products
        quantities = GF.quantities_per_orderline(pallets=truck.pallets, 
                                                 orderlines=truck.orderlines)
        
        # Transform pallets to broken pallets given the preset pulls from the percentage of broken pallets
        quantities = [qty - 1 + GC.Broken_Pallet_Size if truck.broken_pallets[idx] else qty for idx, qty in enumerate(quantities)]

        # Retrieve number of oldest products identical to the desired number of orderlines
        product_IDs = self.products.return_oldest_products(qtys=quantities)
        
        # Retrieve all product class objects and set status to reserved
        requested_products = []
        locations_products = []
        for idx, product in enumerate(product_IDs):
            
            # If product is not N/A, thus found
            if product != 'N/A':
                product.not_reserved = False
                qty = quantities[idx]
                
                # Create search list for preventing duplicate selection
                product_stored_list = product.stored.copy()
                product_pallet_fractions_list = product.pallet_fractions.copy()
                product_placement_times_list = product.placement_times.copy()
                
                # Create orderlist item for all requested products
                for prod in range(math.floor(qty)):
                    requested_products.append((product, 1))
                     
                    # Look for oldest product location
                    oldest_loc = product.return_oldest_product_location(look_list=product_stored_list,
                                                                        pf_look_list=product_pallet_fractions_list, 
                                                                        pt_look_list=product_placement_times_list)
                    
                    # Remove from looklist to prevent duplicate selection and add to monitoring list
                    del product_pallet_fractions_list[product_stored_list.index(oldest_loc)]
                    del product_placement_times_list[product_stored_list.index(oldest_loc)]
                    product_stored_list.remove(oldest_loc)
                    locations_products.append(oldest_loc)
                    
                    # Decrease counter
                    qty -= 1
                    
                # If broken pallet remains, add this as well
                if qty % 1 == GC.Broken_Pallet_Size:
                    requested_products.append((product, GC.Broken_Pallet_Size))
                    
                    # Look for oldest product location
                    oldest_loc = product.return_oldest_product_location(look_list=product_stored_list,
                                                                        pf_look_list=product_pallet_fractions_list, 
                                                                        pt_look_list=product_placement_times_list, 
                                                                        broken_pallet=True)
                    # Remove from looklist to prevent duplicate selection and add to monitoring list
                    del product_pallet_fractions_list[product_stored_list.index(oldest_loc)]
                    del product_placement_times_list[product_stored_list.index(oldest_loc)]
                    product_stored_list.remove(oldest_loc)
                    locations_products.append(oldest_loc)
            
            # If product is N/A, meaning it is not found, apply fallback option
            else:
                # Retrieve quantities
                qty = quantities[idx]
                
                # First place all full pallets
                for prod in range(math.floor(qty)):
                    requested_products.append((product, 1))
                    locations_products.append('N/A')
                    qty -= 1
                
                # If broken pallet remains, place this as well
                if qty % 1 == GC.Broken_Pallet_Size:
                    requested_products.append((product, 1))
                    locations_products.append('N/A')
                    
        # Attach created orderlist and retrieval locations to truck
        truck.orderlist, truck.orderlist_W, truck.prod_list = requested_products, requested_products.copy(), [p[0] for p in requested_products]
        truck.retrieval_locations = locations_products
    
    def initialize_truck_arrivals(self, fes, simulation_time):
        """ Initialize all outbound truck arrivals for the complete simulation run """        
        # Loop over all timesteps for truck arrival in simulation time
        for idx, time in enumerate(np.arange(GC.warmup_time, simulation_time + GC.warmup_time, GC.Truck_Timestep)):
            
            # Retrieve number of trucks to arrive in the timestep to be created
            nr_of_trucks = GC.OB_Trucks_Per_Timestep[idx % len(GC.OB_Trucks_Per_Timestep)]
            
            # Create number of trucks for the given timestep
            for _ in range(nr_of_trucks):
                # Generate number of pallets using the inbound pallet percentages
                nr_of_pallets = bisect_right(self.pallets_percentages, random.random()) + 1
            
                # Generate the number of orderlines using the inbound orderlines percentages
                nr_of_orderlines = bisect_right(self.orderlines_percentages[nr_of_pallets], random.random()) + 1
                
                # If truck concerns a rush order
                if random.random() < GC.OB_Truck_Rush_Percentage:
                    # Create truck object, append to truck list and increment inbound truck counter
                    truck_arrival_time = time + random.random() * GC.Truck_Timestep
                    new_outbound_truck = Truck(truck_nr=self.truck_counter, 
                                               truck_type='OUTB',
                                               arrival_time=truck_arrival_time,
                                               nr_of_pallets=nr_of_pallets, 
                                               nr_of_orderlines=nr_of_orderlines,
                                               rush_order=True,
                                               broken_pallets=[random.random() < GC.OB_Broken_Pallet for _ in range(nr_of_orderlines)])
                    self.truck_counter += 1
                    
                    # Only create truck arrival event on stochastic time in timestep and add to FES
                    truck_arrival_event = Event(typ=GC.Outbound_Truck_Arrival,
                                                cur_time=0,
                                                time=truck_arrival_time,
                                                truck=new_outbound_truck)
                    fes.add(truck_arrival_event)
                
                # If truck is not a rush order
                else:
                    # Create truck object, append to truck list and increment inbound truck counter
                    truck_arrival_time = time + random.random() * GC.Truck_Timestep
                    new_outbound_truck = Truck(truck_nr=self.truck_counter, 
                                               truck_type='OUTB',
                                               arrival_time=truck_arrival_time,
                                               nr_of_pallets=nr_of_pallets, 
                                               nr_of_orderlines=nr_of_orderlines,
                                               rush_order=False,
                                               broken_pallets=[random.random() < GC.OB_Broken_Pallet for _ in range(nr_of_orderlines)])
                    self.truck_counter += 1
                    
                    # Create outbound truck order arrival event and add to FES
                    order_arrival_event = Event(typ=GC.Outbound_Truck_Order_Arrival,
                                                cur_time=0,
                                                time=(truck_arrival_time - GC.Order_Arrival_Before_Truck),
                                                truck=new_outbound_truck)
                    fes.add(order_arrival_event)
                    
                    # Create outbound truck arrival event and add to FES
                    truck_arrival_event = Event(typ=GC.Outbound_Truck_Arrival,
                                                cur_time=0,
                                                time=truck_arrival_time,
                                                truck=new_outbound_truck)
                    fes.add(truck_arrival_event)
                    
    def initialize_trucks_from_nobleo_seed(self, df, fes):
        """ Use Nobleo truck seed to initialize all (order and) truck arrivals. """
        # Create two additional columns to the dataframe indicating a rush order and arrival time in hours
        df['rush'] = df['Arrival Time'] == df['Order Time']
        df['TA_hrs'] = df['Arrival Time'] / 3600
        df['OA_hrs'] = df['Order Time'] / 3600
        
        # Loop over rows of the dataframe
        for idx, row in df.iterrows():
            if row['TA_hrs'] > 48:
                pass
            else:
                new_outbound_truck = Truck(truck_nr=idx, 
                                           truck_type='OUTB',
                                           arrival_time=row['TA_hrs'],
                                           nr_of_pallets=row['Number of Pallets'], 
                                           nr_of_orderlines=row['Number of Orderlines'],
                                           rush_order=row['rush'],
                                           broken_pallets=[random.random() < GC.OB_Broken_Pallet for _ in range(row['Number of Orderlines'])])
                
                # If truck does not concern rush order, create turkc order arrival event
                if not row['rush']:
                    # Create outbound truck order arrival event and add to FES
                    order_arrival_event = Event(typ=GC.Outbound_Truck_Order_Arrival,
                                                cur_time=0,
                                                time=(row['TA_hrs'] - GC.Order_Arrival_Before_Truck),
                                                truck=new_outbound_truck)
                    fes.add(order_arrival_event)
                
                # Create outbound truck arrival event and add to FES
                truck_arrival_event = Event(typ=GC.Outbound_Truck_Arrival,
                                            cur_time=0,
                                            time=row['TA_hrs'],
                                            truck=new_outbound_truck)
                fes.add(truck_arrival_event)
            

class Truck:
    """
    Description: Placeholder for all information regarding a truck
    Input:  - truck_nr: the index of the truck
            - truck_type: If the truck concerns either inbound or outbound
            - arrival_time: The arrival time of the truck
            - orderlist: list of products and their quantities requested/delivered
            - nr_of_pallets: the number of pallets for/in the truck
            - nr_of_orderlines: the number of orderlines to make up the pallets
            - rush_order: indicates if truck concerns rush order (only for outbound trucks)
    """
    def __init__(self, truck_nr, truck_type, arrival_time, nr_of_pallets=0, nr_of_orderlines=0, orderlist=[], rush_order=False, broken_pallets=[]):
        self.nr = truck_nr
        self.type = truck_type
        self.arrival_time = arrival_time
        self.orderlist = orderlist
        self.orderlist_W = None
        self.prod_list = None
        self.retrieval_locations = None
        self.pallets = nr_of_pallets
        self.orderlines = nr_of_orderlines
        self.rush_order = rush_order
        self.broken_pallets = broken_pallets
        self.cons_ready = False
        self.departure_time = None
        self.dock_assigned = None
        self.cons_assigned = None
    
    def __str__(self):
        """ Print truck information """
        message = f' ----- Truck: {self.type}_{self.nr} ----- \n \t - Arrival_Time: {self.arrival_time} \n \t - Pallets: {self.pallets}, \n \t - Orderlines: {self.orderlines}'
        if self.type == 'OUTB' and self.orderlist_W != None:
            message += f'\n \t - Products picked: {len(self.orderlist) - len(self.orderlist_W)}/{len(self.orderlist)} \n \t - Rush Order: {self.rush_order}'
        elif self.type == 'OUTBD':
            message += f'\n \t - Rush Order: {self.rush_order}'
        if self.dock_assigned != None:
            message += f'\n \t - Dock Assigned: {self.dock_assigned.id}'
        if self.cons_assigned != None:
            message += f'\n \t - Consolidation Assigned: {self.cons_assigned.area}_{self.cons_assigned.nr}'  
        return message
    
    def arrival(self, dock, cons_lane):
        """ Document truck arrival """ 
        # Retrieve consolidation lane and attach to dock
        cons_lane.available = False
        cons_lane.truck_assigned = self
        cons_lane.truck_arrival = self.arrival_time
        
        # Attach truck to dock and dock to truck
        self.dock_assigned = dock
        dock.dock_set.dock_amounts[self.type]['docked'] += 1
        
        # Set dock variables
        dock.cons_lane_assigned = cons_lane
        dock.truck_assigned = self
        
        # Set truck variables
        self.cons_assigned = cons_lane
        self.dock_assigned = dock
        
    
    def outbd_order_arrival(self, cons_lane):
        """ Handle outbound truck order arrival, by reserving a consolidation lane in the selected consolidation area """
        # Set consolidation lane to unavailable and attach truck
        cons_lane.available = False
        cons_lane.truck_assigned = self
        
        # Set truck variables
        self.cons_assigned = cons_lane
        
    
    def outbd_truck_arrival(self, dock=None):
        """ Handle arrival of non rush order outbound truck """
        # Retrieve consolidation lane from truck object
        cons_lane = self.cons_assigned
        cons_lane.truck_arrival = self.arrival_time
        
        # Retrieve consolidation area and try to find an available dock (if dock is not given)
        cons_area = cons_lane.consolidation_object
        if dock == None:
            # If there are docks available, retrieve dock and continue
            if cons_area.connected_docks.dock_available():
                dock = cons_area.connected_docks.return_available_dock()
                
            # If no docks available, return none
            else:
                return None

        # Attach dock to consolidation and truck
        dock.dock_set.dock_amounts[self.type]['docked'] += 1
        dock.cons_lane_assigned = cons_lane
        dock.truck_assigned = self
        self.dock_assigned = dock
        
        # Return consolidation lane
        return cons_lane
        
    
    def departure(self, time, simres):
        """ Document truck departure """
        # Record completion time
        self.departure_time = time
        
        # Retrieve dock object and reset dock and consolidation
        dock = self.dock_assigned
        cons_lane = dock.cons_lane_assigned
        dock.truck_assigned = None
        dock.dock_set.dock_amounts[self.type]['docked'] -= 1
        dock.cons_lane_assigned = None
        cons_lane.truck_arrival = None
        
        
        # If outbound truck departure, document departure and performance
        if self.type == 'OUTB':
            simres.document_outbound_departure(self)
            
        # If inbound truck departure, document departure
        else:
            simres.document_inbound_departure(self)
            
    def pop_product_and_location(self):
        """ Pop and return a location and its retrieval location from truck lists """
        _ = self.prod_list.pop(0)  # Also remove first item of the monitoring product list for computation speed up
        return self.orderlist_W.pop(0), self.retrieval_locations.pop(0)
