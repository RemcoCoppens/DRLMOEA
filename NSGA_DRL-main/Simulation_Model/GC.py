""" Global Constants """
# Set DEBUG MODE to print statements while running
DEBUG_MODE = False

dims_per_section = 2

B2B, vals_B2B = 0, 9
SHT, vals_SHT = 1, 7
BLK, vals_BLK = 2, 2
hall_storage_type = {'A': [True, True, True], 'B': [True, True, False], 'C': [True, True, False], 'D': [True, True, True], 'H': [True, True, False]}
                  
section_widths = {'B': {'SHT': {1: [20.5, 6], 2: [20.5, 10], 3: [20.5, 6], 4: [41.0, 6], 5: [44.0, 10], 6: [44.0, 10]},
                        'B2B': {7: 44.0, 8: 44.0, 9: 44.0, 10: 44.0, 11: 44.0, 12: 44.0,
                                13: 44.0, 14: 44.0, 15: 44.0, 16: 44.0, 17: 44.0, 18: 44.0,
                                19: 44.0, 20: 44.0, 21: 44.0, 22: 38.5, 23: 44.0, 24: 44.0,
                                25: 44.0, 26: 44.0, 27: 44.0, 28: 44.0, 29: 44.0, 30: 44.0,
                                31: 44.0, 32: 44.0, 33: 44.0, 47: 18.1, 48: 18.1, 49: 18.1}},
                  'A': {'B2B': {34: 44.0, 35: 44.0, 36: 44.0, 37: 44.0, 38: 44.0, 39: 44.0,
                                40: 44.0, 41: 44.0, 42: 44.0, 43: 44.0, 44: 44.0, 45: 44.0, 
                                46: 36.0, 50: 18.1, 51: 18.1, 52: 18.1, 53: 44.0, 54: 44.0,
                                55: 44.0, 56: 44.0, 57: 44.0, 58: 44.0, 59: 44.0, 60: 44.0, 
                                61: 44.0, 62: 44.0, 63: 44.0},
                        'SHT': {65: [8.80, 6], 66: [20.4, 10], 67: [20.4, 10], 68: [20.4, 6], 70: [5.90, 10],
                                72: [44.0, 10], 73: [44.0, 6], 74: [44.0, 6], 75: [44.7, 10]},
                        'BLK': {64: [44.0, 10], 71: [44.0, 6]}},
                  'D': {'BLK': {76: [43.6, 6], 77: [43.6, 10], 78: [43.6, 6], 82: [20.8, 10], 83: [20.8, 6], 97: [43.6, 10]},
                        'SHT': {79: [43.6, 6], 80: [44.1, 10], 81: [18.6, 6], 84: [20.3, 6], 85: [23.2, 10]},
                        'B2B': {86: 44.0, 87: 44.0, 88: 44.0, 89: 44.0, 90: 44.0, 91: 44.0, 
                                92: 44.0, 93: 44.0, 94: 44.0, 95: 44.0, 96: 44.0, 109: 44.0,
                                110: 44.0, 111: 44.0, 112: 44.0, 113: 44.0, 114: 44.0, 115: 44.0,
                                116: 44.0, 117: 44.0, 118: 44.0, 119: 44.0, 120: 44.0, 121: 37.7,
                                125: 18.1, 126: 18.1, 127: 18.1}},
                  'C': {'B2B': {98: 44.0, 99: 44.0, 100: 44.0, 101: 44.0, 102: 44.0, 103: 44.0,
                                104: 44.0, 105: 44.0, 106: 44.0, 107: 44.0, 108: 44.0, 122: 14.5,
                                123: 18.1, 124: 18.1, 130: 44.0, 131: 44.0, 132: 44.0},
                        'SHT': {128: [44.0, 6], 129: [44.0, 10], 133: [44.0, 10], 134: [36.4, 6], 135: [44.0, 10], 136: [44.0, 10]}},
                  'H': {'SHT': {137: [43.5, 10], 138: [43.5, 6], 139: [43.5, 10], 140: [34.8, 6], 141: [33.1, 6]},
                        'B2B': {142: 21.5}}}
                  
storage_area_reference = {1: list(range(1, 4)),
                          2: list(range(4, 7)),
                          3: list(range(7, 23)),
                          4: list(range(23, 47)),
                          5: list(range(47, 53)),
                          6: list(range(53, 65)),
                          7: list(range(65, 67)),
                          8: list(range(67, 71)),
                          9: list(range(71, 76)),
                          10: list(range(76, 81)),
                          11: list(range(81, 86)),
                          12: list(range(86, 98)),
                          13: list(range(98, 122)),
                          14: list(range(122, 128)),
                          15: list(range(128, 134)),
                          16: list(range(134, 137)),
                          17: list(range(137, 143))}


# Set environmental variables
warmup_time = 8
simulation_time = 40 + warmup_time
init_B2B, init_BLK, init_SHT = 0.75, 0.75, 0.75
resource_initial_location = 'CONS_A'
avg_cons2storage_time = 0.049027677840643685

# Set events
Inbound_Truck_Arrival = 0
Deload_Truck_Complete = 1
Quality_Check_Done = 2
Product2Storage_Complete = 3
Outbound_Truck_Order_Arrival = 4
Outbound_Truck_Arrival = 5
Product2Consolidation_Complete = 6
Load_Truck_Complete = 7
EVENT_NAMES = ['Inbound Truck Arrival', 
               'Deload Truck Complete', 
               'Quality Check Done', 
               'P2S Complete', 
               'Outbound Order Arrival', 
               'Outbound Truck Arrival', 
               'P2C Complete', 
               'Load Truck Complete']

# Set Jobs Indeces and priorities
Intermediate_Job_Cons2Storage = 999
Intermediate_Job_Storage2Cons = 9999
Job_Deload_Truck = 0
Job_Cons_2_BLK = 1
Job_Cons_2_B2B = 2
Job_Cons_2_SHT = 3
Job_BLK_2_Cons = 4
Job_B2B_2_Cons = 5
Job_SHT_2_Cons = 6
Job_Load_Truck = 7
job_cons2storage_deadline_increment = 2
job_needed_skill = [(1, 0, 0), (1, 1, 1), (0, 1, 1), (0, 0, 1),
                    (1, 1, 1), (0, 1, 1), (0, 0, 1), (1, 0, 0)]
JOB_NAMES = ['Deload Truck', 
             'CONS 2 BLK', 
             'CONS 2 B2B', 
             'CONS 2 SHT', 
             'BLK 2 CONS', 
             'B2B 2 CONS', 
             'SHT 2 CONS', 
             'Load Truck']

# Set Resources
Forklift = 0
Reachtruck = 1
Reachtruckplus = 2
resource_cost = {'forklift': 55000, 'reachtruck': 55800, 'reachtruck+': 61800}

# Set skills and speed of different resources
resource_speed = {Forklift: 4.722222222 * 0.55 * 3600, Reachtruck: 3.888888889 * 0.65 * 3600, Reachtruckplus: 3.888888889 * 0.65 * 3600}
# No Fallback initialization: 0.55, 0.625, 0.625
# With Fallback initialization: 0.62, 0.67, 0.67
resource_skill_idx = {Forklift: 0, Reachtruck: 1, Reachtruckplus: 2}
resource_allocation_order = [Reachtruck, Forklift, Reachtruckplus]
truck_deload_load_time = 70/3600
pick_drop_time = 25/3600
mol_retrieval_time = 10/3600
quality_check_time = 1/60  # Per pallet
distance_direct_load = 6  #m
truck_deload_load_time_DL = 60/3600
orderlines_DL = 6

# Truck Information
OB_Truck_Rush_Percentage = 0.1  # Percentage of outbound trucks that are rush orders (0.1)
Order_Arrival_Before_Truck = 2  # The amount of hrs the order arrives before the truck
OB_Broken_Pallet = 0.67         # Outbound broken pallet percentage
Broken_Pallet_Size = 0.25   # The amount that is taken off of a pallet when broken
Truck_Timestep = 0.5            # Truck arrival sequences take 30 minutes (0.5 hrs)
IB_Trucks_Per_Timestep = 2
OB_Trucks_Per_Timestep = [4, 3, 3, 3, 4, 3, 3, 3, 4, 3, 3, 3, 4, 3, 3, 3, 4, 3]
OB_Truck_OK = 0.5
OB_Truck_Late = 2.0

# Set environment variables
avg_driving = 5.000
depth_consolidation = 21.000

# Set different heights
heights = {'S': 1.13,
           'M': 1.66,
           'L': 1.93,
           'O': 2.30}
widths = {'S': 1.20,
          'M': 1.40,
          'L': 1.60}

# Set storage location widths and height combinations
width_S, width_M, width_L = widths['S'], widths['M'], widths['L']
height_O = [heights['O'], heights['O']]
height_C = [heights['L'], heights['L'], heights['S']]
height_M = [heights['M'], heights['M'], heights['M']]
height_H = [heights['L'], heights['L'], heights['L']]
empty_stack_level_BLK = 4

# Set block storage possiblities
block_S = width_S
block_L = width_L

# Different locations widths taking frames into account
loc_width = {'SHT': {'S': width_S + 0.25,
                     'M': width_M + 0.25,
                     'L': width_L + 0.25},
             'B2B': {'S': 2.725/2,    # Calculated from 2 storage locations to include racks
                     'M': 3.150/2,    # Calculated from 2 storage locations to include racks
                     'L': 3.525/2},   # Calculated from 2 storage locations to include racks
             'BLK': {'S': width_S + 0.10,
                     'L': width_L + 0.10}}
# Set universal location length
loc_length = 1.25

loc_widths = {'SHT': {'Mo': {'width': loc_width['SHT']['M'], 'height':height_O},
                      'Sc': {'width': loc_width['SHT']['S'], 'height':height_C},
                      'Mc': {'width': loc_width['SHT']['M'], 'height':height_C},
                      'Lc': {'width': loc_width['SHT']['L'], 'height':height_C},
                      'Sm': {'width': loc_width['SHT']['S'], 'height':height_M},
                      'Mm': {'width': loc_width['SHT']['M'], 'height':height_M},
                      'Lm': {'width': loc_width['SHT']['L'], 'height':height_M},
                      'Mh': {'width': loc_width['SHT']['M'], 'height':height_H}},
              'B2B': {'So': {'width': loc_width['B2B']['S'], 'height':height_O},
                      'Mo': {'width': loc_width['B2B']['M'], 'height':height_O},
                      'Lo': {'width': loc_width['B2B']['L'], 'height':height_O},
                      'Sc': {'width': loc_width['B2B']['S'], 'height':height_C},
                      'Mc': {'width': loc_width['B2B']['M'], 'height':height_C},
                      'Lc': {'width': loc_width['B2B']['L'], 'height':height_C},
                      'Sm': {'width': loc_width['B2B']['S'], 'height':height_M},
                      'Mm': {'width': loc_width['B2B']['M'], 'height':height_M},
                      'Lm': {'width': loc_width['B2B']['L'], 'height':height_M}},
              'BLK': {'S': {'width': loc_width['BLK']['S']},
                      'L': {'width': loc_width['BLK']['L']}}}

dimensions = {'SHT': {'Mo': {'width': widths['M'], 'height':height_O},
                      'Sc': {'width': widths['S'], 'height':height_C},
                      'Mc': {'width': widths['M'], 'height':height_C},
                      'Lc': {'width': widths['L'], 'height':height_C},
                      'Sm': {'width': widths['S'], 'height':height_M},
                      'Mm': {'width': widths['M'], 'height':height_M},
                      'Lm': {'width': widths['L'], 'height':height_M},
                      'Mh': {'width': widths['M'], 'height':height_H}},
              'B2B': {'So': {'width': widths['S'], 'height':height_O},
                      'Mo': {'width': widths['M'], 'height':height_O},
                      'Lo': {'width': widths['L'], 'height':height_O},
                      'Sc': {'width': widths['S'], 'height':height_C},
                      'Mc': {'width': widths['M'], 'height':height_C},
                      'Lc': {'width': widths['L'], 'height':height_C},
                      'Sm': {'width': widths['S'], 'height':height_M},
                      'Mm': {'width': widths['M'], 'height':height_M},
                      'Lm': {'width': widths['L'], 'height':height_M}},
              'BLK': {'S': {'width': widths['S']},
                      'L': {'width': widths['L']}}}

# Set predefined depth to fix difficult layout design
predef_depths = {'SA_7': 10, 'SA_11': 6}

# Define cornerpoints and extract lists and create reversed dict for lookup
corner_points = {1: {'RD': 'X1'},
                 2: {'RD': 'X1'},
                 3: {'LD': 'X1', 'RD': 'X2', 'LW': 'X10', 'RW': 'X11'},
                 4: {'LD': 'X2', 'RD': 'X3', 'LW': 'X11'},
                 5: {'LD': 'X2', 'RD': 'X3'},
                 6: {'LD': 'X3', 'RD': 'X4', 'RW': 'X13'},
                 7: {'LD': 'X3', 'RD': 'X4'},
                 8: {'LD': 'X4'},
                 9: {'LD': 'X4', 'LW': 'X13'},
                 10: {'LD': 'X5', 'LW': 'X13'},
                 11: {'LD': 'X5'},
                 12: {'LD': 'X6', 'RD': 'X5', 'LW': 'X12', 'RW': 'X13'},
                 13: {'LD': 'X7', 'RD': 'X6', 'RW': 'X12'},
                 14: {'LD': 'X7', 'RD': 'X6'},
                 15: {'LD': 'X8', 'RD': 'X7', 'LW': 'X10'},
                 16: {'LD': 'X9', 'RD': 'X8', 'RW': 'X10'},
                 17: {'RD': 'X9'}}

corner_list = [list(corner_points[k].values()) for k in corner_points.keys()]
area_restrictions = {3: [('R', 'RD')],               # Most right lane of sa 3 needs to go to X2 on (RD) due to one way traffic
                     4: [('L', 'LD'), ('R', 'RW')],  
                     6: [('L', 'LW')],
                     12: [('L', 'LD')],
                     13: [('R', 'RD'), ('L', 'LW')],
                     15: [('R', 'RW')]}

# Define all cornerpoints for reverse travel
corner_points_full = {1: {'LD': None, 'RD': 'X1', 'LW': None, 'RW': None},
                      2: {'LD': None, 'RD': 'X1', 'LW': None, 'RW': None},
                      3: {'LD': 'X1', 'RD': 'X2', 'LW': 'X10', 'RW': 'X11'},
                      4: {'LD': 'X2', 'RD': 'X3', 'LW': 'X11', 'RW': 'X12'},
                      5: {'LD': 'X2', 'RD': 'X3', 'LW': None, 'RW': None},
                      6: {'LD': 'X3', 'RD': 'X4', 'LW': 'X12', 'RW': 'X13'},
                      7: {'LD': 'X3', 'RD': 'X4', 'LW': None, 'RW': None},
                      8: {'LD': 'X4', 'RD': None, 'LW': None, 'RW': None},
                      9: {'LD': 'X4', 'RD': None, 'LW': 'X13', 'RW': None},
                      10: {'LD': 'X5', 'RD': None, 'LW': 'X13', 'RW': None},
                      11: {'LD': 'X5', 'RD': None, 'LW': None, 'RW': None},
                      12: {'LD': 'X6', 'RD': 'X5', 'LW': 'X12', 'RW': 'X13'},
                      13: {'LD': 'X7', 'RD': 'X6', 'LW': 'X11', 'RW': 'X12'},
                      14: {'LD': 'X7', 'RD': 'X6', 'LW': None, 'RW': None},
                      15: {'LD': 'X8', 'RD': 'X7', 'LW': 'X10', 'RW': 'X11'},
                      16: {'LD': 'X9', 'RD': 'X8', 'LW': None, 'RW': 'X10'},
                      17: {'LD': None, 'RD': 'X9', 'LW': None, 'RW': None}}
corner_list_full = [list(corner_points_full[k].values()) for k in corner_points_full.keys()]
corner_points_full_r = {area: {corner_points_full[area][k]: k for k in corner_points_full[area] if corner_points_full[area][k] != None} for area in corner_points_full}

# Define cornerpoint lookup list for storage areas (only forward movement)
corner_point_lookup = {'X1': {1: 'RD', 2: 'RD', 3: 'LD'},
                       'X2': {3: 'RD', 4: 'LD', 5: 'LD'},
                       'X3': {4: 'RD', 5: 'RD', 6: 'LD', 7: 'LD'},
                       'X4': {6: 'RD', 7: 'RD', 8: 'LD', 9: 'LD'},
                       'X5': {12: 'RD', 10: 'LD', 11: 'LD'},
                       'X6': {13: 'RD', 14: 'RD', 12: 'LD'},
                       'X7': {15: 'RD', 13: 'LD', 14: 'LD'},
                       'X8': {16: 'RD', 15: 'LD'},
                       'X9': {17: 'RD', 16: 'LD'},
                       'X10': {2: 'RW', 16: 'RW', 3: 'LW', 15: 'LW'},
                       'X11': {3: 'RW', 15: 'RW', 4: 'LW', 13: 'LW'},
                       'X12': {4: 'RW', 13: 'RW', 6: 'LW', 12: 'LW'},
                       'X13': {6: 'RW', 12: 'RW', 9: 'LW', 10: 'LW'}}

# Initialize consolidation area information
cons_lanes = {'A': 6, 'B': 6, 'C': 6, 'D': 6, 'H': 4}
cons_in_area = {'A': {4: 55, 5: 55}, 
                'B': {4: 13, 5: 13}, 
                'C': {13: 1, 14: 1}, 
                'D': {13: 55, 14: 55}, 
                'H': {17: 41}}

# Define which areas are to be reverse assigned and define dimension dictionary template
reversed_assignment = {1: True, 2: False, 3: False, 4: False,
                       5: True, 6: False, 7: True, 8: True,
                       9: False, 10: True, 11: False, 12: True,
                       13: True, 14: False, 15: True, 16: True, 17: True}

# Create template for dimensions dictionary of storage areas
dimension_dict_template = {'SHT': {widths['S']: {heights['S']: [],
                                                 heights['M']: [],
                                                 heights['L']: []},
                                   widths['M']: {heights['S']: [],
                                                 heights['M']: [],
                                                 heights['L']: [],
                                                 heights['O']: []},
                                   widths['L']: {heights['S']: [],
                                                 heights['M']: [],
                                                 heights['L']: []}},
                            'B2B': {widths['S']: {heights['S']: [],
                                                  heights['M']: [],
                                                  heights['L']: [],
                                                  heights['O']: []},
                                    widths['M']: {heights['S']: [],
                                                  heights['M']: [],
                                                  heights['L']: [],
                                                  heights['O']: []},
                                    widths['L']: {heights['S']: [],
                                                  heights['M']: [],
                                                  heights['L']: [],
                                                  heights['O']: []}},
                            'BLK': {widths['S']: [],
                                    widths['L']: []}}


# Set dimension scale-up for all storage types (width, height) or (width) for block
dimension_list = {'SHT': [(1.2, 1.13), (1.2, 1.66), (1.2, 1.93), (1.4, 1.13), (1.4, 1.66), (1.4, 1.93), (1.4, 2.3), (1.6, 1.13), (1.6, 1.66), (1.6, 1.93)],
                  'B2B': [(1.2, 1.13), (1.2, 1.66), (1.2, 1.93), (1.2, 2.30), (1.4, 1.13), (1.4, 1.66), (1.4, 1.93), (1.4, 2.30), (1.6, 1.13), (1.6, 1.66), (1.6, 1.93), (1.6, 2.30)],
                  'BLK': [(1.2), (1.6)]}    
dimension_scale_up = {'SHT': {(1.2, 1.13): {1: [(1.2, 1.66), (1.4, 1.13)],
                                            2: [(1.2, 1.93), (1.6, 1.13), (1.4, 1.66)],
                                            3: [(1.6, 1.66), (1.4, 1.93)],
                                            4: [(1.4, 2.30), (1.6, 1.93)],
                                            5: []},
                              (1.2, 1.66): {1: [(1.4, 1.66), (1.2, 1.93)],
                                            2: [(1.4, 1.93), (1.6, 1.66)],
                                            3: [(1.4, 2.30), (1.6, 1.93)],
                                            4: []},
                              (1.2, 1.93): {1: [(1.4, 1.93)],
                                            2: [(1.4, 2.30), (1.6, 1.93)],
                                            3: []},
                              (1.4, 1.13): {1: [(1.4, 1.66), (1.6, 1.13)],
                                            2: [(1.4, 1.93), (1.6, 1.66)],
                                            3: [(1.4, 2.30), (1.6, 1.93)],
                                            4: []},
                              (1.4, 1.66): {1: [(1.4, 1.93), (1.6, 1.66)],
                                            2: [(1.4, 2.30), (1.6, 1.93)],
                                            3: []},
                              (1.4, 1.93): {1: [(1.4, 2.30), (1.6, 1.93)],
                                            2: []},
                              (1.4, 2.30): {1: []},
                              (1.6, 1.13): {1: [(1.6, 1.66)],
                                            2: [(1.6, 1.93)],
                                            3: []},
                              (1.6, 1.66): {1: [(1.6, 1.93)],
                                            2: []},
                              (1.6, 1.93): {1: []}}, 
                              
                      'B2B': {(1.2, 1.13): {1: [(1.2, 1.66), (1.4, 1.13)],
                                            2: [(1.2, 1.93), (1.6, 1.13), (1.4, 1.66)],
                                            3: [(1.6, 1.66), (1.4, 1.93), (1.2, 2.30)],
                                            4: [(1.4, 2.30), (1.6, 1.93)],
                                            5: [(1.6, 2.30)],
                                            6: []},
                              (1.2, 1.66): {1: [(1.4, 1.66), (1.2, 1.93)],
                                            2: [(1.4, 1.93), (1.6, 1.66)],
                                            3: [(1.4, 2.30), (1.6, 1.93)],
                                            4: []},
                              (1.2, 1.93): {1: [(1.4, 1.93), (1.2, 2.30)],
                                            2: [(1.4, 2.30), (1.6, 1.93)],
                                            3: [(1.6, 2.30)],
                                            4: []},
                              (1.2, 2.30): {1: [(1.4, 2.30)],
                                            2: [(1.6, 2.30)],
                                            3: []},
                              (1.4, 1.13): {1: [(1.4, 1.66), (1.6, 1.13)],
                                            2: [(1.6, 1.66), (1.4, 1.93)],
                                            3: [(1.6, 1.93), (1.4, 2.30)],
                                            4: [(1.6, 2.30)],
                                            5: []},
                              (1.4, 1.66): {1: [(1.4, 1.93), (1.6, 1.66)],
                                            2: [(1.4, 2.30), (1.6, 1.93)],
                                            3: [(1.6, 2.30)],
                                            4: []},
                              (1.4, 1.93): {1: [(1.4, 2.30), (1.6, 1.93)],
                                            2: [(1.6, 2.30)],
                                            3: []},
                              (1.4, 2.30): {1: [(1.6, 2.30)],
                                            2: []},
                              (1.6, 1.13): {1: [(1.6, 1.66)],
                                            2: [(1.6, 1.93)],
                                            3: [(1.6, 2.30)],
                                            4: []},
                              (1.6, 1.66): {1: [(1.6, 1.93)],
                                            2: [(1.6, 2.30)],
                                            3: []},
                              (1.6, 1.93): {1: [(1.6, 2.30)],
                                            2: []},
                              (1.6, 2.30): {1: []}},
                              
                      'BLK': {(1.2): {1: [(1.6)],
                                      2: []},
                              (1.6): {1: []}}}  

# Create initial configuration                                                
init_config = {1: [{'Mo': 1.0}, {'Mo': 1.0}, {'Lm': 0.5, 'Mo': 0.5}],
               2: [{'Lc': 1.0}, {'Lc': 1.0}, {'Mc': 1.0}],
               3: [{'Mc': 1.0}, {'Mc': 1.0}, {'Mc': 1.0}, {'Mc': 1.0}, {'Mc': 1.0}, {'Lc': 1.0},
                   {'Lc': 1.0}, {'Lc': 1.0}, {'Lc': 1.0}, {'Lc': 1.0}, {'Lc': 1.0}, {'So': 1.0},
                   {'Sc': 1.0}, {'Sc': 1.0}, {'Sc': 1.0}, {'Sc': 1.0}],
               4: [{'Sc': 1.0}, {'Sm': 1.0}, {'Sm': 1.0}, {'Sm': 1.0}, {'Sm': 1.0}, {'Sm': 1.0},
                   {'Sm': 1.0}, {'Sm': 1.0}, {'Sm': 1.0}, {'Mm': 1.0}, {'Mm': 1.0}, {'Mm': 1.0}, 
                   {'Mm': 1.0}, {'Mc': 1.0}, {'Mc': 1.0}, {'So': 1.0}, {'So': 1.0}, {'Sm': 1.0},
                   {'Sm': 1.0}, {'Lm': 1.0}, {'Lm': 1.0}, {'Sc': 1.0}, {'Sc': 1.0}, {'Sc': 1.0}],
               5: [{'Lc': 1.0}, {'Lc': 1.0}, {'Lc': 1.0}, {'Lc': 1.0}, {'Lc': 1.0}, {'Lc': 1.0}],
               6: [{'Sc': 1.0}, {'Lo': 1.0}, {'Lo': 1.0}, {'Lc': 1.0}, {'Lc': 1.0}, {'Lc': 1.0},
                   {'Lc': 1.0}, {'Lc': 1.0}, {'Lc': 1.0}, {'Lc': 1.0}, {'Lc': 1.0}, {'L': 1.0}],
               7: [{'Sc': 1.0}, {'Sc': 1.0}],
               8: [{'Sc': 1.0}, {'Sc': 1.0}, {'Sc': 1.0}, {'Sc': 1.0}],
               9: [{'L': 1.0}, {'Mo': 1.0}, {'Mo': 1.0}, {'Mo': 1.0}, {'Mo': 1.0}],
               10: [{'L': 1.0}, {'S': 0.92, 'L':0.08}, {'S': 0.92, 'L':0.08}, 
                    {'Mm': 0.53, 'Lm':0.47}, {'Sm': 0.3625, 'Mm':0.6375}],
               11: [{'Lc': 1.0}, {'L': 1.0}, {'L': 1.0}, {'Lm': 1.0}, {'Mm': 1.0}],
               12: [{'Mc': 1.0}, {'Mc': 1.0}, {'So': 1.0}, {'Mm': 1.0}, {'Mm': 1.0}, {'Lc': 1.0},
                    {'Lc': 1.0}, {'Lc': 1.0}, {'Lc': 1.0}, {'Lc': 1.0}, {'Lc': 1.0}, {'L': 1.0}],
               13: [{'Lm': 1.0}, {'Lm': 1.0}, {'Lm': 1.0}, {'Lc': 1.0}, {'Lc': 1.0}, {'Lc': 1.0}, 
                    {'Lc': 1.0}, {'Lo': 1.0}, {'Sm': 1.0}, {'Mm': 1.0}, {'Mm': 1.0}, {'Sm': 1.0},
                    {'Sm': 1.0}, {'Sm': 1.0}, {'Sm': 1.0}, {'Sm': 1.0}, {'Sm': 1.0}, {'Sm': 1.0},
                    {'Sm': 1.0}, {'Sc': 1.0}, {'Sc': 1.0}, {'Mc': 1.0}, {'Mc': 1.0}, {'Mc': 1.0}],
               14: [{'Lc': 1.0}, {'Lc': 1.0}, {'Lc': 1.0}, {'Lc': 1.0}, {'Lc': 1.0}, {'Lc': 1.0}],
               15: [{'Sm': 1.0}, {'Lc': 1.0}, {'Lc': 1.0}, {'Lc': 1.0}, {'Lc': 1.0}, {'Lm': 1.0}],
               16: [{'Sc': 1.0}, {'Sc': 1.0}, {'Sm': 1.0}],
               17: [{'Mh': 1.0}, {'Mh': 1.0}, {'Mh': 1.0}, {'Mh': 1.0}, {'Mh': 1.0}, {'Lc': 1.0}]}

# Truck docking variables
number_of_docks = {'A': 5, 'B': 5, 'C': 5, 'D': 5, 'H': 4}
dock_assignment = {'A': 0, 'B': -1, 'C': -1, 'D': 0, 'H': -1}
max_inbound = {'A': 3, 'B': 3, 'C': 3, 'D': 3, 'H': 2}
max_outbound = {'A': 3, 'B': 3, 'C': 3, 'D': 3, 'H': 2}
