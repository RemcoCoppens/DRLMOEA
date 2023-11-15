import numpy as np
from Simulation_Model import GF, GC

class Distances:
    """
    Description: Manages all distance calculations in the warehouse
    Input:  - storage: Storage class containing all functionality of the warehouse
    """
    def __init__(self, storage):
        self.cons2loc = GF.dataframe_to_dictionary(fileName='Distances', sheetName='Cons_2_Loc')
        self.loc2cons = GF.dataframe_to_dictionary(fileName='Distances', sheetName='Loc_2_Cons')
        self.dock2cons = GF.dataframe_to_dictionary(fileName='Distances', sheetName='Dock_2_Cons')
        self.loc2loc = GF.dataframe_to_dictionary(fileName='Distances', sheetName='Loc_2_Loc')
        
        self.storage = storage
        
        # Adjust cons2loc to only show consolidation to corner-point distances
        self.cons2crnr = {cons:{k: self.cons2loc[cons][k] for k in self.cons2loc[cons].keys() if k[0] != 'C'} for cons in self.cons2loc.keys()}

    def retrieve_cons2crnr_list(self, cons):
        """ Return the distances to all corner-points from the given consolidation area """
        return [self.cons2loc[cons]]

    def retrieve_dock2cons(self, dock):
        """ Return the distance from dock to assigned consolidation area """
        return self.dock2cons[dock]
    
    def retrieve_cons2loc(self, cons, corner_point):
        """ Return the distance from the consolidation area to the corner-point (X1 - X13) """
        return self.cons2loc[cons.nr][corner_point]
    
    def retrieve_loc2loc(self, loc1, loc2):
        """ Return the distance from corner-point to corner-point """
        return self.loc2loc[loc1][loc2]
    
    def retrieve_loc2cons(self, corner_point, cons):
        """ Return the distance from the corner-point (X1 - X13) to the consolidation area """
        try:
            return self.loc2cons[corner_point][cons.nr]
        except TypeError:
            print(f'No Loc2cons for {corner_point} -- {cons.nr}')
    
    def calc_crnr2storage(self, corner, sa, loc):
        """ Return single way manhattan distance from given corner-point to a storage location""" 
        # Set starting dist to be half a driving lane, as measure start from middle of cornerpoint
        dist = GC.avg_driving / 2
        
        # If cornerpoint concerns the left-side of the driving lane        
        if corner == 'LD':
            return dist + sa.IO[loc[0]] * GC.loc_length + sa.storage_sections[loc[0]]['ipd'][loc[1]]
        
        # Else if cornerpoint conerns the right-side of the driving lane
        elif corner == 'RD':  # Right Drivinglane
            return dist + (sa.layout.shape[1] - sa.IO[loc[0]]) * GC.loc_length + sa.storage_sections[loc[0]]['ipd'][loc[1]]
        
        # Else if cornerpoint concerns the left-side of the wall side
        elif corner == 'LW':
            # If on the far left lane, only travel vertically
            if sa.IO[loc[0]] == 0:
                return dist + sa.storage_sections[loc[0]]['ipd'][-1] - sa.storage_sections[loc[0]]['ipd'][loc[1]]
            
            # Else travel all the way up and continue calculation as LD
            else:
                return dist + sa.storage_sections[loc[0]]['ipd'][-1] + sa.IO[loc[0]] * GC.loc_length + sa.storage_sections[loc[0]]['ipd'][loc[1]]
         
        # Else if cornerpoint concerns the right-side of the wall side
        elif corner == 'RW': 
            # If on the far right lane, only travel vertically
            if sa.IO[loc[0]] == (sa.layout.shape[1] - 1):
                return dist + sa.storage_sections[loc[0]]['ipd'][-1] - sa.storage_sections[loc[0]]['ipd'][loc[1]]
            
            # Else travel all the way up and continue calculation as RD
            else:
                return dist + sa.storage_sections[loc[0]]['ipd'][-1] + (sa.layout.shape[1] - sa.IO[loc[0]]) * GC.loc_length + sa.storage_sections[loc[0]]['ipd'][loc[1]]
        
        # If corner not found, assert error
        else:
            raise ValueError(f'corner: "{corner}" does not exist!')
    
    def calc_cons2storage_inarea(self, cons, sa, loc):
        """ Calculate immediate distance between consolidation area and location (if consolidation area is located within the storage area) """
        # Retrieve IO point of the consolidation area in the storage area
        try:
            IO = cons.in_area[sa.nr]
        except KeyError:
            raise KeyError(f'Calc_cons2loc_inarea: {cons.nr} is not located in SA {sa.nr}')
        
        # Return total distance (horizontal + vertical (half consolidation depth + half a driving lane (cons-path) + ipd))
        return abs(IO - sa.IO[loc[0]]) * GC.loc_length + (GC.depth_consolidation/2 + GC.avg_driving/2 + sa.storage_sections[loc[0]]['ipd'][loc[1]])
    
    def closest_corner(self, cons, sa, cons2crnr=True):
        """ Return the closest corner from the consolidation area, differentiating on direction due to path restrictions """
        # Calculate distances from consolidation to all available forward corners
        if cons2crnr:
            # Initialize looping variable for available consolidation to cornerpoint
            corners = GC.corner_list[sa.nr - 1]
            distances = np.array([self.retrieve_cons2loc(cons=cons, corner_point=c) if c != None else 9999999 for c in corners])
        
        # Calculate distance from cornerpoints to consolidation areas
        else:
            corners = GC.corner_list_full[sa.nr - 1]
            distances = np.array([self.retrieve_loc2cons(cons=cons, corner_point=c) if c != None else 9999999 for c in corners])
        
        # Return closest corner 
        return GC.corner_points_full_r[sa.nr][corners[np.argmin(distances)]]
        
    def cons2storage(self, cons, sa, loc):
        """ Return the distance to travel between the consolidation area and the storage location """
        # If the consolidation area is located in the storage area
        if sa.nr in cons.in_area.keys():    
            # Calculate and return distance from consolidation to storage location immediately
            return self.calc_cons2storage_inarea(cons=cons, sa=sa, loc=loc)
        
        # If consolidation area is not located in the storage area
        else:
            # Find the closest cornerpoint of the storage area from the consolidation area
            crnr = self.closest_corner(cons=cons, sa=sa)
            corner_point = GC.corner_points_full[sa.nr][crnr]

            # Calculate and return distance to travel from consolidation to storage location
            return self.retrieve_cons2loc(cons=cons, corner_point=corner_point) + self.calc_crnr2storage(corner=crnr, sa=sa, loc=loc)
    
    def storage2cons(self, loc, sa, cons):
        """ Return the distance from the storage location to the consolidation area """
        # If there are direction restrictions on the given storage area
        if sa.nr in GC.area_restrictions.keys():
            # Loop over restrictions and check if it is applicable
            for (side, direction) in GC.area_restrictions[sa.nr]:
                # If applicable, calculate distance from storage to destined cornerpoint
                if side == 'L' and sa.IO[loc[0]] == 0 or side == 'R' and sa.IO[loc[0]] == (sa.layout.shape[1]-1):
                    storage2crnr = self.calc_crnr2storage(corner=direction,
                                                          sa=sa,
                                                          loc=loc)                   
                    # Return sum of storage to corner and corner to consolidation area distance
                    return storage2crnr + self.retrieve_loc2cons(cons=cons, corner_point=GC.corner_points_full[sa.nr][direction])
                
                # If no restrictions continue function with following code
                else:
                    continue
        
        # If no (applicable) restriction found, find closest cornerpoint
        corner_point = self.closest_corner(cons=cons, sa=sa, cons2crnr=False)
        crnr = GC.corner_points_full[sa.nr][corner_point]
        
        # Return sum of storage to cornerpoint and cornerpoint to consolidation distance
        return self.calc_crnr2storage(corner=corner_point, sa=sa, loc=loc) + self.retrieve_loc2cons(corner_point=crnr, cons=cons)

    def retrieve_loc2loc_dist(self, loc1, loc2):
        """ Calculate travel distance from one location to another """
        # If both locations in the same storage area
        if loc1[0] == loc2[0]:
            # Retrieve storage area object
            sa = self.storage.areas[loc1[0]]
            
            # Retrieve storage section ipd of both locations
            loc1_ipd = sa.storage_sections[loc1[1][0]]['ipd'][loc1[1][1]]
            loc2_ipd = sa.storage_sections[loc2[1][0]]['ipd'][loc2[1][1]]
            
            # If both are in the same section
            if loc1[1][0] == loc2[1][0]:
                # Return the different in In Path Distance (IPD)
                return abs(loc1_ipd - loc2_ipd)
            
            # If in the same storage area but in different sections
            else:
                # Retrieve IO points of both sections
                loc1_IO = sa.IO[loc1[1][0]]
                loc2_IO = sa.IO[loc2[1][0]]
                
                # Return IPD values + horizontal distance
                return loc1_ipd + abs(loc1_IO - loc2_IO) * GC.loc_length + loc2_ipd
        
        # If both locations are in different storage areas
        else:
            # Retrieve storage areas of both storage locations
            sa1, sa2 = self.storage.areas[loc1[0]], self.storage.areas[loc2[0]]
            
            # Retrieve cornerpoints of both the storage areas
            crnr1, crnr2 = GC.corner_list[sa1.nr-1], GC.corner_list[sa2.nr-1]
            
            # Calculate all corner-point to corner-point distances
            from_crnr, to_crnr = [], []
            dist = []
            for c1 in crnr1:
                for c2 in crnr2:
                    from_crnr.append(c1)
                    to_crnr.append(c2)
                    
                    # If both corners are identical, break loop and return these
                    if c1 == c2:
                        dist.append(0)
                        break
                    
                    # If not, calculate distance between the two and add to comparison list
                    else:
                        dist.append(self.retrieve_loc2loc(loc1=c1, loc2=c2))
            
            # Retrieve closest corner-point to corner distance
            min_dist, min_index = min(dist), np.argmin(np.array(dist))
            sa1_crnr = GC.corner_points_full_r[sa1.nr][from_crnr[min_index]]
            sa2_crnr = GC.corner_points_full_r[sa2.nr][to_crnr[min_index]]
            
            # Calculate in storage area dist to get to the selected corner
            sa1_dist = self.calc_crnr2storage(corner=sa1_crnr, 
                                              sa=sa1, 
                                              loc=loc1[1])
            sa2_dist = self.calc_crnr2storage(corner=sa2_crnr, 
                                              sa=sa2, 
                                              loc=loc2[1])
            
            # Return total distance
            return sa1_dist + min_dist + sa2_dist
                    


    
    
    
    
    
    
    
    