from Simulation_Model import GC

class Event:
    """
    Description: Create events for the future event set.
    Input:  - typ: What kind of event does it concern
            - cur_time = current time (upon creation)
            - time: When will this event take place
            - truck: Truck applicable to the event (if available)
            - cons_lane: Consolidation lane applicable to the event (if available)
            - job: Job applicable to the event (if available)
    """

    def __init__(self, typ, cur_time, time, truck=None, cons_lane=None, job=None):
        self.type = typ
        self.creation_time = cur_time
        self.time = time
        self.truck = truck
        self.cons_lane = cons_lane
        self.job = job

    def __lt__(self, other):
        """ Define sorting order of events in the FES """
        return self.time < other.time
    
    def __str__(self):
        """ Print the details of a given event """
        return f' ----- Event: {GC.EVENT_NAMES[self.type]} ----- \n \t -Creation: {self.creation_time} \n \t -Execution_time: {self.time}'
