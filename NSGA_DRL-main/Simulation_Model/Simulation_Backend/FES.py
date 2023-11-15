import heapq

class FES:
    """
    Description: The Future Event Set (FES) contains all the scheduled events.
    """

    def __init__(self):
        self.events = []

    def __str__(self):
        """ Print current events present in the Future Event Set """
        message = f'The FES currently contains {len(self.events)} events, namely: \n'
        sortedEvents = sorted(self.events)
        for event in sortedEvents:
            message += '  ' + str(event) + '\n'
        return message

    def add(self, event):
        """Add event"""
        heapq.heappush(self.events, event)

    def next(self):
        """Next event"""
        return heapq.heappop(self.events)

    def is_empty(self):
        """Check for empty queue"""
        return len(self.events) == 0
