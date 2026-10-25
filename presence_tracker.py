import time
class PresenceTracker:
    def __init__(self):
        self.start_time = time.perf_counter()
        self.current_state = None
        self.state_start = None
        self.events = []
        self.pending_state = None
        self.pending_since = None
        self.smoothing_threshold = 1.0
    
    def update(self, face_present):
        # Get current elapsed session time
        current_time = time.perf_counter()
        elapsed = current_time - self.start_time

        # Convert face_present Boolean into
        # "PRESENT" or "ABSENT"

        new_state = "PRESENT" if face_present else "ABSENT"


        # Handle the first observation --- chronos start tracking present, not save an event yet
        if self.current_state is None:
            self.current_state = new_state
            self.state_start = elapsed
            return

        #  If the state changed:
        # - Close the previous event
        # - Store start, end, duration
        # - Update current_state
        # - Update state_start

        if new_state == self.current_state:
            self.pending_state = None
            self.pending_since = None 
            return
        
        if self.pending_state != new_state:
            # Record new pending state
            self.pending_state = new_state
            # Record when it started
            self.pending_since = elapsed
            return

        # Check how long the candidate has persisted
        pending_duration = elapsed - self.pending_since

        if pending_duration >= self.smoothing_threshold:

            duration = self.pending_since - self.state_start
  

            event = { 
                "state": self.current_state,
                "start": self.state_start,
                "end": self.pending_since,
                "duration": duration
            }
            self.events.append(event)

            self.current_state = new_state
            self.state_start = self.pending_since

            # clear pending
            self.pending_state = None
            self.pending_since = None




    def get_summary(self):
        total_present = 0
        total_absent = 0

        # 1. Sum completed events
        for event in self.events:
            if event["state"] == "PRESENT":
                total_present += event['duration']
            else:
                total_absent += event['duration']

        # 2. Include the currently active event
        if self.current_state is not None:
            elapsed = time.perf_counter() - self.start_time
            active_duration = elapsed - self.state_start

            if self.current_state == "PRESENT":
                total_present = total_present + active_duration
            else:
                total_absent = total_absent + active_duration

        # 3. Return a dictionary
        return {
            "present": total_present,
            "absent": total_absent,
            "total": total_absent + total_present
        }

   
        