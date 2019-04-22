
import numpy as np
from params import *

class Environment:
    def __init__(self, all_cooked_time, all_cooked_bw, rnd_seed=0):
        self.num_trace       = len(all_cooked_bw)
        self.all_cooked_time = all_cooked_time
        self.all_cooked_bw   = all_cooked_bw
        # pick a random start, in a random trace file
        np.random.seed(rnd_seed)
        self.trace_idx = np.random.randint(num_trace)
        self.trace = self.get_trace(self.trace_idx)
        self.trace_ptr = np.random.randint(len(self.curr_trace))
        # request generator
        (self.req_time, self.req_file) = \
            get_next_request(self.trace[self.trace_ptr][0], 0)
        pass
    
    def get_next_request(last_time, last_file):
        # for time, Poission; for file, Zipf (F_DIM=6)
        zipf_dist = [(i+1)^(-REQ_ZIPF) for i in range(F_DIM)]
        zipf_dist = [x/sum(zipf_dist) for x in zipf_dist]
        zipf_cumsum = np.cumsum(zipf_cumsum)

        req_time = np.random.poisson(REQ_MEAN)
        req_file = (np.random.rand() > zipf_dist).index(0) #locate first '0'
        return (req_time, req_file)

    def get_trace(self, idx):
        return list(zip(self.all_cooked_time[idx], self.all_cooked_bw[idx]))

    def whats_next(storage, action):

        # Phase I, Proactive Replacement
        if action_add:
            #p1_delay
            pass
        elif action_delete:
            #no time consumed
            pass
        
        # Phase II, Bypass Request
        if request_happend:
            #p2_delay
            pass

        return (request_indicator,
                p1_delay,
                p2_delay,
                storage)
        pass
    
    pass
