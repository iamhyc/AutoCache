
import numpy as np
from params import *

class Environment:
    def __init__(self, all_cooked_time, all_cooked_bw, rnd_seed=0):
        self.num_trace       = len(all_cooked_bw)
        self.all_cooked_time = all_cooked_time
        self.all_cooked_bw   = all_cooked_bw
        # pick a random start, in a random trace file
        np.random.seed(rnd_seed)
        trace_idx  = np.random.randint(num_trace)
        self.switch_trace_file(trace_idx)
        # request generator
        (self.req_time, self.req_file) = \
            self.get_next_request()
        pass
    
    def get_next_request(self):
        # for time, Poission
        req_time = self.last_time + np.random.poisson(REQ_MEAN)
        if req_time > self.trace[-1][0]: #in case of extreme condition
            req_time = req_time - self.trace[-1][0]
        # for file, Zipf (F_DIM=6)
        zipf_dist = [(i+1)^(-REQ_ZIPF) for i in range(F_DIM)]
        zipf_dist = [x/sum(zipf_dist) for x in zipf_dist]
        zipf_cumsum = np.cumsum(zipf_dist)
        req_file = (np.random.rand() > zipf_cumsum).index(0) #locate first '0'
        return (req_time, req_file)

    def switch_trace_file(self, idx):
        self.trace      = self.get_trace(idx)
        self.trace_ptr  = np.random.randint(len(self.trace))
        self.last_time  = self.trace[max(self.trace_ptr-1, 0)][0]
        self.end_of_trace = False
        pass

    def get_trace(self, idx):
        return list(zip(self.all_cooked_time[idx], self.all_cooked_bw[idx]))
    
    def get_segment(self, seg_idx, bypass_request=0):
        delay, received = 0.0, 0.0

        while (self.trace[self.trace_ptr] < self.req_time)|(bypass_request):
            _duration   = self.trace[self.trace_ptr][0] - self.last_time
            _throughput = self.trace[self.trace_ptr][1]*MB

            if received+_throughput*_duration > SEG_SIZE:
                fractional_time = \
                    (SEG_SIZE-received)/_throughput
                delay           += fractional_time
                self.last_time  += fractional_time
                break # End of Segment Downloading
            else:
                received += _throughput*_duration
                delay    += _duration
                self.last_time = self.trace[self.trace_ptr][0]
                pass
            
            self.trace_ptr += 1
            if self.trace_ptr>=len(self.trace): #loopback this trace
                self.end_of_trace = True
                self.trace_ptr = 0
                self.last_time = 0.0
                pass
            pass
        
        request_indicator = (self.trace[self.trace_ptr]>=self.req_time)
        return (delay, request_indicator)

    def parse_action(self, action):
        idx = action.index(1)
        a_idx  = int(idx / A_VAL)   # action over which entry
        a_val  =     idx % A_VAL    # action with which segment
        a_type = 'D' if a_val==0 else 'A'
        return (a_type, a_idx, a_val)

    def whats_next(self, storage, action):
        (a_type, a_idx, a_val) = self.parse_action(action)
        request_indicator, p1_delay, p2_delay = 0, 0, 0

        # Phase I, Proactive Replacement
        if a_type=='D':
            storage[a_idx] = 0
            pass
        elif a_type=='A':
            (p1_delay, request_indicator) = self.get_segment(a_val)
            storage[a_idx] = a_val
            # Phase II, Bypass Request
            if request_indicator:
                #Bypass unhitted segments (FIXME: sub-optimal algorithm)
                unhitted_segments = set(storage) - F_MAT[self.req_file]
                p2_delay = [self.get_segment(x, True)[0] for x in unhitted_segments]
                p2_delay = sum(p2_delay)
                #Next Phase-I: switch trace file
                if self.end_of_trace:
                    self.switch_trace_file()
                (self.req_time, self.req_file) = \
                    self.get_next_request(self.trace[self.trace_ptr][0], 0)
                pass
            pass

        return (request_indicator,
                req_file,
                p1_delay,
                p2_delay,
                storage)
    
    pass # End of Class Environment
