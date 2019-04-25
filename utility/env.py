
import numpy as np
from params import *
from utility.utility import printh

class Environment:
    def __init__(self, all_cooked_time, all_cooked_bw, rnd_seed=0, _random=True):
        self.num_trace       = len(all_cooked_bw)
        self.all_cooked_time = all_cooked_time
        self.all_cooked_bw   = all_cooked_bw
        self._random         = _random
        self.trace_idx       = 0
        self.req_idx         = 0
        self.last_time       = 0.0
        self.req_time        = 0.0
        self.timer           = 0
        self.trace           = None #initialize
        # pick a random start, in a random trace file
        np.random.seed(rnd_seed)
        self.next_trace()
        (self.req_time, self.req_file) = \
                    self.get_next_request()
        pass

    def parse_action(self, action):
        idx = list(action).index(1)
        a_idx  = int(idx / A_VAL)   # action over which entry
        a_val  =     idx % A_VAL    # action with which segment
        a_type = 'D' if a_val==0 else 'A'
        return (a_type, a_idx, a_val)
    
    def get_trace(self, idx):
        return list(zip(self.all_cooked_time[idx], self.all_cooked_bw[idx]))

    def next_trace(self):
        self.timer += self.trace[-1][0] if self.trace else 0 #NOTE: always update timer!

        if self._random:
            self.trace_idx = np.random.randint(self.num_trace)
        else:
            self.trace_idx = 0
            # self.trace_idx += 1
            self.trace_idx %= 127
        
        #NOTE: update [trace[:][0], trace_ptr, req_time]
        _get_trace      = self.get_trace(self.trace_idx)
        self.trace = [(np.array(x)+[self.timer,0]) for x in _get_trace]
        self.trace_ptr  = 1
        self.req_time   = self.timer + self.req_time
        pass

    def validate_timer(self):
        #NOTE: trace_ptr, req_time
        if (self.trace_ptr>=len(self.trace)) or (self.req_time>=self.trace[-1][0]):
            self.next_trace()
        pass

    def get_next_request(self):
        # for time, fixed Poission
        # req_time = self.last_time + 1.0 + abs(np.random.poisson(REQ_MEAN))
        req_time = self.last_time + REQ_MEAN
        self.validate_timer()
        # for file, Zipf (F_DIM=6)
        zipf_dist = [np.power(i+1, -REQ_ZIPF) for i in range(F_DIM)]
        zipf_dist = [x/sum(zipf_dist) for x in zipf_dist]
        zipf_cumsum = np.cumsum(zipf_dist)
        req_file = (np.random.rand() > zipf_cumsum).argmin() #locate first '0'
        return (req_time, req_file)

    def get_segment(self, seg_idx):
        delay, received = 0.0, 0.0

        while True:
            _duration   = self.trace[self.trace_ptr][0] - self.last_time
            _throughput = self.trace[self.trace_ptr][1]*MB
            assert _duration>=0

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
            self.validate_timer()
            pass # End of While
        
        return delay

    def whats_next(self, storage, action):
        (a_type, a_idx, a_val) = self.parse_action(action)
        request_indicator, p1_delay, p2_delay = 0, 0, 0

        if a_type=='D':
            storage[a_idx] = 0
            p1_delay = self.trace[self.trace_ptr][0] - self.last_time
            self.last_time = self.trace[self.trace_ptr][0]
            self.trace_ptr += 1
            self.validate_timer()
            pass
        else: #ADD action
            # Phase I, Proactive Replacement
            p1_delay = self.get_segment(a_val)
            storage[a_idx] = a_val
            # Phase II, Bypass Request
            request_indicator = self.last_time>self.req_time
            if request_indicator:
                unhitted_segments = F_MAT[self.req_file] - set(storage)
                p2_delay = [self.get_segment(x) for x in unhitted_segments]
                p2_delay = sum(p2_delay)
                #Generate next request
                (self.req_time, self.req_file) = \
                    self.get_next_request()
                pass
            pass
        
        _req_file = self.req_file if request_indicator else -1
        return (request_indicator,
                _req_file,
                p1_delay,
                p2_delay,
                self.req_time)
    
    pass # End of Class Environment
