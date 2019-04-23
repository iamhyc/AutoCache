from functools import reduce
# Unit Conversion
BIT  = 1
BYTE = 8
KB   = 1E3*BYTE
MB   = 1E6*BYTE
MS   = 1E-3

# Request Parameters
C_DIM           = 5         # cpacity of 5 segments
F_DIM           = 6         # total 6 files
F_MAT           = [ set(range(0,5)),    set(range(5, 12)),  set(range(12, 15)),
                    set(range(15,26)),  set(range(26, 30)), set(range(30, 40))
                  ] #total 40 segments = (5, 7, 3, 11, 4, 10)
A_VAL           = len(reduce(lambda x,y:x|y, F_MAT)) # action value range, [0,A_VAL]
REQ_MEAN        = 6000*MS   # Time interval: Poisson with average interval 6s
REQ_ZIPF        = 1.10      # Index Dist.  : Zipf parameter

# RL Params
S_DIM           = 3         # (past_download_time, past_download_bw, last_storage)
S_LEN           = 8         # (depth for *past* states)
S_MAT           = [S_LEN, S_LEN, C_DIM] # depth for states
A_DIM           = C_DIM*(A_VAL+1) # Action space dimesion, 5*41=205

GAMMA           = 0.99
ACTOR_LRATE     = 0.0001
CIRITC_LRATE    = 0.001
TRAIN_SEQ_LEN   = 100
MODEL_SAVE_INTERVAL = 100

# Model Params
SEG_SIZE        = 10*MB      # Each segment with same size as 10MB