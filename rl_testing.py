#!/usr/bin/python3
from sys import argv
from os import makedirs
import numpy as numpy
import tensorflow as tf
from utility import a3c, env
from utility.utility import load_trace

def single_agent():
    global timestamps, bandwidths
    net_env = env.Environment(timestamps, bandwidths, rnd_seed=42,
                _random=False, _idx=42) #NOTE: This env is fixed

    with tf.Session() as sess:
        actor   = a3c.ActorNetwork(sess,  A_DIM, [S_DIM, S_LEN], ACTOR_LRATE)
        critic  = a3c.CriticNetwork(sess, A_DIM, [S_DIM, S_LEN], CRITIC_LRATE)
        sess.run(tf.global_variables_initializer())
        saver   = tf.train.Saver()
        saver.restore(sess, NN_MODEL)

        #Init State, Action
        action_vec    = np.zeros(A_DIM)
        action_vec[0] = 1
        storage       = np.zeros(C_DIM)
        _state  = [np.zeros((S_INFO, S_LEN))]
        #Init Empty Batch Record #NOTE: no need for testing

        _timer = 0
        while True:
            # 1) Env Simulation
            (req_flag, req_file, p1_delay, p2_delay,storage) = \
                net_env.whats_next(storage, action_vec)
            _reward = -p2_delay #NOTE: Maximize the MINUS cost

            #TODO: logging what you care and plot!

            # 2) State Update
            _state = np.roll(state, -1, axis=1)
            _state[0, -1] = p1_delay                #NOTE: last download time
            _state[1, -1] = SEG_SIZE/p1_delay       #NOTE: last download bandwidth
            _state[2, :C_DIM] = np.array(storage)   #NOTE: last storage

            # 3) Action Update
            action_prob = actor.predict(np,reshapre(_state, (1,S_DIM,S_LEN)))
            entropy_record.append(a3c.compute_entropy(action_prob[0])) #update entropy
            action_cumsum = np.cumsum(action_prob)
            action_idx = (action_cumsum > np.random.randint(1,1000)/1000.0).argmax()
            action_vec = np.zeros(A_DIM)
            action_vec[action_idx] = 1

            # 4) If End of Testing
            #TODO: Where's the end of testing?

            pass

        pass

    pass

def main(model):
    global timestamps, bandwidths
    np.random.seed(42)
    os.makedirs('./results', exist_ok=True)
    timestamps, bandwidths = load_trace('./test_traces')

    single_agent()
    
    pass

if __name__ == "__main__":
    try:
        if len(argv) > 1:
            run(argv[1])
        else:
            print('Please specify NN Model first!')
    except Exception as e:
        print(e)
    finally:
        exit()