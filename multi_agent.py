#!/usr/bin/python3
import numpy as numpy
import tensorflow as tf
from utility import a3c, env
from utility.utility import load_trace

def compute_gradients(s_batch, a_batch, r_batch, terminal, actor, critic):
    '''
    batch of s, a, r is from samples in a sequence
    the format is in np.array([batch_size, s/a/r_dim])
    terminal is True when sequence ends as a terminal state
    '''
    # BA Size
    ba_size = s_batch.shape[0]
    # Value Batch
    v_batch = critic.predict(s_batch)
    # Reward Batch
    R_batch = np.zeros(r_batch.shape)
    R_batch[-1, 0] = 0 if terminal else v_batch[-1, 0] # else boot strap from last state
    for t in reversed(range(ba_size - 1)):
        R_batch[t, 0] = r_batch[t] + GAMMA * R_batch[t + 1, 0]
    # TD Batch
    td_batch = R_batch - v_batch

    actor_gradients = actor.get_gradients(s_batch, a_batch, td_batch)
    critic_gradients = critic.get_gradients(s_batch, R_batch)
    return (actor_gradients, critic_gradients, td_batch)

def main():
    pass

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
    finally:
        exit()
    pass