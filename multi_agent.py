#!/usr/bin/python3
import numpy as numpy
import tensorflow as tf
from queue import Queue
import multiprocessing as mp
from multiprocessing import Process, Pool

from utility import a3c, env
from utility.utility import load_trace

from params import *
NUM_AGENTS = mp.cpu_count()-1 # use n-1 core


global timestamps, bandwidths

def central_agent(params_qs, exp_qs):
    with tf.Session() as sess:
        actor   = a3c.ActorNetwork(sess, A_DIM, [S_DIM, S_LEN], ACTOR_LRATE)
        critic  = a3c.CriticNetowkr(sess, A_DIM, [S_DIM, S_LEN], CRITIC_LRATE)

        summary_ops, summary_vars = a3c.BaseNetwork.build_summaries()
        sess.run(tf.global_variables_initializer())
        # writer  = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)  # training monitor
        saver   = tf.train.Saver()  # save neural net parameters

        #TODO: load intermidiate NN model or not

        epoch = 0
        while True:
            #NOTE: Synchronous Parameter Update
            actor_params  = actor.get_network_params()
            critic_params = critic.get_network_params()
            [q.put([actor_params, critic_params]) for q in params_qs]

            # update gradients
            for exp_q in exp_qs:
                #TODO: exp_q.get()
                #TODO: compute gradients
                pass

            epoch += 1
            if epoch % MODEL_SAVE_INTERVAL == 0: #save the NN model
                saver.save(sess, './model/nn_ep_%d.ckpt'%epoch)
                pass
            pass
        pass
    pass

def agent(agent_id, params_q, exp_q):
    global timestamps, bandwidths
    net_env = env.Environment(timestamps, bandwidths, rnd_seed=agent_id)

    with tf.Session() as sess:
        actor   = a3c.ActorNetwork(sess, A_DIM, [S_DIM, S_LEN], ACTOR_LRATE)
        critic  = a3c.CriticNetowkr(sess, A_DIM, [S_DIM, S_LEN], CRITIC_LRATE)

        acotr_params, critic_params = params_q.get() #block until get
        actor.set_network_params(acotr_params)
        critic.set_network_params(critic_params)

        #TODO: Init Last State
        cache_mat = list()          # init::empty storage
        #TODO: Init Last Action
        #TODO: Init Empty Batch Record (state,action,reward)

        global_timer = 0
        while True:
            net_env.whats_next(cache_mat)

            #TODO: update timer
            #TODO: calculate reward
            #TODO: update state

            #TODO: determine next action
            # action_prob = actor.predict(np,reshapre(state, (1,S_DIM,S_LEN)))
            # action_cumsum = np.cumsum(action_prob)
            # action = ().argmax()

            # if len(r_batch)>=TRAIN_SEQ_LEN or episodic_flag:
                #TODO: report experience to coordinator
                # exp_q.put()
                # #empty all the batch record
                # pass
            
            #TODO: update batch record (state,action,reward)
            pass
        pass

    pass

def run():
    global timestamps, bandwidths
    # Setup
    np.random.seed(RANDOM_SEED)
    params_qs = [Queue() for i in range(NUM_AGENTS)]
    exp_qs    = [Queue() for i in range(NUM_AGENTS)]
    timestamps, bandwidths = load_trace('./training_traces')

    # create a coordinator and multiple agent processes
    coordinator = Process(central_agent, args=(params_qs, exq_qs))
    coordinator.start()
    # execute Pool of parallel agents (until the Poll is down)
    with Pool(NUM_AGENTS) as p:
        agent_params = zip(range(NUM_AGENTS), params_qs, exp_qs)
        p.map(agent, list(agent_params))
        pass
    pass

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(e)
    finally:
        exit()
    pass