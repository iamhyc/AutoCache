import numpy as numpy
import tensorflow as tf
import tflearn
from params import GAMMA

ENTROPY_EPS    = 1E-6
ENTROPY_WEIGHT = 0.5

def build_summaries():
    # (1) TD Loss
    td_loss = tf.Variable(0.)
    tf.summary.scalar("TD_loss", td_loss)
    # (2) Episode Reward
    eps_total_reward = tf.Variable(0.)
    tf.summary.scalar("Eps_total_reward", eps_total_reward)
    # (3) Average Entropy
    avg_entropy = tf.Variable(0.)
    tf.summary.scalar("Avg_entropy", avg_entropy)

    summary_ops = tf.summary.merge_all()
    summary_vars = [td_loss, eps_total_reward, avg_entropy]
    return summary_ops, summary_vars

def compute_entropy(vec):
    '''
    Given vector x, computes the entropy
    H(x) = - sum( p*log(p) )
    '''
    return -sum([x*np.log(x) if 0<x<1 else 0 for x in vec])

def compute_returns(x, gamma):
    '''
    Given vector x, computes a vector y such that
    y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
    '''
    ret = np.zeros(len(x))

    ret[-1] = x[-1]
    for i in reversed(range(len(x)-1)):
        y[i] = x[i] + gamma * y[i+1]
        pass

    return ret

def compute_gradients(s_batch, a_batch, r_batch, actor, critic):
    '''
    batch of s, a, r is from samples in a sequence
    the format is in np.array([batch_size, s/a/r_dim])
    '''
    # BA Size
    ba_size = s_batch.shape[0]
    # Value Batch
    v_batch = critic.predict(s_batch)
    # Reward Batch
    R_batch = np.zeros(r_batch.shape)
    R_batch[-1, 0] = v_batch[-1, 0] #NOTE: bootstrap from last state
    for t in reversed(range(ba_size - 1)):
        R_batch[t, 0] = r_batch[t] + GAMMA * R_batch[t + 1, 0]
    # TD Batch
    td_batch = R_batch - v_batch

    actor_gradients = actor.get_gradients(s_batch, a_batch, td_batch)
    critic_gradients = critic.get_gradients(s_batch, R_batch)
    return (actor_gradients, critic_gradients, td_batch)

def generate_splits(inputs, depth_l):
    splits = list()
    for i,depth in enumerate(depth_l):
        if depth>1:
            split = tflearn.conv_1d(self.inputs[:,i-1:i,depth], 128,4, activation='relu')
            split = tflearn.flatten(split)
            splits.append(split)
            pass
        elif depth==1:
            split = tflearn.fully_connected(self.inputs[:,i-1:i,-1], 128, activation='relu')
            splits.append(split)
            pass
        pass
    return splits

class ActorNetwork:
    def __init__(self, session, action_dim, state_dim, learning_rate):
        self.sess, self.a_dim, self.s_dim, self.lr_rate = \
            session, action_dim, state_dim, learning_rate
        
        # Create the actor network
        self.create_actor_network()
        # [network_params] Get all network parameters
        self.network_params = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')
        # [input_network_params, set_network_params_op] Set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(self.network_params[idx].assign(param))

        # [acts] selected action, 0-1 vector
        self.acts =             tf.placeholder(tf.float32, [None, self.a_dim])
        # [act_grad_wdights] This gradient will be provided by the critic network
        self.act_grad_weights = tf.placeholder(tf.float32, [None, 1])

        # [objective] Compute the objective (log action_vector and entropy)
        self.obj =  tf.reduce_sum(
                        tf.multiply(
                            tf.log(
                                tf.reduce_sum(
                                    tf.multiply(self.out, self.acts),
                                    reduction_indices=1,
                                    keepdims=True
                                )
                            ),
                            - self.act_grad_weights
                        )
                    )
                    + ENTROPY_WEIGHT * \
                        tf.reduce_sum(
                            tf.multiply(self.out, tf.log(self.out + ENTROPY_EPS))
                        )

        # [actor_gradients] Combine the gradients here
        self.actor_gradients = tf.gradients(self.obj, self.network_params)

        # [optimize] Optimization Op
        self.optimize = tf.train.RMSPropOptimizer(self.lr_rate) \
                        .apply_gradients(zip(self.actor_gradients, self.network_params))
        pass

    def create_actor_network(self):
        with tf.variable_scope('actor'):
            self.inputs = tflearn.input_data(shape=[None, self.s_dim[0], self.s_dim[1]])
            
            splits = generate_splits(self.inputs, S_MAT)
            merge_net = tflearn.merge(splits, 'concat')
            dense_net_0 = tflearn.fully_connected(merge_net, 128, activation='relu')
            
            self.out = tflearn.fully_connected(dense_net_0, self.a_dim, activation='softmax')
            pass
        pass

    def train(self, inputs, acts, act_grad_weights):
        self.sess.run(self.optimize, 
            feed_dict={
                self.inputs: inputs,
                self.acts: acts,
                self.act_grad_weights: act_grad_weights
            })
        pass

    def predict(self, inputs):
        ret = self.sess.run(self.out,
                feed_dict={
                    self.inputs: inputs
                })
        return ret

    def get_gradients(self, inputs, acts, act_grad_weights):
        ret = self.sess.run(self.actor_gradients,
                feed_dict={
                    self.inputs: inputs,
                    self.acts: acts,
                    self.act_grad_weights: act_grad_weights
                })
        return ret

    def apply_gradients(self, this_actor_gradients):
        ret = self.sess.run(self.optimize,
                feed_dict={
                    i: d for i, d in zip(self.actor_gradients, this_actor_gradients)
                })
        return ret

    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, this_input_network_params):
        self.sess.run(self.set_network_params_op,
            feed_dict={
                i: d for i, d in zip(self.input_network_params, this_input_network_params)
            })
        pass

    pass

class CriticNetwork:
    def __init__(self, session, action_dim, state_dim, learning_rate):
        self.sess, self.a_dim, self.s_dim, self.lr_rate = \
            session, action_dim, state_dim, learning_rate

        # Create the critic network
        self.create_critic_network()

        # [network_params] Get all network parameters
        self.network_params = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')

        # [input_network_params, set_network_params_op] Set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(self.network_params[idx].assign(param))

        # [td_target] Network target V(s)
        self.td_target = tf.placeholder(tf.float32, [None, 1])
        # [td] Temporal Difference, will also be weights for actor_gradients
        self.td = tf.subtract(self.td_target, self.out)
        # [loss] Mean square error
        self.loss = tflearn.mean_square(self.td_target, self.out)
        # [gradient] Compute critic gradient
        self.critic_gradients = tf.gradients(self.loss, self.network_params)

        # Optimization Op
        self.optimize = tf.train.RMSPropOptimizer(self.lr_rate) \
                        .apply_gradients(zip(self.critic_gradients, self.network_params))
        pass
    
    def create_critic_network(self):
        with tf.variable_scope('critic'):
            self.inputs = tflearn.input_data(shape=[None, self.s_dim[0], self.s_dim[1]])

            splits = generate_splits(self.inputs, S_MAT)
            merge_net = tflearn.merge(splits, 'concat')
            dense_net_0 = tflearn.fully_connected(merge_net, 128, activation='relu')

            self.out = tflearn.fully_connected(dense_net_0, 1, activation='linear')
            pass
        pass

    def train(self, inputs, td_target):
        return self.sess.run([self.loss, self.optimize], feed_dict={
            self.inputs: inputs,
            self.td_target: td_target
        })

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })

    def get_td(self, inputs, td_target):
        return self.sess.run(self.td, feed_dict={
            self.inputs: inputs,
            self.td_target: td_target
        })

    def get_gradients(self, inputs, td_target):
        return self.sess.run(self.critic_gradients, feed_dict={
            self.inputs: inputs,
            self.td_target: td_target
        })

    def apply_gradients(self, this_critic_gradients):
        return self.sess.run(self.optimize, feed_dict={
            i: d for i, d in zip(self.critic_gradients, this_critic_gradients)
        })

    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, this_input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, this_input_network_params)
        })
        pass

    pass