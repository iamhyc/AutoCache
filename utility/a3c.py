import numpy as numpy
import tensorflow as tf
import tflearn

GAMMA          = 0.99
ENTROPY_EPS    = 1E-6
ENTROPY_WEIGHT = 0.5

class BaseNetwork:
    def __init__(self, sess, s_dim, a_dim, lr_rate):
        self.sess = sess
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.lr_rate = lr_rate
        pass

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

    pass

class ActorNetwork(BaseNetwork):
    def __init__(self, session, action_dim, state_dim, learning_rate):
        super().__init__(session, action_dim, state_dim, learning_rate)
        
        # Create the actor network
        self.inputs, self.out = self.create_actor_network()
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

        # [acts] Selected action, 0-1 vector
        self.acts = tf.placeholder(tf.float32, [None, self.a_dim])
        # [act_grad_wdights] This gradient will be provided by the critic network
        self.act_grad_weights = tf.placeholder(tf.float32, [None, 1])

        # [obj?] Compute the objective (log action_vector and entropy)
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
        self.optimize = tf.train.RMSPropOptimizer(self.lr_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        pass
    
    #TODO: design state space and the network
    def create_actor_network(self):
        with tf.variable_scope('actor'):
            inputs = tflearn.input_data(shape=[None, self.s_dim[0], self.s_dim[1]])
            
            #TODO: design state space and the network

            merge_net = tflearn.merge([], 'concat')
            dense_net_0 = tflearn.fully_connected(merge_net, 128, activation='relu')
            out = tflearn.fully_connected(dense_net_0, self.a_dim, activation='softmax')

            return inputs, out
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

    def apply_gradients(self, actor_gradients):
        ret = self.sess.run(self.optimize,
                feed_dict={
                    i: d for i, d in zip(self.actor_gradients, actor_gradients)
                })
        return ret

    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op,
            feed_dict={
                i: d for i, d in zip(self.input_network_params, input_network_params)
            })
        pass

    pass

class CriticNetwork(BaseNetwork):
    def __init__(self, session, action_dim, state_dim, learning_rate):
        super().__init__(session, action_dim, state_dim, learning_rate)

        # Create the critic network
        self.inputs, self.out = self.create_critic_network()

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
        self.optimize = tf.train.RMSPropOptimizer(self.lr_rate).\
            apply_gradients(zip(self.critic_gradients, self.network_params))
        pass
    
    #TODO: design space and neural network
    def create_critic_network(self):
        with tf.variable_scope('critic'):
            inputs = tflearn.input_data(shape=[None, self.s_dim[0], self.s_dim[1]]) #(batch_axis, states, depth)

            #TODO: design space and neural network

            merge_net = tflearn.merge([], 'concat')
            dense_net_0 = tflearn.fully_connected(merge_net, 128, activation='relu')
            out = tflearn.fully_connected(dense_net_0, 1, activation='linear')

            return inputs, out
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

    def apply_gradients(self, critic_gradients):
        return self.sess.run(self.optimize, feed_dict={
            i: d for i, d in zip(self.critic_gradients, critic_gradients)
        })

    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })
        pass

    pass