import numpy as np
import tensorflow as tf


# -------------- basic specifications ----------------- #
"""
state_spec: state vector, 
            dtype must be np.float32, shape consistent with tf, shape[-1] is the dimension of state
"""
STATE_SPEC = dict(dtype=np.float32, shape=[None, 50])

"""
action_spec: action vector,
             dtype must be np.float32, shape consistent with tf, shape[-1] is the dimension of action
"""
ACTION_SPEC = dict(dtype=np.float32, shape=[None, 5])
"""
reward_spec: reward vector,
             dtype must be np.int32, shape consistent with tf, shape[-1] is the dimension of reward,
             one-hot vector, conventional output will be index of non-zero;
             input dtype to transition function is fixed as np.float32           
"""
REWARD_SPEC = dict(dtype=np.int32, shape=[None, 2])
"""
obs_spec: obs vector,
          dtype must be np.float32, shape consistent with tf, shape[-1] is the dimension of obs
          when obs_equal_state == True, no effect, as if obs_spec == state_spec;
                                  else, it is preferred obs_spec["shape"][-1] < state_spec["shape"][-1]
"""
OBS_SPEC = dict(dtype=np.float32, shape=[None, 50])
"""
terminal_spec: terminal vector,
               dtype must be np.int32, shape consistent with tf, shape[-1] is default 2,
               one-hot vector, conventional output will be terminal[1]>0, True means terminal;
"""
TERMINAL_SPEC = dict(dtype=np.int32, shape=[None, 2])
"""
model_transition_spec: NN layers specification for transition function
               list specify units values for each layer from input from key
                   e.g., state_layers=[50] means the units of the layer connected from state input is 50
                   e.g., combine means layers after concatenate hidden layers from all inputs
               nonlinear is the activation function for each layer
"""
MODEL_TRANSITION_SPEC = dict(action_layers=[],
                             state_layers=[50],
                             combine_layers=[50],
                             reward_layers=[10],
                             nonlinear=tf.nn.tanh)
"""
model_reward_spec: NN layers specification for transition function
               list specify units values for each layer from input from key
                   e.g., state_layers=[50] means the units of the layer connected from state input is 50
                   e.g., combine means layers after concatenate hidden layers from all inputs
               nonlinear is the activation function for each layer
"""
MODEL_REWARD_SPEC = dict(action_layers=[],
                         state_layers=[50],
                         combine_layers=[50],
                         nonlinear=tf.nn.tanh)

# ---------------- special configure for different setting -------------- #
"""
terminal_prior: bias prior for terminal distribution
"""
TERMINAL_PRIOR = np.array([0.99, 0.01])
"""
state_next_randomness: top-level randomness for state_next output in transition function
                    noise is default Gaussian
                    if 0.0: deterministic state_next given input
                    elif 0.0 <= float <= 1.0: stochastic state_next given input
                    elif 1.0: stochastic state_next regardless of input
"""
STATE_NEXT_RANDOMNESS = 0.0
"""
state_next_action_removal: flag whether action input is disabled in state_next prediction in transition function
                    if True: action input is replaced with zero vector of the same shape,
                        which means state_next is not related to action, and then environment may reduced to dynamic MAB
                    else: default MDP setting
"""
STATE_NEXT_ACTION_REMOVAL = False
"""
obs_equal_state: flag whether obs == state
                    if True: obs_spec is disabled, as true MDP environment
                    else: suggesting obs_spec["shape"][-1] < state_spec["shape"][-1], POMDP environment
"""
OBS_EQUAL_STATE = False
"""
reward_prior: bias prior for reward distribution
"""
REWARD_PRIOR = np.array([0.7, 0.3])
"""
reward_randomness: top-level randomness for reward output in reward function
                    if 0.0: deterministic reward given input
                    elif 0.0 <= float <= 1.0: stochastic reward given input
                    elif 1.0: stochastic reward regardless of input
"""
REWARD_RANDOMNESS = 0.0
"""
reward_action_removal: flag whether action input is disabled in reward function
                    if True: action input is replaced with zero vector of the same shape,
                        which means reward is not related to action, and then environment is only about future rewards 
                        determined by states
                    else: default MDP setting
"""
REWARD_ACTION_REMOVAL = False
"""
reward_state_removal: flag whether state input is disabled in reward function
                    if True: state input is replaced with zero vector of the same shape,
                        which means reward is not related to state, and then environment is only about current rewards
                        determined by actions, and state transition becomes completely irrelevant
                    else: default MDP setting
"""
REWARD_STATE_REMOVAL = False
"""
seed: random seed
"""
SEED = 2018
"""
env_name: name of environment 
"""
ENV_NAME = "default"
with open("./mytest/configs/" + ENV_NAME + "_SEED%" % SEED, "w") as f:
    var_dist = locals()
    var_names = sorted(var_dist.keys())
    for var_name in var_names:
        if "__" not in var_name and var_name not in ["f", "np", "tf", "var_dist", "var_names", "var_name", "ENV_NAME"]:
            f.write("{} = {}\n".format(var_name, str(var_dist[var_name])))