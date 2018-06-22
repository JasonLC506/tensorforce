import tensorforce as tforce
import tensorflow as tf
from tensorforce.environments import Environment
import numpy as np
import warnings

from reward import Reward
from transition import Transition
from config import *

# # default #
# STATE_SPEC = dict(dtype=np.float32, shape=[None, 50])
# ACTION_SPEC = dict(dtype=np.float32, shape=[None, 5])
# REWARD_SPEC = dict(dtype=np.int32, shape=[None, 2])
# OBS_SPEC = dict(dtype=np.float32, shape=[None, 50])
# TERMINAL_SPEC = dict(dtype=np.int32, shape=[None, 2])
# MODEL_TRANSITION_SPEC = dict(action_layers=[],
#                              state_layers=[50],
#                              combine_layers=[50],
#                              reward_layers=[10],
#                              nonlinear=tf.nn.tanh)
# MODEL_REWARD_SPEC = dict(action_layers=[],
#                          state_layers=[50],
#                          combine_layers=[50],
#                          reward_layers=[10],
#                          nonlinear=tf.nn.tanh)
#
# # termporary solution for terminal and reward bias #
# # dependent TERMINAL_SPEC, REWARD_SPEC #
# TERMINAL_PRIOR = np.array([0.99, 0.01])
# STATE_NEXT_RANDOMNESS = 0.0
# STATE_NEXT_ACTION_REMOVAL = False
# OBS_EQUAL_STATE = False
# REWARD_PRIOR = np.array([0.7, 0.3])
# REWARD_RANDOMNESS = 0.0
# REWARD_ACTION_REMOVAL = False
# REWARD_STATE_REMOVAL = False
#
# SEED = 2018


class SearchUser(Environment):
    def __init__(self,
                 state_spec=STATE_SPEC,
                 obs_spec=OBS_SPEC,
                 action_spec=ACTION_SPEC,
                 reward_spec=REWARD_SPEC,
                 terminal_spec=TERMINAL_SPEC,
                 model_reward_spec=MODEL_REWARD_SPEC,
                 model_transition_spec=MODEL_TRANSITION_SPEC,
                 reward_model=Reward,
                 transition_model=Transition,
                 global_user=None,
                 seed=SEED,
                 terminal_prior=TERMINAL_PRIOR,
                 state_next_randomness=STATE_NEXT_RANDOMNESS,
                 state_next_action_removal=STATE_NEXT_ACTION_REMOVAL,
                 obs_equal_state=OBS_EQUAL_STATE,
                 reward_prior=REWARD_PRIOR,
                 reward_randomness=REWARD_RANDOMNESS,
                 reward_action_removal=REWARD_ACTION_REMOVAL,
                 reward_state_removal=REWARD_STATE_REMOVAL,
                 env_name=ENV_NAME):
        """
        :param global_user: if this user is a replicate of a global_user, if yes, type(global_user) = SearchUser
        """

        # specification for variables #
        self.state_spec = state_spec
        self.obs_spec = obs_spec
        self.action_spec = action_spec
        self.reward_spec = reward_spec
        self.terminal_spec = terminal_spec
        self.model_reward_spec = model_reward_spec
        self.model_transition_spec = model_transition_spec

        self.terminal_prior = terminal_prior
        self.state_next_randomness = state_next_randomness
        self.state_next_action_removal = state_next_action_removal
        self.obs_equal_state = obs_equal_state
        self.reward_prior = reward_prior
        self.reward_randomness = reward_randomness
        self.reward_action_removal = reward_action_removal
        self.reward_state_removal = reward_state_removal

        # variables #
        self.state = None                                               # current state
        self.obs = None                                                 # current observation
        self.action = None                                              # current action
        self.reward = None                                              # current reward
        self.state_next = None                                          # next state
        self.terminal = None                                            # flag of episode termination
        self.t = None                                                   # time step within this user

        np.random.seed(seed)
        tf.set_random_seed(seed)
        # parameters/model #
        self.reward_model = reward_model(
                                    state_spec=self.state_spec,
                                    action_spec=self.action_spec,
                                    reward_spec=self.reward_spec,
                                    model_spec=self.model_reward_spec,
                                    prior=self.reward_prior,
                                    randomness=self.reward_randomness,
                                    action_removal=self.reward_action_removal,
                                    state_removal=self.reward_state_removal
                                   )                                    # reward model
        self.transition_model = transition_model(
                                    state_spec=self.state_spec,
                                    action_spec=self.action_spec,
                                    reward_spec=self.reward_spec,
                                    obs_spec=self.obs_spec,
                                    terminal_spec=self.terminal_spec,
                                    model_spec=self.model_transition_spec,
                                    terminal_prior=self.terminal_prior,
                                    state_next_randomness=self.state_next_randomness,
                                    state_next_action_removal=self.state_next_action_removal,
                                    obs_equal_state=self.obs_equal_state
                                   )                                    # state transition model
        self._model_init(global_user=global_user)

        self.env_name = env_name

    def reset(self,
              state=None):
        """
        :param state: current state assign
        """
        self._state_init(state, self.state_spec)
        self.terminal = np.array([1, 0])
        self.t = 0
        self.obs = self._obs(state_next=self.state)

        return self.obs

    def execute(self, actions, train=False, output_conventional=True):
        """
        take one step on receiving an action
        :param train: flag on train mode
        :param ouput_conventional: flag if terminal and reward are in conventional format (bool_, float)
        """
        if self.terminal[1] > 0:
            return None, 1, 0                                             # a new episode should be set, self.reset()

        # assert action.shape == self.action_spec                         # check action taken legal
        actions = self._sanity_check_action(actions)
        self.action = actions.squeeze()                                   # tforce default ndim=2

        # reward #
        self.reward, reward_dist = self._reward(state=self.state, action=self.action, state_next=self.state_next)
        self.state_next = self._transition(state=self.state, action=self.action, reward=self.reward)
        self.terminal, terminal_dist = self._terminal(state_next=self.state_next)
        self.obs = self._obs(state_next=self.state_next)

        # train #
        if train:
            # when training environment model #
            raise NotImplementedError

        # move on #
        self._state_init(self.state_next)
        self.t += 1

        if output_conventional:
            return self.obs, self.terminal[1], float(np.argmax(self.reward))
        else:
            return self.obs, self.terminal, self.reward

    def _state_init(self, state=None, state_spec=None):
        if state_spec is None:
            state_spec = self.state_spec
        if state is not None:
            # assert state.shape == state_spec                            # check state assigned legal
            self.state = state
        else:
            self.state = np.random.random(size=state_spec['shape'][-1])

    def _model_init(self, global_user=None):
        """
        do initialization on self.reward_model, self.transition_model
        :param global_user:
        """
        if global_user is None:
            global_reward_model = None
            global_transition_model = None
        else:
            global_reward_model = global_user.reward_model
            global_transition_model = global_user.transition_model
        self.reward_model.initialize(global_reward_model)
        self.transition_model.initialize(global_transition_model)

    def _reward(self, state=None, action=None, state_next=None, t=None):
        """
        calculate reward
        :param state: current state
        :param action: current action
        :param state_next: next state (optional)
        :return: reward
        """
        return self.reward_model.generate(state=state, action=action, state_next=state_next)

    def _transition(self, state=None, action=None, reward=None):
        """
        calculate next state
        :param state: current state
        :param action: current action
        :param reward: current reward (optional)
        :return: state_next
        """
        return self.transition_model.generate(state=state, action=action, reward=reward)

    def _terminal(self, state_next=None):
        """
        calculate whether terminal
        :return: terminal
        """
        return self.transition_model.terminal(state=state_next, prior=self.terminal_prior)

    def _obs(self, state_next=None):
        """
        calculate observation from state for next time step
        :return: obs
        """
        return self.transition_model.obs(state=state_next)

    @property
    def states(self):
        """
        :return: obs_spec, name to be consistent with agent definition in tforce
        """
        return self._spec_format_transform(self.obs_spec)

    @property
    def actions(self):
        """
        :return: action_spec,  name to be consistent with agent definition in tforce
        """
        action_spec = self._spec_format_transform(self.action_spec)
        return action_spec

    def _spec_format_transform(self, spec):
        # transform spec to be consistent #
        new_spec = spec.copy()
        new_spec['type'] = new_spec['dtype']
        if new_spec['type'] == int or new_spec['type'] == np.int32 or new_spec['type'] == np.int64:
            new_spec['type'] = 'int'
        else:
            new_spec['type'] = 'float'
        del new_spec['dtype']
        if 'shape' in new_spec:
            if new_spec['shape'][0] is None:
                new_spec['shape'] = (new_spec['shape'][-1],)
        return new_spec

    def __str__(self):
        return "SearchUser_" + self.env_name

    def _sanity_check_action(self, actions):
        if actions.shape[-1] != self.action_spec['shape'][-1]:
            warnings.warn("actions type wrong, instead use random action")
            print("type: %s" % str(type(actions)))
            print("actions: %s" % str(actions))
            return np.random.random(size=ACTION_SPEC['shape'][-1]).astype(ACTION_SPEC['dtype'])
        else:
            return actions


if __name__ == "__main__":
    state = np.random.random(size=STATE_SPEC['shape'][-1]).astype(STATE_SPEC['dtype'])
    action = np.random.random(size=ACTION_SPEC['shape'][-1]).astype(ACTION_SPEC['dtype'])
    search_user = SearchUser()

    obs = search_user.reset(state=state)
    for i in range(10):
        state = search_user.state
        obs, terminal, reward = search_user.execute(action, output_conventional=True)
        print("------- %d time step -------" % i)
        print("state: %s" % str(state))
        print("action: %s" % str(action))
        print("obs: %s" % str(obs))
        print("terminal: %s" % str(terminal))
        print("reward: %s" % str(reward))
        state_next = search_user.state
        print("state_next: %s" % str(state_next))
