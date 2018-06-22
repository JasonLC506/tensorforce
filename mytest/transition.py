import tensorflow as tf
import numpy as np
import scipy
import warnings

from baseNet import NN
from reward import *


OBS_SPEC = dict(dtype=np.float64, shape=[None, 20])
TERMINAL_SPEC = dict(dtype=np.int32, shape=[None, 2])
MODEL_SPEC["reward_layers"] = []


class Transition(NN):
    def __init__(self,
                 state_spec=STATE_SPEC,
                 action_spec=ACTION_SPEC,
                 reward_spec=REWARD_SPEC,
                 obs_spec=OBS_SPEC,
                 terminal_spec=TERMINAL_SPEC,
                 model_spec=MODEL_SPEC,
                 optimizer=tf.train.AdamOptimizer,
                 terminal_prior=None,
                 state_next_randomness=0.0,
                 state_next_action_removal=False,
                 obs_equal_state=False):

        self.state_spec = state_spec
        self.action_spec = action_spec
        self.reward_spec = reward_spec
        self.obs_spec = obs_spec
        self.terminal_spec = terminal_spec
        self.model_spec = model_spec
        self.optimizer = optimizer

        self.terminal_prior = terminal_prior
        self.state_next_randomness = state_next_randomness
        self.state_next_action_removal = state_next_action_removal
        self.obs_equal_state = obs_equal_state

        super(Transition, self).__init__()

    # setup graph #
    def _setup_placeholder(self):
        with tf.name_scope("placeholder"):
            self.state = tf.placeholder(dtype=self.state_spec["dtype"],
                                        shape=self.state_spec["shape"],
                                        name="state")
            self.action = tf.placeholder(dtype=self.action_spec["dtype"],
                                         shape=self.action_spec["shape"],
                                         name="action")
            self.reward = tf.placeholder(dtype=np.float32,
                                         shape=self.reward_spec["shape"],
                                         name="reward")
            self.state_next_target = tf.placeholder(dtype=self.state_spec["dtype"],
                                                    shape=self.state_spec["shape"],
                                                    name="state_next_target")           # not useful in test mode
            self.obs_target = tf.placeholder(dtype=self.obs_spec["dtype"],
                                             shape=self.obs_spec["shape"],
                                             name="obs_target")                         # not useful in test mode
            self.terminal_target = tf.placeholder(dtype=self.terminal_spec["dtype"],
                                                  shape=self.terminal_spec["shape"],
                                                  name="terminal_target")               # not useful in test mode

    def _setup_net(self):
        # dependent model detail #
        # --- for state_next prediction --- #
        nl = self.model_spec["nonlinear"]
        with tf.name_scope("state_next_scope"):
            layer_actions = [self.action]
            action_layers = self.model_spec["action_layers"]
            for i in range(len(action_layers)):
                layer_actions.append(
                    tf.layers.dense(inputs=layer_actions[i], units=action_layers[i], name="action_%02d" % i,
                                    activation=nl)
                )

            layer_states = [self.state]
            state_layers = self.model_spec["state_layers"]
            for i in range(len(state_layers)):
                layer_states.append(
                    tf.layers.dense(inputs=layer_states[i], units=state_layers[i], name="state_%02d" % i,
                                    activation=nl)
                )

            layer_rewards = [self.reward]
            reward_layers = self.model_spec["reward_layers"]
            for i in range(len(reward_layers)):
                layer_rewards.append(
                    tf.layers.dense(inputs=layer_rewards[i], units=reward_layers[i], name="reward_%02d" % i,
                                    activation=nl)
                )

            layer_combines = [tf.concat([layer_actions[-1], layer_states[-1], layer_rewards[-1]], axis=-1)]
            combine_layers = self.model_spec["combine_layers"]
            for i in range(len(combine_layers)):
                layer_combines.append(
                    tf.layers.dense(inputs=layer_combines[i], units=combine_layers[i], name="combine_%02d" % i,
                                    activation=nl)
                )

        # current method, deterministic prediction
        self.logits_state_next = tf.layers.dense(inputs=layer_combines[-1], units=self.state_spec["shape"][-1],
                                                 name="logits_state_next")
        self.state_next = tf.identity(input=self.logits_state_next, name="state_next")

        # --- for terminal prediction --- #
        self._setup_net_terminal()

        # --- for observation prediction --- #
        self._setup_net_obs()

    def _setup_net_terminal(self):
        nl = self.model_spec["nonlinear"]
        with tf.name_scope("terminal_scope"):
            layer_states = [self.state]
            state_layers = self.model_spec["state_layers"]
            for i in range(len(state_layers)):
                layer_states.append(
                    tf.layers.dense(inputs=layer_states[i], units=state_layers[i], name="terminal_state_%02d" % i,
                                    activation=nl)
                )
        self.logits_terminal = tf.layers.dense(inputs=layer_states[-1], units=self.terminal_spec["shape"][-1],
                                               name="logits_terminal")
        self._terminal = tf.nn.softmax(self.logits_terminal, axis=-1, name="terminal")

    def _setup_net_obs(self):
        nl = self.model_spec["nonlinear"]
        # if self.obs_equal_state:
        #     self.logits_obs = tf.identity(input=self.state, name="logits_obs")
        #     self._obs = tf.identity(self.logits_obs, name="obs")
        #     return self._obs
        with tf.name_scope("obs_scope"):
            layer_states = [self.state]
            state_layers = self.model_spec["state_layers"]
            for i in range(len(state_layers)):
                layer_states.append(
                    tf.layers.dense(inputs=layer_states[i], units=state_layers[i], name="obs_state_%02d" % i,
                                    activation=nl)
                )
        # current method, deterministic prediction
        self.logits_obs = tf.layers.dense(inputs=layer_states[-1], units=self.obs_spec["shape"][-1],
                                          name="logits_obs")
        self._obs = tf.identity(input=self.logits_obs, name="obs")

    def _setup_loss(self):
        self.loss_state_next = tf.losses.mean_squared_error(self.state_next_target, self.state_next)
        self.loss_terminal = tf.losses.softmax_cross_entropy(self.terminal_target, self.logits_terminal)
        self.loss_obs = tf.losses.mean_squared_error(self.obs_target, self._obs)
        return self.loss_state_next

    def _setup_optim(self):
        self.optimizer_state_next = self.optimizer().minimize(self.loss_state_next)
        self.optimizer_terminal = self.optimizer().minimize(self.loss_terminal)
        self.optimizer_obs = self.optimizer().minimize(self.loss_obs)
        return self.optimizer_state_next

    def generate(self,
                 state,
                 action,
                 reward=None,
                 randomness=None):
        """
        generate state_next
        """
        if state.ndim < 2:
            state_batch = state[np.newaxis, :]
        else:
            state_batch = state
        if self.state_next_action_removal:
            action = np.zeros_like(action)
        if action.ndim < 2:
            action_batch = action[np.newaxis, :]
        else:
            action_batch = action
        if reward is not None:
            if self.state_next_action_removal:
                warnings.warn("reward input may introduce dependence on action in state_next prediction")
            if reward.ndim < 2:
                tmp = reward[np.newaxis, :]
            else:
                tmp = reward
            reward_batch = tmp.astype(dtype=np.float32)
        else:
            reward_batch = np.zeros(shape=[action_batch.shape[0], self.reward_spec['shape'][-1]])
        feed_dict = {self.state: state_batch, self.action: action_batch, self.reward: reward_batch}
        state_next = self.sess.run(self.state_next, feed_dict=feed_dict)
        state_next = self._add_random(state_next, randomness)
        return state_next

    def _add_random(self, origin_state, randomness):
        if randomness is None:
            randomness = self.state_next_randomness
        if randomness is None or randomness == 0.0:
            return origin_state
        assert 0.0 <= randomness <= 1.0
        noise = np.random.normal(size=origin_state.shape)
        noise = noise / scipy.linalg.norm(noise, axis=-1) * scipy.linalg.norm(origin_state, axis=-1)
        new_state = (1.0 - randomness) * origin_state + randomness * noise
        return new_state

    def terminal(self, state, prior=None):
        """
        check terminal
        :param state: the next state of the environment
        :return: terminal bool_
        """
        if state.ndim < 2:
            state_batch = state[np.newaxis, :]
        else:
            state_batch = state
        feed_dict = {self.state: state_batch}
        terminal_dist = self.sess.run(self._terminal, feed_dict=feed_dict)
        terminal_dist = np.squeeze(terminal_dist)

        terminal_dist = self._add_prior(terminal_dist, prior)

        terminal = np.random.multinomial(1, terminal_dist)
        return terminal, terminal_dist

    def obs(self, state):
        """
        calculate observation from state
        :param state: the next state of the environment
        :return: obs
        """
        if self.obs_equal_state:
            return state

        if state.ndim < 2:
            state_batch = state[np.newaxis, :]
        else:
            state_batch = state
        feed_dict = {self.state: state_batch}
        obs = self.sess.run(self._obs, feed_dict=feed_dict)
        return obs

    def _add_prior(self, origin_dist, prior):
        if prior is None:
            prior = self.terminal_prior
        if prior is None:
            prior = np.ones_like(origin_dist)
        origin_dist = np.multiply(origin_dist, prior)
        origin_dist = origin_dist / np.sum(origin_dist, axis=-1)
        return origin_dist


if __name__ == "__main__":
    state = np.random.random(size=STATE_SPEC['shape'][-1]).astype(STATE_SPEC['dtype'])
    transition = Transition(terminal_prior=np.array([0.99, 0.01]),
                            state_next_randomness=0.0,
                            state_next_action_removal=True,
                            obs_equal_state=True)
    transition.initialize()
    for i in range(10):
        action = np.random.random(size=ACTION_SPEC['shape'][-1]).astype(ACTION_SPEC['dtype'])
        state_next = transition.generate(state=state, action=action)
        obs_next = transition.obs(state_next)
        terminal = transition.terminal(state_next)
        # state = state_next
        print "------ %i step ------" % i
        print state_next
        print obs_next
        print terminal
