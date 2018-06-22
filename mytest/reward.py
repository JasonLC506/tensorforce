import tensorflow as tf
import numpy as np
import warnings

from baseNet import NN


# default #
STATE_SPEC = dict(dtype=np.float32, shape=[None, 50])
ACTION_SPEC = dict(dtype=np.float32, shape=[None, 25])
REWARD_SPEC = dict(dtype=np.float32, shape=[None, 2])
MODEL_SPEC = dict(action_layers=[],
                  state_layers=[20],
                  combine_layers=[10],
                  nonlinear=tf.nn.relu)


class Reward(NN):
    def __init__(self,
                 state_spec=STATE_SPEC,
                 action_spec=ACTION_SPEC,
                 reward_spec=REWARD_SPEC,
                 model_spec=MODEL_SPEC,
                 optimizer=tf.train.AdamOptimizer,
                 prior=None,
                 randomness=0.0,
                 action_removal=False,
                 state_removal=False
                 ):
        """
        :param state_spec: for basic, dict(dtype=np.float64, shape=[None, ])
        :param action_spec: for basic, dict(dtype=np.float64, shape=[None, ])
        :param reward_spec: for basic, dict(dtype=np.float64, shape=[None, ])
        :param model_spec: dict(
                                action_layers = [units_first_layer, units_second_layer, ...],
                                state_layers = [units_first_layer, units_second_layer, ...],
                                combine_layers = [units_first_layer, units_second_layer, ...]
                               )
        :param prior: prior bias to control prior distribution bias
        :param randomness: random level, (1 - randomness) * original_output + randomness * random_output
        """
        self.state_spec = state_spec
        self.action_spec = action_spec
        self.reward_spec = reward_spec
        self.model_spec = model_spec
        self.optimizer = optimizer

        self.prior = prior
        self.randomness = randomness
        self.action_removal = action_removal
        self.state_removal = state_removal

        super(Reward, self).__init__()

    def _setup_placeholder(self):
        with tf.name_scope("placeholder"):
            self.state = tf.placeholder(dtype=self.state_spec["dtype"],
                                        shape=self.state_spec["shape"],
                                        name="state")
            self.action = tf.placeholder(dtype=self.action_spec["dtype"],
                                         shape=self.action_spec["shape"],
                                         name="action")
            self.reward_target = tf.placeholder(dtype=self.reward_spec["dtype"],
                                                shape=self.reward_spec["shape"],
                                                name="reward")  # not useful in test mode

    def _setup_net(self):
        # dependent model detail #
        nl = self.model_spec["nonlinear"]
        layer_actions = [self.action]
        action_layers = self.model_spec["action_layers"]
        for i in range(len(action_layers)):
            layer_actions.append(
                tf.layers.dense(inputs=layer_actions[i], units=action_layers[i], name="action_%02d" % i, activation=nl))

        layer_states = [self.state]
        state_layers = self.model_spec["state_layers"]
        for i in range(len(state_layers)):
            layer_states.append(
                tf.layers.dense(inputs=layer_states[i], units=state_layers[i], name="state_%02d" % i, activation=nl))

        layer_combines = [tf.concat([layer_actions[-1], layer_states[-1]], axis=-1)]
        combine_layers = self.model_spec["combine_layers"]
        for i in range(len(combine_layers)):
            layer_combines.append(
                tf.layers.dense(inputs=layer_combines[i], units=combine_layers[i], name="combine_%02d" % i, activation=nl))

        # currently consider discrete reward, and probabilistic reward function #
        self.logits = tf.layers.dense(layer_combines[-1], units=self.reward_spec["shape"][-1], name="reward_logits")
        self.reward_dist = tf.nn.softmax(self.logits, axis=1)

        return self.reward_dist

    def _setup_loss(self):
        self.loss = tf.losses.softmax_cross_entropy(self.reward_target, self.logits)
        return self.loss

    def _setup_optim(self):
        self.optimizer = self.optimizer().minimize(self.loss)
        return self.optimizer

    def generate(self, state, action, state_next=None, prior=None, randomness=None):
        """
        generate reward
        current version no state_next input
        """
        if self.state_removal:
            state = np.zeros_like(state)
        if state.ndim < 2:
            state_batch = state[np.newaxis, :]
        else:
            state_batch = state
        if self.action_removal:
            action = np.zeros_like(action)
        if action.ndim < 2:
            action_batch = action[np.newaxis, :]
        else:
            action_batch = action

        if self.state_removal and self.action_removal:
            warnings.warn("reward is independent of state and also action")

        reward_dist = self.sess.run(self.reward_dist, feed_dict={
                                                                    self.state: state_batch,
                                                                    self.action: action_batch,
                                                                }
                                    )
        reward_dist = np.squeeze(reward_dist)
        reward_dist = self._add_random(reward_dist, randomness)
        reward_dist = self._add_prior(reward_dist, prior)
        reward = np.random.multinomial(1, reward_dist)
        return reward, reward_dist

    def _add_random(self, origin_dist, randomness):
        if randomness is None:
            randomness = self.randomness
        if randomness is None or randomness == 0.0:
            return origin_dist
        assert 0.0 <= randomness <= 1.0
        # assert np.sum(origin_dist, axis=-1) == 1.0
        noise = np.random.random(size=origin_dist.shape)
        noise = noise / np.sum(noise, axis=-1)
        new_dist = (1.0 - randomness) * origin_dist + randomness * noise
        return new_dist

    def _add_prior(self, origin_dist, prior):
        if prior is None:
            prior = self.prior
        if prior is None:
            prior = np.ones_like(origin_dist)
        origin_dist = np.multiply(origin_dist, prior)
        origin_dist = origin_dist / np.sum(origin_dist, axis=-1)
        return origin_dist


if __name__ == "__main__":
    state = np.random.random(size=STATE_SPEC['shape'][-1]).astype(STATE_SPEC['dtype'])
    action = np.random.random(size=ACTION_SPEC['shape'][-1]).astype(ACTION_SPEC['dtype'])
    reward = Reward(state_spec=STATE_SPEC,
                    action_spec=ACTION_SPEC,
                    reward_spec=REWARD_SPEC,
                    model_spec=MODEL_SPEC,
                    prior=np.array([0.7, 0.3]),
                    randomness=0.0,
                    action_removal=True,
                    state_removal=False)
    reward.initialize()
    for i in range(10):
        print reward.generate(state=state, action=action)
