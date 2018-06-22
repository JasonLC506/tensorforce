# base class of Neural Network #
import tensorflow as tf


class NN(object):
    def __init__(self):

        self.model = self._setup_graph()                                           # reward model details
        self.sess = self._setup_session()

    def _setup_graph(self):
        # dependent state, action and reward specification #
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.device("/cpu:0"):
                self._setup_placeholder()
                self._setup_net()
                self._setup_loss()
                self._setup_optim()
                self.init = tf.global_variables_initializer()
        self.graph.finalize()
        return self.graph

    def _setup_placeholder(self):
        raise NotImplementedError

    def _setup_net(self):
        raise NotImplementedError

    def _setup_loss(self):
        raise NotImplementedError

    def _setup_optim(self):
        raise NotImplementedError

    def _setup_session(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)
        return self.sess

    def initialize(self, model_import=None):
        """
        initialize model
        :param model_import: a global model to use __class__ == Reward
        """
        if model_import is not None:
            raise NotImplementedError
        else:
            self.sess.run(self.init)


if __name__ == "__main__":
    reward = NN()