from model import model_generator
import tensorflow.python.keras as keras

class A2CAgent(object):
    def __init__(self, **kwargs):
        self.params = {}
        self.params.update(kwargs)

    def _build_net(self):
        assert 'num_action' in self.params.keys() and isinstance(self.params['num_action'], int), \
            'param num_action (type int) needed ... '
        # load base model
        self.base_model = model_generator(model_name=self.params['model_name']) \
            if 'model_name' in self.params.keys() else None
        self.action_net = self.base_model. \
            add(keras.layers.Dense(units=self.params['num_action'])). \
            add(keras.layers.Softmax())
        self.value_net = self.base_model. \
            add(keras.layers.Dense(units=1))

        pass
    def _load_net(self, load_file):
        pass