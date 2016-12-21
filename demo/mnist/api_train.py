import py_paddle.swig_paddle as api
import numpy as np

import paddle.trainer_config_helpers.config_parser as config_parser
from paddle.trainer_config_helpers import *


def optimizer_config():
    settings(
        learning_rate=1e-4, learning_method=AdamOptimizer(), batch_size=1000)


def network_config():
    imgs = data_layer(name='pixel', size=784)
    hidden1 = fc_layer(input=imgs, size=200)
    hidden2 = fc_layer(input=hidden1, size=200)
    inference = fc_layer(input=hidden2, size=10, act=SoftmaxActivation())
    cost = classification_cost(
        input=inference, label=data_layer(
            name='label', size=10))
    outputs(cost)


def init_parameter(network):
    assert isinstance(network, api.GradientMachine)
    for each_param in network.getParameters():
        assert isinstance(each_param, api.Parameter)
        array = each_param.getBuf(api.PARAMETER_VALUE).toNumpyArrayInplace()
        assert isinstance(array, np.ndarray)
        for i in xrange(len(array)):
            array[i] = np.random.uniform(-1.0, 1.0)


def main():
    api.initPaddle("-use_gpu=false", "-trainer_count=4")  # use 4 cpu cores

    opt_config_proto = config_parser.parse_optimizer_config(optimizer_config)
    opt_config = api.OptimizationConfig.createFromProto(opt_config_proto)
    _temp_optimizer_ = api.ParameterOptimizer.create(opt_config)
    enable_types = _temp_optimizer_.getParameterTypes()

    model_config = config_parser.parse_network_config(network_config)
    m = api.GradientMachine.createFromConfigProto(
        model_config, api.CREATE_MODE_NORMAL, enable_types)
    assert isinstance(m, api.GradientMachine)
    init_parameter(network=m)

    updater = api.ParameterUpdater.createLocalUpdater(opt_config)
    assert isinstance(updater, api.ParameterUpdater)
    updater.init(m)
    m.start()

    for _ in xrange(100):
        updater.startPass()

    m.finish()


if __name__ == '__main__':
    main()
