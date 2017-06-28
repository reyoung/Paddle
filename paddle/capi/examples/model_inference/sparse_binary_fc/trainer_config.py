from paddle.trainer_config_helpers import *
input_dim = 500001
num_classes = 2
input = data_layer(name='input_fea', size=input_dim)
hidden = fc_layer(input=input, size=8, act=ReluActivation())
hidden = fc_layer(input=hidden, size=32, act=ReluActivation())
hidden = fc_layer(input=hidden, size=16, act=ReluActivation())
prediction = fc_layer(input=hidden, size=num_classes, act=SoftmaxActivation())
outputs(prediction)
