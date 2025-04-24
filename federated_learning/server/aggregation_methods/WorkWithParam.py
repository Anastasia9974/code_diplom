import flwr as fl
from new_code.federated_learning.NN.model import CNN
def getWeightsByParam(parameters, input_shape):
    ndarr = fl.common.parameters_to_ndarrays(parameters)
    temp_net = CNN(input_shape=input_shape)
    temp_net.model.set_weights(ndarr)
    return temp_net.model.get_weights()