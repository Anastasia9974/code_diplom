from contextvars import Context
from typing import List
import numpy as np
import flwr
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from tensorflow.python.ops.metrics_impl import accuracy
from new_code.federated_learning.NN.model import CNN
from new_code.federated_learning.NN.train import training_model
from new_code.federated_learning.NN.test import testing_model
from  new_code.work_with_datasets.change_db import change_db
from new_code.attacks.backdoor_atack import backdoor
class FederatedClient(flwr.client.Client):
    def __init__(self, client_id, model, train_data, test_data, epochs):
        self.client_id = client_id
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.epochs = epochs
        #print(f"Client ID: {self.client_id}, type train: {type(train_data)}, type test: {type(test_data)}")
    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        print(f"[Client {self.client_id}] get_parameters")

        # Get parameters as a list of NumPy ndarray's
        ndarrays = self.model.model.get_weights()
        # Serialize ndarray's into a Parameters object
        parameters = ndarrays_to_parameters(ndarrays)
        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return GetParametersRes(
            status=status,
            parameters=parameters,
        )
    def fit(self, ins: FitIns) -> FitRes:
        print(f"[Client {self.client_id}] fit, config: {ins.config}")

        parameters_original = ins.parameters
        ndarrays_parameters = parameters_to_ndarrays(parameters_original)
        # Обновление локальной модели, тренировка, получение обновленные параметры
        self.model.model.set_weights(ndarrays_parameters)

        self.model = training_model(model= self.model, epochs=self.epochs, train_ds=self.train_data, test_ds=self.test_data)
        parameters_updated = ndarrays_to_parameters(self.model.model.get_weights())
        loss, accuracy_model = testing_model(model=self.model, test_ds=self.test_data)
        # Создайте и верните ответ
        status = Status(code=Code.OK, message="Success")
        return FitRes(
            status=status,
            parameters=parameters_updated,
            num_examples=sum(1 for _ in self.train_data),
            metrics={"loss": float(loss)},
        )
    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        print(f"[Client {self.client_id}] evaluate, config: {ins.config}")
        parameters_original = ins.parameters
        ndarrays_parameters = parameters_to_ndarrays(parameters_original)
        self.model.model.set_weights(ndarrays_parameters)
        loss, accuracy_model = testing_model(model= self.model, test_ds=self.test_data)
        # Создайте и верните ответ
        status = Status(code=Code.OK, message="Success")
        return EvaluateRes(
            status=status,
            loss=float(loss),
            num_examples=sum(1 for _ in self.test_data),
            metrics={"accuracy": float(accuracy_model)},
        )

def get_client_fn_with_data(epochs:int, batch_size, attacks, bad_clients,  data_for_cl = None):
    def client_fn(cid: str) -> flwr.client.Client:
        print(f"[Client {cid}] client_fn")
        client_id = cid
        # Load model and data
        input_shape = data_for_cl["train_data"][int(client_id)][0][0].shape
        model = CNN(input_shape=input_shape)
        if attacks == "backdoor" and (int(client_id) in bad_clients):
            print(f"bad_clients: {bad_clients}")
            data_for_cl["train_data"][int(client_id)] = backdoor(train_ds=data_for_cl["train_data"][int(client_id)], clients_id=client_id)
            data_for_cl["test_data"][int(client_id)] = backdoor(train_ds=data_for_cl["test_data"][int(client_id)], clients_id=client_id)
        else:
            #тут какая нибудь другая атака должна быть
            ...
        train_data, test_data = change_db(db_cl=data_for_cl, cid=int(client_id), batch_size=batch_size)
        # Return Client instance
        return FederatedClient(model=model, client_id=cid, epochs=epochs, train_data=train_data,
                               test_data=test_data).to_client()
    return client_fn