import flwr
from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from new_code.federated_learning.NN.model import CNN
from new_code.federated_learning.server.aggregation_methods.FedAvg import FedAvgCustomStrategy
from new_code.federated_learning.server.aggregation_methods.reliable_aggregation import ReliableAggregation
from new_code.federated_learning.clients.fl_client import get_client_fn_with_data
from new_code.conf import conf_app, resualt_work
from new_code.Filter.fedDefender import fedDefender
from new_code.Filter.modified_filter import ModifiedNewFilter
from new_code.Filter.not_filter import not_filter
from new_code.work_with_file_resualts.write_in_file import write_in_file_csv, write_in_file_json
import numpy as np
class FederatedServer:
    def __init__(self, name_strategy: str, num_rounds: int, filter: str, num_clients: int, part_math_wait):
        self.name_strategy = name_strategy
        self.num_rounds = num_rounds
        self.filter = filter
        self.input_shape = conf_app.data_for_cl["train_data"][0][0][0].shape
        self.parameters_model = ndarrays_to_parameters(CNN(input_shape= self.input_shape).model.get_weights())
        self.config_server = flwr.server.ServerConfig(num_rounds=self.num_rounds)
        self.num_clients = num_clients
        self.part_math_wait = part_math_wait
        self.tau = 1
        self.i_iter=1
        ...
    def get_strategy(self, filter_func):
        if self.name_strategy == "FedAvg":
            strategy = FedAvgCustomStrategy(filter_func=filter_func, part_math_wait=self.part_math_wait, input_shape=self.input_shape)
        elif self.name_strategy == "reliable_aggregation":
            strategy = ReliableAggregation(tau=self.tau, i_iter=self.i_iter, filter_func=filter_func, part_math_wait=self.part_math_wait, input_shape=self.input_shape)
        else:
            print("Invalid name_strategy")
            raise NotImplementedError
        return strategy
    def get_filter(self):
        sample_image, _ = conf_app.data_for_cl["all_train_data"][0]
        input_shape = tuple(sample_image.shape)
        if self.filter == "fedDefender":
            filter_func = fedDefender(round=self.num_rounds,input_shape=input_shape,dname=conf_app.database_conf["name_dataset"])
        elif self.filter == "new_metod":
            filter_func = ModifiedNewFilter(round=self.num_rounds,input_shape=input_shape,dname=conf_app.database_conf["name_dataset"])
        elif self.filter == "not":
            filter_func = not_filter(round=self.num_rounds,input_shape=input_shape,dname=conf_app.database_conf["name_dataset"])
        else:
            print("Invalid filter")
            raise NotImplementedError
        return filter_func

    def start_train(self):
        print("Starting FL train...")
        num_test = 1
        if conf_app.view_resualt["mode_work"] == "change_param_agg_1":
            num_test = 400
        elif conf_app.view_resualt["mode_work"] == "change_param_agg_2":
            num_test = 10
        elif conf_app.view_resualt["mode_work"] == "change_param_filter":
            num_test = int(0.7/0.01)
        for i in range(num_test):
            resualt_work.resualt_for_param_agg1[f"tau_{self.tau}"] = {"data_name":conf_app.database_conf["name_dataset"]}
            resualt_work.resualt_for_param_agg2[f"i_iter_{self.i_iter}"] = {"data_name":conf_app.database_conf["name_dataset"]}
            resualt_work.resualt_get_data_for_model[f"part_math_wait_{self.part_math_wait}"] = {}
            history = flwr.simulation.start_simulation(
                client_fn=get_client_fn_with_data(epochs=conf_app.NN_conf["epochs"],
                                                  batch_size=conf_app.database_conf["batch_size"],
                                                  data_for_cl=conf_app.data_for_cl, attacks=conf_app.FL_conf["attacks"],
                                                  bad_clients=conf_app.FL_conf["bad_clients"]),
                num_clients=self.num_clients,
                config=self.config_server,
                strategy=self.get_strategy(filter_func=self.get_filter()),
                client_resources={"num_gpus": 1}
            )
            for rnd, loss in enumerate(history.losses_distributed, start=1):
                print(f"Round {rnd}: {loss[1]}")
                if self.name_strategy == "reliable_aggregation":
                    resualt_work.resualt_for_param_agg1[f"tau_{self.tau}"][f"rounds_{rnd}"][1] = loss[1]
                    resualt_work.resualt_for_param_agg2[f"i_iter_{self.i_iter}"][f"rounds_{rnd}"][1] = loss[1]
                resualt_work.resualt_get_data_for_model[f"part_math_wait_{self.part_math_wait}"][f"round:{rnd}"]["loss"] = loss[1]
                for heading in resualt_work.resualt_FL:
                    resualt_work.resualt_FL[heading][f"rounds_{rnd}"][1] = loss[1]
            if conf_app.view_resualt["mode_work"] == "change_param_agg_1":
                self.tau += 1
            elif conf_app.view_resualt["mode_work"] == "change_param_agg_2":
                self.i_iter += 1
            elif conf_app.view_resualt["mode_work"] == "change_param_filter":
                self.part_math_wait += 0.01
        if self.name_strategy == "reliable_aggregation":
            write_in_file_json(name_json_file="/home/anvi/code_diplom/new_code/results/resualt_for_param_tau.json", data = resualt_work.resualt_for_param_agg1)
            write_in_file_json(name_json_file="/home/anvi/code_diplom/new_code/results/resualt_for_param_i_iter.json", data = resualt_work.resualt_for_param_agg2)
        write_in_file_json(name_json_file="/home/anvi/code_diplom/new_code/results/data_for_model.json", data = resualt_work.resualt_get_data_for_model)

