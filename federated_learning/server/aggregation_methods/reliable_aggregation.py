import torch
from typing import Dict, List, Optional, Tuple, Union
from flwr.server.client_proxy import ClientProxy
import flwr as fl
from new_code.federated_learning.server.aggregation_methods.WorkWithParam import getWeightsByParam
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

import numpy as np
from new_code.federated_learning.NN.model import CNN
from new_code.conf import resualt_work, conf_app

# class Clipping():
#     def __init__(self, tau, n_iter=1):
#         self.tau = tau
#         self.n_iter = n_iter
#         super(Clipping, self).__init__()
#         self.momentum = None
#
#     def clip(self, v):
#         v_norm = torch.norm(v)
#         scale = min(1, self.tau / v_norm)
#         return v * scale
#
#     def __call__(self, inputs):
#         if self.momentum is None:
#             self.momentum = torch.zeros_like(inputs[0])
#
#         for _ in range(self.n_iter):
#             self.momentum = (
#                 sum(self.clip(v - self.momentum) for v in inputs) / len(inputs)
#                 + self.momentum
#             )
#
#         # print(self.momentum[:5])
#         # raise NotImplementedError
#         return torch.clone(self.momentum).detach()
#
#     def __str__(self):
#         return "Clipping (tau={}, n_iter={})".format(self.tau, self.n_iter)


#делаем наследником fl.server.strategy.FedAvg, чтобы не переопределять кучу классов
class ReliableAggregation(fl.server.strategy.FedAvg):
    def __init__(self, tau, filter_func, part_math_wait, input_shape, initial_parameters, i_iter=1):
        self.tau = tau
        self.i_iter = i_iter
        self.filter_func = filter_func
        self.part_math_wait = part_math_wait
        self.input_shape = input_shape
        self.momentum = None
        self.server_round = 0
        super().__init__(initial_parameters=initial_parameters)

    def aggregate_fit(self,
                      server_round: int,
                      results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
                      failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]], ) \
            -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        #server_round = server_round + 4
        self.server_round = server_round
        provdict = {}
        provdict["client2ws"] = {client.partition_id: getWeightsByParam(fit_res.parameters, input_shape=self.input_shape) for
                                 client, fit_res in results}
        for client, fit_res in results:
            print(f"**** Custom Metrics from clients: client : {client.partition_id}, metrics: {fit_res.metrics}")


        malacious_clients2conf, _ = self.run_filter(client2model_ws=provdict["client2ws"], results_cl=results)
        self.update_results_client(malacious_clients2confidence=malacious_clients2conf, results=results)

        weights_list = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        client_weights = np.array([fit_res.num_examples for _, fit_res in results], dtype=np.float32)
        client_weights_sum = np.sum(client_weights)
        client_weights = client_weights / (client_weights_sum + 1e-6)

        # 1. Начальная инициализация центра
        center = [sum(w * client_weights[i] for i, w in enumerate(layer)) for layer in zip(*weights_list)]

        # 2. Итеративное обновление центра
        for _ in range(self.i_iter):
            clipped_deltas_list = []
            for weights in weights_list:
                # Δ_i = W_i - center
                deltas = [w - c for w, c in zip(weights, center)]
                flat_delta = np.concatenate([d.flatten() for d in deltas])
                norm = np.linalg.norm(flat_delta)

                # Обрезка
                scale = min(1.0, self.tau / (norm + 1e-6))
                print(f"scale: {scale}, norm: {norm}, tau: {self.tau}")
                clipped_deltas = [scale * d for d in deltas]
                clipped_deltas_list.append(clipped_deltas)

            # 3. Усреднение обрезанных дельт
            avg_deltas = [sum(d * client_weights[i] for i, d in enumerate(layer_deltas)) for layer_deltas in zip(*clipped_deltas_list)]

            # 4. Обновление центра
            center = [c + delta for c, delta in zip(center, avg_deltas)]

        # После итераций, финальный агрегированный результат — это центр
        agg_weights = center
        aggregated_metrics = {"example_metric": 1.0}
        print(f"server_round: {server_round}")
        if conf_app.view_resualt["mode_work"] == "change_param_agg_1":
            resualt_work.resualt_for_param_agg1[f"tau_{self.tau}"][f"rounds_{server_round}"]=[[arr.tolist() for arr in agg_weights], 0]
        if conf_app.view_resualt["mode_work"] == "change_param_agg_2":
            resualt_work.resualt_for_param_agg2[f"i_iter_{self.i_iter}"][f"rounds_{server_round}"]=[[arr.tolist() for arr in agg_weights], 0]
        for heading in resualt_work.resualt_FL:
             resualt_work.resualt_FL[heading][f"rounds_{server_round}"] = [[arr.tolist() for arr in agg_weights], 0, malacious_clients2conf]
        return ndarrays_to_parameters(agg_weights), aggregated_metrics

    def definition_loss(self, results_cl):
        # Агрегация потерь (loss)
        loss_sum = 0.0
        total_examples = 0
        for client, fit_res in results_cl:
            client_loss = fit_res.metrics.get("loss", None)
            num_examples = fit_res.num_examples

            if client_loss is not None:
                loss_sum += client_loss * num_examples
                total_examples += num_examples
        # Среднее значение loss (если хотя бы один клиент его передал)
        if total_examples > 0:
            avg_loss = loss_sum / total_examples
            return avg_loss
        else:
            avg_loss = None
            return avg_loss
    def run_filter(self, client2model_ws, results_cl):
        client2model = {cid: CNN(input_shape=self.input_shape).model for cid in client2model_ws.keys()}
        for cid, model in client2model.items():
            model.set_weights(client2model_ws[cid])
        potential_malclient2_confidence, feddfuzz_acc = self.filter_func.run_filter(result_clients = client2model, nc_t=0.0, part_math_wait=self.part_math_wait, server_round=self.server_round, loss = self.definition_loss(results_cl=results_cl))
        return potential_malclient2_confidence, feddfuzz_acc

    def update_results_client(self, malacious_clients2confidence :dict[str:int], results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]]):# тут как раз вклад оценивается, надо подкоректировать завтра будет
        # fuzz_inputs = 10
        print(f"malacious_clients2confidence:{malacious_clients2confidence}")
        if conf_app.view_resualt["mode_work"] == "change_param_filter":
            resualt_work.resualt_get_data_for_model[f"part_math_wait_{self.part_math_wait}"][f"round:{self.server_round}"][
            "result"] = True
        min_nk = min([r[1].num_examples for r in results])
        for i in range(len(results)):
            cid =  results[i][0].partition_id
            if cid in malacious_clients2confidence:
                before = results[i][1].num_examples
                #то есть если доверие меньше 60 процентов, то он обнуляется
                if malacious_clients2confidence[cid] <= 0.5:
                    if not(cid in conf_app.FL_conf["bad_clients"]) and conf_app.view_resualt["mode_work"] == "change_param_filter":
                        resualt_work.resualt_get_data_for_model[f"part_math_wait_{self.part_math_wait}"][f"round:{self.server_round}"]["result"] = False
                    results[i][1].num_examples = 0
                else:
                    results[i][1].num_examples = int(min_nk * (malacious_clients2confidence[cid]))
                print(f">> Server Defense Result: Client {cid} is malicious, confidence {malacious_clients2confidence[cid]}, num_examples before: {before}, after: {results[i][1].num_examples}")
