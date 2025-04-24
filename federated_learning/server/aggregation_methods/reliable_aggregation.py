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
    def __init__(self, tau, filter_func, part_math_wait, input_shape, n_iter=1):
        self.tau = tau
        self.n_iter = n_iter
        self.filter_func = filter_func
        self.part_math_wait = part_math_wait,
        self.input_shape = input_shape
        self.momentum = None
        super().__init__()

    def aggregate_fit(self,
                      server_round: int,
                      results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
                      failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]], ) \
            -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        provdict = {}
        provdict["client2ws"] = {client.cid: getWeightsByParam(fit_res.parameters, input_shape=self.input_shape) for
                                 client, fit_res in results}
        for client, fit_res in results:
            print(f"\n **** Custom Metrics from clients: client : {client.cid}, metrics: {fit_res.metrics}")
        malacious_clients2conf, acc = self.run_filter(client2model_ws=provdict["client2ws"])

        weights_list = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        n_clients = len(weights_list)

        # 1. Начальная инициализация центра
        center = [sum(layer) / n_clients for layer in zip(*weights_list)]

        # 2. Итеративное обновление центра
        for _ in range(self.n_iter):
            clipped_deltas_list = []
            for weights in weights_list:
                # Δ_i = W_i - center
                deltas = [w - c for w, c in zip(weights, center)]
                flat_delta = np.concatenate([d.flatten() for d in deltas])
                norm = np.linalg.norm(flat_delta)

                # Обрезка
                scale = min(1.0, self.tau / (norm + 1e-6))
                clipped_deltas = [scale * d for d in deltas]
                clipped_deltas_list.append(clipped_deltas)

            # 3. Усреднение обрезанных дельт
            avg_deltas = [sum(layer_deltas) / n_clients for layer_deltas in zip(*clipped_deltas_list)]

            # 4. Обновление центра
            center = [c + delta for c, delta in zip(center, avg_deltas)]

        # После итераций, финальный агрегированный результат — это центр
        agg_weights = center
        aggregated_metrics = {"example_metric": 1.0}
        resualt_work.resualt_for_param_agg1[f"tau_{self.tau}"][f"rounds_{server_round}"]=[[arr.tolist() for arr in agg_weights], 0]
        resualt_work.resualt_for_param_agg1[f"i_iter_{self.tau}"][f"rounds_{server_round}"]=[[arr.tolist() for arr in agg_weights], 0]
        return ndarrays_to_parameters(agg_weights), aggregated_metrics

    def run_filter(self, client2model_ws):
        client2model = {cid: CNN(input_shape=self.input_shape).model for cid in client2model_ws.keys()}
        for cid, model in client2model.items():
            model.set_weights(client2model_ws[cid])
        potential_malclient2_confidence, feddfuzz_acc = self.filter_func.run_filter(result_clients = client2model, nc_t=0.0, part_math_wait=self.part_math_wait)
        return potential_malclient2_confidence, feddfuzz_acc

    def update_results_client(self, malacious_clients2confidence :dict[str:int], results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]]):# тут как раз вклад оценивается, надо подкоректировать завтра будет
        # fuzz_inputs = 10
        print(f"malacious_clients2confidence:{malacious_clients2confidence}")
        min_nk = min([r[1].num_examples for r in results])
        for i in range(len(results)):
            cid =  results[i][0].cid
            if cid in malacious_clients2confidence:
                before = results[i][1].num_examples
                #то есть если доверие меньше 70 процентов, то он обнуляется
                if malacious_clients2confidence[cid] <= 0.6:
                    results[i][1].num_examples = 0
                else:
                    results[i][1].num_examples = int(min_nk * (malacious_clients2confidence[cid]))
                print(f">> Server Defense Result: Client {cid} is malicious, confidence {malacious_clients2confidence[cid]}, num_examples before: {before}, after: {results[i][1].num_examples}")
