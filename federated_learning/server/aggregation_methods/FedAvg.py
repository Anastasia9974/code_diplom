import flwr as fl
import numpy as np
from flwr.common import (FitRes, Parameters, Scalar, parameters_to_ndarrays)
from typing import Dict, List, Optional, Tuple, Union
from flwr.server.client_proxy import ClientProxy
from new_code.federated_learning.server.aggregation_methods.WorkWithParam import getWeightsByParam
from new_code.federated_learning.NN.model import CNN
from new_code.conf import resualt_work, conf_app
from new_code.work_with_file_resualts.write_in_file import write_in_file_csv
#добавить возможно сохранение верультатов обучения, но это дальше
class FedAvgCustomStrategy(fl.server.strategy.FedAvg):
    def __init__(self, filter_func, part_math_wait, input_shape):
        self.filter_func = filter_func
        self.part_math_wait = part_math_wait
        self.input_shape = input_shape
        self.server_round = 0
        super().__init__()
    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        #Вызовите aggregate_fit из базового класса (FedAvg) для агрегирования параметров и метрик
        self.server_round = server_round
        provdict = {}
        provdict["client2ws"] = {}
        for client, fit_res in results:
            print(f"cid_client: {client.partition_id}")
            provdict["client2ws"][client.partition_id] = getWeightsByParam(fit_res.parameters, input_shape=self.input_shape)
        for client, fit_res in results:
            print(f"\n **** Custom Metrics from clients: client : {client.partition_id}, metrics: {fit_res.metrics}" )
        malacious_clients2conf, acc = self.run_filter(client2model_ws=provdict["client2ws"])
        self.update_results_client(malacious_clients2confidence=malacious_clients2conf, results=results)
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        # Агрегация потерь (loss)
        loss_sum = 0.0
        total_examples = 0

        for client, fit_res in results:
            client_loss = fit_res.metrics.get("loss", None)
            num_examples = fit_res.num_examples

            if client_loss is not None:
                loss_sum += client_loss * num_examples
                total_examples += num_examples

        # Среднее значение loss (если хотя бы один клиент его передал)
        if total_examples > 0:
            avg_loss = loss_sum / total_examples
        else:
            avg_loss = None

        if avg_loss is not None:
            print(f"Approximated global loss (client-aggregated): {avg_loss:.4f}")
            aggregated_metrics["approx_global_loss"] = avg_loss

        if aggregated_parameters is not None:
            provdict["gm_ws"] = getWeightsByParam(aggregated_parameters, input_shape=self.input_shape)
        for heading in resualt_work.resualt_FL:
            resualt_work.resualt_FL[heading][f"rounds_{server_round}"] = [[arr.tolist() for arr in parameters_to_ndarrays(aggregated_parameters)], 0, malacious_clients2conf]
        return aggregated_parameters, aggregated_metrics

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
            cid =  results[i][0].partition_id
            if cid in malacious_clients2confidence:
                before = results[i][1].num_examples
                #то есть если доверие меньше 70 процентов, то он обнуляется
                if malacious_clients2confidence[cid] < 0.6:
                    if not(cid in conf_app.FL_conf["bad_clients"]):
                        resualt_work.resualt_get_data_for_model[f"part_math_wait_{self.part_math_wait}"][f"round:{self.server_round}"]["result"] = False
                    results[i][1].num_examples = 0
                else:
                    results[i][1].num_examples = int(min_nk * (malacious_clients2confidence[cid]))
                print(f">> Server Defense Result: Client {cid} is malicious, confidence {malacious_clients2confidence[cid]}, num_examples before: {before}, after: {results[i][1].num_examples}")

    def setCacheAndExpKey(self):
        ...



