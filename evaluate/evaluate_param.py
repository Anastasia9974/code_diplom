from sklearn.metrics import roc_curve, auc
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import json
from flwr.common import ndarrays_to_parameters
import numpy as np
from tensorflow.python.keras.optimizer_v2.utils import all_reduce_sum_gradients

from new_code.federated_learning.NN.model import CNN
from new_code.work_with_datasets.get_dataset import WorkWithDataset
from new_code.federated_learning.NN.test import testing_model
from new_code.work_with_file_resualts.write_in_file import write_in_file_csv
class EvaluateParam:
    def __init__(self, name_file_resault_json, name_csv_file):
        self.name_file_resault_json = name_file_resault_json
        self.name_csv_file = name_csv_file
        self.model = None
        self.test_data = []
        self.data_name = ""
        self.result_acc = {}
        self.result_loss = {}

    def convert_json_to_ndarrays(self, data_from_json):
        # Преобразуем списки обратно в numpy.ndarray
        ndarrays = [np.array(arr) for arr in data_from_json]
        # Преобразуем в ndarrays
        return ndarrays
    
    def data_transformation(self, section, var):#передать данные сюда и переписать немного саму функцию
        raund_acc = []
        raund_loss = []
        for raunds in var:
            if raunds == "data_name":
                continue
            weights = self.convert_json_to_ndarrays(var[raunds][0])
            loss = var[raunds][1]
            self.model.model.set_weights(weights)
            _, accuracy = testing_model(model=self.model, test_ds=self.test_data)
            raund_acc.append(accuracy)
            raund_loss.append(loss)
        self.result_acc[section] = raund_acc
        self.result_loss[section] = raund_loss
        # with open(self.name_file_resault_json, "r") as json_file:
        #     data_results_param = json.load(json_file)
        # for section, var in data_results_param.items():

    def get_data_test_and_set_model(self):
        db = WorkWithDataset()
        db.download_dataset(data_name=self.data_name)
        db.data_division(num_client=12, test_percent=0.1)
        (_, _, _, self.test_data) = db.get_dataset()
        self.model = CNN(input_shape=self.test_data[0][0].shape)
        db.data_processing(batch_size = 128, input_shape=self.test_data[0][0].shape)
        (_, _, _, self.test_data) = db.get_dataset()

    def start(self):
        with open(self.name_file_resault_json, "r") as json_file:
            data_with_conf = json.load(json_file)
        for section, var in data_with_conf.items():
            for section2, var2 in var.items():
                if section2 == "data_name":
                    self.data_name = var2
                    self.get_data_test_and_set_model()
                    break
            self.data_transformation(section, var)
        #self.show_result()
        self.print_result()


    def show_result(self):
        fig, ax = plt.subplots()
        fig2, ax2 = plt.subplots()
        all_result_acc = []
        for section in self.result_acc:
            one_result = {}
            heading = str(section)
            x = np.arange(1, len(self.result_acc[section])+1)
            y_acc = [acc * 100 for acc in self.result_acc[section]]
            one_result["heading"] = heading
            one_result["x"] = x
            one_result["y"] = y_acc
            all_result_acc.append(one_result)
        all_result_loss = []
        for section in self.result_loss:
            one_result = {}
            heading = str(section)
            x = np.arange(1, len(self.result_loss[section])+1)
            y_loss = self.result_loss[section]
            one_result["heading"] = heading
            one_result["x"] = x
            one_result["y"] = y_loss
            all_result_loss.append(one_result)

        for section in all_result_acc:
            ax.plot(section["x"], section["y"], '-s', label=section["heading"])
        ax.set_ylim(0, 105)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.set_title(f"Тестирование точности, при {12} клиентах")
        ax.set_xlabel('Раунд')
        ax.set_ylabel("Точность (%)")
        ax.legend()
        
        for section in all_result_loss:
            ax2.plot(section["x"], section["y"], '-s', label=section["heading"])
        ax2.set_ylim(0, 4)
        ax2.xaxis.set_minor_locator(AutoMinorLocator())
        ax2.yaxis.set_minor_locator(AutoMinorLocator())
        ax2.yaxis.set_ticks_position('both')
        ax2.xaxis.set_ticks_position('both')
        ax2.set_title(f"Тестирование функции потерь, при {12} клиентах")
        ax2.set_xlabel('Раунд')
        ax2.set_ylabel("Значение функции потерь")
        ax2.legend()
        plt.show()
    def print_result(self):
        for section_acc, section_loss in zip(self.result_acc, self.result_loss):
            print_data = {}
            if "tau" in section_acc:
                print_data["tau"] = str(section_acc).replace("tau_", "")
            if "i_iter" in section_acc:
                print_data["i_iter"] = int(str(section_acc).replace("i_iter_", ""))
            for var_acc, i, var_loss in zip(self.result_acc[section_acc], range(len(self.result_acc[section_acc])), self.result_loss[section_loss]):
                print_data[f"round_{i+1}_acc"] = var_acc
                print_data[f"round_{i+1}_loss"] = var_loss
            write_in_file_csv(name_csv_file=self.name_csv_file, data=print_data)


evaluate_param_tau = EvaluateParam(name_file_resault_json="/home/anvi/code_diplom/new_code/results/resualt_for_param_i_iter_attacks_cifar_1_10_4round.json", name_csv_file="/home/anvi/code_diplom/new_code/results/resualt_for_param_tau_cifar_1_10_8round.csv")
evaluate_param_tau.start()
# evaluate_param_i_iter = EvaluateParam(name_file_resault_json="/home/anvi/code_diplom/new_code/results/resualt_for_param_i_iter.json", name_csv_file="/home/anvi/code_diplom/new_code/results/data_param_i_iter.csv")
# evaluate_param_i_iter.start()