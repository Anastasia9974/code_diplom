# Данные будут в формате json, а именно:
# {
#     "name_1+raund": {resualt_round_1, ..., resualt_round_2},
#     .
#     .
#     .
#     "name_n+raund": {resualt_round_1, ..., resualt_round_2}
# }
#resualt_round_1 состоит из масисва, где записаны результаты обучения, результаты функции потерь и результаты тестирования
#
#
from new_code.attacks.backdoor_atack import backdoor

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import json

from flwr.common import ndarrays_to_parameters
import numpy as np

from new_code.federated_learning.NN.model import CNN
from new_code.work_with_datasets.get_dataset import WorkWithDataset
from new_code.federated_learning.NN.test import testing_model
from new_code.conf import conf_app
from new_code.work_with_conf import get_conf
from new_code.work_with_conf import set_default_conf
from new_code.work_with_file_resualts.write_in_file import write_in_file_csv
class EvaluateResult:
    def __init__(self, name_file_result, name_file_conf):
        self.name_file_result = name_file_result
        self.name_file_conf = name_file_conf
        self.model = None
        self.test_data = []
        self.result_acc = {}
        self.result_loss = {}
        self.result_conf = {}
        self.result_security = {}

    def start(self):
        with open(self.name_file_conf) as json_file_conf:
            data_with_conf = json.load(json_file_conf)
        with open(self.name_file_result, "r") as json_file_result:
            data_results = json.load(json_file_result)
        for (section_conf, var_conf), (section_result, var_result) in zip(data_with_conf.items(), data_results.items()):
            self.get_data_conf(section_conf, var_conf)
            self.get_data_test_and_set_model()
            if section_result == section_conf:
                self.data_transformation(var_result, section_result)
            set_default_conf()
        self.evaluate_accuracy_for_round()
        #self.evaluate_loss()
        self.evaluate_accuracy_for_all_round()
        self.evaluate_security()
    def get_data_conf(self, section,var):
        print(f"situation: {section}")
        conf_app.security_conf["name_situation"] = section
        get_conf(data_with_conf=var)

    def get_data_test_and_set_model(self):
        db = WorkWithDataset()
        db.download_dataset(data_name=conf_app.database_conf["name_dataset"])
        db.data_division(num_client=conf_app.FL_conf["num_client"], test_percent=conf_app.database_conf["test_percent"])
        (qwe0, qwe1, qwe, self.test_data) = db.get_dataset()
        #self.test_data = backdoor(train_ds=self.test_data, clients_id=0)
        db = WorkWithDataset()
        db.set_dataset(qwe0, qwe1, qwe, self.test_data)
        self.model = CNN(input_shape=self.test_data[0][0].shape)
        db.data_processing(batch_size=conf_app.database_conf["batch_size"], input_shape=self.test_data[0][0].shape)
        (_, _, _, self.test_data) = db.get_dataset()



    def convert_json_to_parameters(self, data_from_json):
        # Преобразуем списки обратно в numpy.ndarray
        ndarrays = [np.array(arr) for arr in data_from_json]
        # Преобразуем ndarrays обратно в Parameters
        return ndarrays
    def data_transformation(self, var_result, section_result):
        ##еще добавить выгрузку 3 параметра
        raund_acc = []
        raund_loss = []
        raund_security = []
        for raunds in var_result:
            weights = self.convert_json_to_parameters(var_result[raunds][0])
            loss = var_result[raunds][1]
            self.model.model.set_weights(weights)
            _, accuracy = testing_model(model=self.model, test_ds=self.test_data)
            raund_acc.append(accuracy)
            raund_loss.append(loss)
            raund_security.append(var_result[raunds][2])
        self.result_acc[section_result] = raund_acc
        self.result_loss[section_result] = raund_loss
        self.result_conf[section_result] = conf_app
        self.result_security[section_result] = raund_security
        

    def evaluate_accuracy_for_round(self):
        print("evaluate accuracy for round")
        write_in_file_csv("/home/anvi/code_diplom/new_code/results/1_3_attacks.csv", self.result_acc)
        #self.show_results(data=self.result_acc, flag_made="acc")
        ...
    
    def evaluate_accuracy_for_all_round(self):
        print("evaluate accuracy for all round НЕТ")
        ...

    def evaluate_security(self):
        print("evaluate security НЕТ")
        ...
    def evaluate_loss(self):
        print("evaluate loss")
        #self.show_results(data=self.result_loss, flag_made="loss")
        
    def show_results(self, data, flag_made):
        fig, ax = plt.subplots()
        all_result = []
        for section in data:
            one_result = {}
            heading = str(section)
            x = np.arange(1, len(data[section]) + 1)
            if flag_made == "acc":
                y_acc = [acc * 100 for acc in data[section]]
                one_result["y"] = y_acc
            elif flag_made == "loss":
                y_loss = data[section]
                one_result["y"] = y_loss
            one_result["heading"] = heading
            one_result["x"] = x
            all_result.append(one_result)
        for section in all_result:
            ax.plot(section["x"], section["y"], '-s', label=section["heading"])
        if flag_made == "acc":
            ax.set_ylim(0, 105)
        elif flag_made == "loss":
            ax.set_ylim(0, 4)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.set_title(f"Тестирование точности, при {12} клиентах")
        ax.set_xlabel('Раунд')
        if flag_made == "acc":
            ax.set_ylabel("Точность (%)")
        elif flag_made == "loss":
            ax.set_ylabel("Значение функции потерь")
        ax.legend()
        plt.show()


#name_file_conf = "/home/anvi/code_diplom/new_code/conf.json"
#name_file_result = "/home/anvi/code_diplom/new_code/results/resualt_all.json"

evaluate_security = EvaluateResult(name_file_conf = "/home/anvi/code_diplom/new_code/conf.json", name_file_result = "/home/anvi/code_diplom/new_code/results/resualt_all.json")
evaluate_security.start()