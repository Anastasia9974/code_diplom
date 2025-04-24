from sklearn.metrics import roc_curve, auc
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
import json
from flwr.common import ndarrays_to_parameters
import numpy as np
from new_code.federated_learning.NN.model import CNN
from new_code.work_with_datasets.get_dataset import WorkWithDataset
from new_code.federated_learning.NN.test import testing_model
class EvaluateParam:
    def __init__(self, name_file_resault_json):
        self.name_file_resault_json = name_file_resault_json

    def convert_json_to_parameters(self, data_from_json):
        # Преобразуем списки обратно в numpy.ndarray
        ndarrays = [np.array(arr) for arr in data_from_json]
        # Преобразуем ndarrays обратно в Parameters
        return ndarrays_to_parameters(ndarrays)
    
    def data_transformation(self):
        with open(self.name_file_resault_json, "r") as json_file:
            data_with_conf = json.load(json_file)
        for section, var in data_with_conf.items():
            if "tau" in section:
                for raunds in var:
                    weights = self.convert_json_to_parameters(var[raunds][0])
                    loss = var[raunds][0]

                    ...
                ...
            elif "i_iter" in section:
                ...