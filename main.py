#Запуск программы, определение что используем (cpu or gpu), чтение файла с конфигурациями запуска
#Инициализация сервера и клиентов федеративного обучения
#Получение датасетов
#Инициализация нейронки
#Запуск эмуляции федеративного обучения
import torch
from new_code.federated_learning.NN.model import CNN
from new_code.federated_learning.NN.train import training_model
from new_code.work_with_datasets.get_dataset import WorkWithDataset
import tensorflow as tf
from new_code.work_with_conf import get_conf
from new_code.work_with_conf import set_default_conf
from new_code.conf import conf_app
from new_code.federated_learning.server.fl_server import FederatedServer
import json
#Как информация у меня выступает в виде конфигурации: имя датасета(name_dataset), количество данных  для тестирования в процентах(test_percent),
#количество обучающих примеров за один проход вперед/назад(batch size), количество эпох для клиентов(epochs),
#число клиентов(num_client), число зараженных клиентов(num_infected_clients), количество раундов(num_round),
#стратегия агрегирования(aggregation_strategy), присутствие фильтра(filter)
#
#
#
#
#
if "__main__" == __name__:
    global conf_app
    print("TensorFlow version:", tf.__version__)
    print("CUDA version:", tf.sysconfig.get_build_info()["cuda_version"])
    print("cuDNN version:", tf.sysconfig.get_build_info()["cudnn_version"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n\n\n")
    name_file_conf = "./new_code/conf.json"
    with open(name_file_conf) as json_file:
        data_with_conf = json.load(json_file)
    for section, var in data_with_conf.items():
        print(section)
        conf_app.security_conf["name_situation"]=section
        get_conf(data_with_conf=var)
        # получение данных
        db = WorkWithDataset()
        db.download_dataset(data_name=conf_app.database_conf["name_dataset"])
        db.data_division(num_client=conf_app.FL_conf["num_client"], test_percent=conf_app.database_conf["test_percent"])
        # db.data_processing(data_name=conf_app.database_conf["name_dataset"], batch_size=conf_app.database_conf["batch_size"])
        (conf_app.data_for_cl["train_data"], conf_app.data_for_cl["test_data"],
         conf_app.data_for_cl["all_train_data"], conf_app.data_for_cl["all_test_data"]) = db.get_dataset()
        # инициализация модели
        # model = CNN()
        # model.show_arhitecture()
        # инициализация федеративного обучения
        part_math_wait = 0.1
        fl_server = FederatedServer(name_strategy=conf_app.security_conf["aggregation_strategy"],
                                    num_rounds=conf_app.FL_conf["num_round"],
                                    num_clients=conf_app.FL_conf["num_client"], filter=conf_app.security_conf["filter"],
                                    part_math_wait=part_math_wait)
        fl_server.start_train()

        # part_math_wait = 0.01
        # while part_math_wait< 0.8:
        #     fl_server = FederatedServer(name_strategy=conf_app.security_conf["aggregation_strategy"],
        #                             num_rounds=conf_app.FL_conf["num_round"],
        #                             num_clients=conf_app.FL_conf["num_client"], filter=conf_app.security_conf["filter"], part_math_wait=part_math_wait)
        #     fl_server.start_train()
        #     part_math_wait = part_math_wait+0.01


        #здесь в теории надо добавить часть кода с тестированием и сохранением результатов
        set_default_conf()



