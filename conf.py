
class Config:
    #Записаны конфигурационные переменные для настройки федиративного обучения
    FL_conf = {"num_client":0, "num_infected_clients":0, "num_round":0, "bad_clients": [], "attacks":""}
    #Записаны конфигурационные переменные для настройки нейронки  (и возможно если реально, то других способов получения модели)
    NN_conf = {"epochs":0}
    #настройка бд, что результаты хранит
    database_conf = {"name_dataset":"", "test_percent":0, "batch_size": 0}
    #настройка безопасности
    security_conf = {"aggregation_strategy": "", "filter": "not", "name_situation":"", "part_math_wait": 0.5}
    #сами данные
    data_for_cl = {"train_data":[], "test_data":[], "all_train_data":[], "all_test_data":[]}
    #настройка результатов, то есть что за данные собираются
    view_resualt = {"mode_work": "not"}

#значения filter: "fedDefender", "new_metod", "not"
#значения aggregation_strategy: "reliable_aggregation", "FedAvg"
#значения mode_work: "change_param_agg_1", "change_param_agg_2", "change_param_filter", "not"
# значения attacks: "backdoor", "label_flipping"
# значения part_math_wait: -1 - математический способ подбора параметра
# пусть раундов будет 8, а клиентов например 12
conf_app = Config()

class Resualt:
    resualt_FL = {}
    resualt_get_data_for_model = {}
    resualt_for_param_agg1 = {}
    resualt_for_param_agg2 = {}

resualt_work = Resualt()