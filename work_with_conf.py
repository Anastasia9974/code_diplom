from new_code.conf import  conf_app, resualt_work

def get_conf(data_with_conf):
    for key in data_with_conf:
        # print(key)
        if key in conf_app.database_conf:
            conf_app.database_conf[key] = data_with_conf[key]
        elif key in conf_app.security_conf:
            conf_app.security_conf[key] = data_with_conf[key]
        elif key in conf_app.FL_conf:
            conf_app.FL_conf[key] = data_with_conf[key]
        elif key in conf_app.NN_conf:
            conf_app.NN_conf[key] = data_with_conf[key]
        elif key in conf_app.view_resualt:
            conf_app.view_resualt[key] = data_with_conf[key]

    #Записаны конфигурационные переменные для настройки федиративного обучения
    # FL_conf = {"num_client":0, "num_infected_clients":0, "num_round":0, "bad_clients": [], "attacks":""}
    # #Записаны конфигурационные переменные для настройки нейронки  (и возможно если реально, то других способов получения модели)
    # NN_conf = {"epochs":0}
    # #настройка бд, что результаты хранит
    # database_conf = {"name_dataset":"", "test_percent":0, "batch_size": 0}
    # #настройка безопасности
    # security_conf = {"aggregation_strategy": "", "filter": "not", "name_situation":""}
    # #сами данные
    # data_for_cl = {"train_data":[], "test_data":[], "all_train_data":[], "all_test_data":[]}

def set_default_conf():
    conf_app.database_conf["name_dataset"] = ""
    conf_app.database_conf["test_percent"] = 0
    conf_app.database_conf["batch_size"] = 0
    conf_app.security_conf["aggregation_strategy"] = ""
    conf_app.security_conf["filter"] = "not"
    conf_app.security_conf["name_situation"] = ""
    conf_app.data_for_cl["train_data"] = []
    conf_app.data_for_cl["test_data"] = []
    conf_app.data_for_cl["all_train_data"] = []
    conf_app.data_for_cl["all_test_data"] = []
    conf_app.FL_conf["num_client"] = 0
    conf_app.FL_conf["num_infected_clients"] = 0
    conf_app.FL_conf["num_round"] = 0
    conf_app.FL_conf["bad_clients"] = []
    conf_app.FL_conf["attacks"] = ""
    conf_app.NN_conf["epochs"] = 0
    conf_app.view_resualt["mode_work"] = "not"
    conf_app.security_conf["part_math_wait"] =  0.5

def set_default_result():
    resualt_work.resualt_FL = {}
    resualt_work.resualt_for_param_agg1 = {}
    resualt_work.resualt_for_param_agg2 = {}
    resualt_work.resualt_get_data_for_model = {}
#get_conf(name_file_conf="conf.json")

