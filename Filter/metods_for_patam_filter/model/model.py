import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import json
import joblib
import os
class ModelForParam:
    def __init__(self, file_name_with_dataset_json, file_name_for_result_model, file_name_with_dataset_csv):
        self.file_name_with_dataset_json = file_name_with_dataset_json
        self.file_name_for_result_model = file_name_for_result_model
        self.file_name_with_dataset_csv = file_name_with_dataset_csv
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.scaler = StandardScaler()
        self.common_active_neurons_clients = None
        self.result_client = {}
        self.diff_neurons = 0
        self.loss_value = 0
        self.param = 0

    def definition_average(self, common_active_neurons_clients):
        return sum(val for key, val in common_active_neurons_clients) / len(common_active_neurons_clients)

    def definition_model(self):
        if os.path.exists(self.file_name_for_result_model):
            # Модель уже обучена — загружаем
            self.model, self.scaler = joblib.load(self.file_name_for_result_model)
            print("Модель успешно загружена из файла!")
        else:
            # Модель еще не обучена — создаем новую
            self.model = DecisionTreeRegressor(max_depth=5, random_state=42)

    def convert_json_csv(self):
        with open(self.file_name_with_dataset_json, "r") as file_dataset_json:
            json_dataset = json.load(file_dataset_json)
        #здеся преобразование данных
        rows = []

        for param_key, rounds in json_dataset.items():
            # Извлекаем число из строки вида "part_math_wait_0.3"
            param_value = float(param_key.split('_')[-1])
            for round_key, round_data in rounds.items():
                if round_data['result'] is True:
                    row = {
                        'param': param_value,
                        'diff_neurons': round_data['diff_neurons'],
                        'loss': round_data['loss']
                    }
                    rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(self.file_name_with_dataset_csv, index=False)

        data = pd.read_csv(self.file_name_with_dataset_csv)
        X = data[['diff_neurons', 'loss']].values
        y = data['param'].values

        # Нормализация данных (можно и не нормализовать для дерева, но если хочешь — ок)
        X = self.scaler.fit_transform(X)

        # Разбиваем на обучение и тест
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def set_result_model(self):
        self.definition_model()


    def learning(self):
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, self.y_pred)
        print(f"Test MSE: {mse:.4f}")
    def save_result_leaning(self):
        joblib.dump((self.model, self.scaler), self.file_name_for_result_model)
        print("Модель и скейлер успешно сохранены в файл.")

    def predict_param(self):
        input_data = self.scaler.transform([[self.diff_neurons, self.loss_value]])
        prediction = self.model.predict(input_data)
        self.param = prediction[0]
    def start_learning(self):
        self.convert_json_csv()
        self.definition_model()
        self.learning()
        self.save_result_leaning()
    def get_param(self):
        self.definition_model()
        self.predict_param()

    def detection_bad_client(self):
        avg = self.definition_average(common_active_neurons_clients=self.common_active_neurons_clients)
        pogr = avg * self.param
        for cl, x in self.common_active_neurons_clients.items():
            result_client = True
            if x <= avg-pogr:
                result_client = False
            self.result_client[cl] = result_client
    def get_result_client(self, diff_neurons, loss_value, common_active_neurons_clients):
        self.common_active_neurons_clients = common_active_neurons_clients
        self.diff_neurons = diff_neurons
        self.loss_value = loss_value
        self.get_param()
        self.detection_bad_client()
        return self.result_client

#/home/anvi/code_diplom/new_code/results/data_for_model.json
#/home/anvi/code_diplom/new_code/results/data_for_model.csv
#/home/anvi/code_diplom/new_code/results/result_learning_model_param.pkl