import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


class model_for_param:
    def __init__(self, file_name_with_dataser, file_name_for_result_model):
        self.file_name_with_dataser = file_name_with_dataser
        self.file_name_for_result_model = file_name_for_result_model
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def definition_model(self):
        self.model = DecisionTreeRegressor(max_depth=5, random_state=42)
        ...
    def convert_json_csv(self):
        ...
    def set_result_model(self):
        self.definition_model()


    def learning(self):
        self.model.fit(self.X_train, self.y_train)
        ...
    def predict_param(self, diff_neurons, loss_value):
        input_data = scaler.transform([[diff_neurons, loss_value]])
        prediction = model.predict(input_data)
        return prediction[0]
    def start_learning(self):
        self.convert_json_csv()
        self.definition_model()
        self.learning()


# === 1. Загружаем данные ===
data = pd.read_csv('/home/anvi/code_diplom/new_code/results/data_for_model.json')

X = data[['разница_общих_активированных_нейронов', 'функция_потерь']].values
y = data['параметр'].values

# Нормализация данных (можно и не нормализовать для дерева, но если хочешь — ок)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Разбиваем на обучение и тест
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 2. Создаем и обучаем дерево ===
model = DecisionTreeRegressor(max_depth=5, random_state=42)  # max_depth — чтобы не переобучилось сильно
model.fit(X_train, y_train)

# === 3. Оцениваем ===
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse:.4f}")

# === 4. Функция для предсказания ===
def predict_param(diff_neurons, loss_value):
    input_data = scaler.transform([[diff_neurons, loss_value]])
    prediction = model.predict(input_data)
    return prediction[0]

# Пример:
param_prediction = predict_param(diff_neurons=0.15, loss_value=0.03)
print(f"Предсказанный параметр: {param_prediction}")