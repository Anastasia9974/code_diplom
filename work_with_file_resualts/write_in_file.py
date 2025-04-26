import csv
import json
# Так в данный файл пишем следующие:
#     1) пишем сам параметр
#     2) пишем разницу между max и min количеством активированных нейронов
#     4) пишем результаты функции потерь что при обучении глобальной модели получаются
#     6) результат удачный параметр или нет (если найдены атакующие и ложных срабатываний нет то удачный)
def write_in_file_csv(name_csv_file, data):
    with open(name_csv_file, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())
        writer.writeheader()
        writer.writerow(data)


def write_in_file_json(name_json_file, data, made_work = "w"):
    with open(name_json_file, made_work) as f:
        json.dump(data, f, indent=4)