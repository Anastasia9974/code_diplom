import time
from torch.nn.init import (kaiming_normal_, kaiming_uniform_, normal_,
                           orthogonal_, trunc_normal_, uniform_,
                           xavier_normal_, xavier_uniform_)

import gc
import itertools
from new_code.work_with_datasets.generation_for_filter import FuzzGeneration
from keras import Model
import keras
import tensorflow as tf



# def get_activations(model, input_tensor, target_class_index=1):
#     _ = model(input_tensor, training=False)
#     print(model.input)
#     # Получаем выходы всех слоев
#     outputs = [layer.output for layer in model.layers if isinstance(layer, layers.Conv2D) or isinstance(layer, layers.Dense)]
#     # Создаем новую модель: вход — такой же, выход — все активации
#     activation_model = Model(inputs=model.input, outputs=outputs)
#
#     # Используем GradientTape для записи операций
#     with tf.GradientTape(persistent=True) as tape:
#         tape.watch(input_tensor)
#
#         # Получаем активации с нужных слоев
#         activations = activation_model(input_tensor, training=False)
#
#         # Получаем предсказание исходной модели
#         preds = model(input_tensor, training=False)
#
#         # Получаем логит соответствующего класса
#         target_logit = preds[:, target_class_index]
#
#     # Список для хранения grad × activation
#     grad_times_act = []
#
#     for activation in activations:
#         # Вычисляем градиент логита по активации
#         grads = tape.gradient(target_logit, activation)
#
#         # Перемножаем поэлементно
#         grad_x_act = grads * activation
#
#         # Сплющиваем
#         grad_x_act_flat = tf.reshape(grad_x_act, [-1])
#         grad_times_act.append(grad_x_act_flat)
#
#     # Конкатенируем все
#     final_vector = tf.concat(grad_times_act, axis=0)
#
#     return final_vector, grad_times_act

def get_activations(model, input_tensor, target_class_index=1):
    # Преобразуем PyTorch тензор в NumPy массив
    torch_tensor_numpy = input_tensor.detach().cpu().numpy()  # Используем .detach() чтобы избежать зависимостей от графа, если это требуется
    # Преобразуем NumPy массив в TensorFlow тензор
    input_tensor = tf.convert_to_tensor(torch_tensor_numpy, dtype=tf.float32)

    _ = model(input_tensor, training=False)
    # Получаем выходы всех слоев
    outputs = [layer.output for layer in model.layers if 'activation' in layer.name or 'conv' in layer.name]
    # Создаем новую модель: вход — такой же, выход — все активации
    activation_model = Model(inputs=model.inputs, outputs=outputs)
    # Получаем активации
    activations = activation_model(input_tensor, training=False)
    # Сплющиваем каждый слой
    flat_activations = [tf.reshape(act, [-1]) for act in activations]

    # Объединяем в один вектор
    flat_vector = tf.concat(flat_activations, axis=0)

    return flat_vector, flat_activations

    # Конкатенируем все grad * activation
    final_vector = tf.concat(grad_times_act, axis=0) if grad_times_act else None

    return final_vector

def getNeuronCoverage(model, img):
    r = get_activations(model, img)

    return r
def makeAllSubsetsofSizeN(s, n):
    assert n < len(s)
    l_of_subsets = list(itertools.combinations(s, n))
    l_of_lists = [set(sub) for sub in l_of_subsets]
    return l_of_lists

class fedDefender:
    def __init__(self,  round, input_shape, dname: str):
        self.input_shape = input_shape
        self.dname = dname
        self.round = round
        self.fuzz_gen_data = None
        self.clients2fuzzinputs_neurons_activations = {}
        #self.client2layeracts = {}
        self.all_fedfuzz_seqs = []
        self.participating_clients_ids = None
        self.all_combinations = None
    def generation_test_data(self):
        min_t = -1
        max_t = 1
        random_generator = kaiming_normal_
        apply_transform = True
        num_inputs_test_data = 10
        fuzz_gen = FuzzGeneration(input_shape=self.input_shape, randomGenerator=random_generator,
                       apply_transform=apply_transform, dname=self.dname, majority_threshold=5, num_test_data=num_inputs_test_data,
                       min_t=min_t, max_t=max_t)
        self.fuzz_gen_data, _ = fuzz_gen.getFuzzInputs()  # это генерация как раз случайных тестовых данных
    def filter(self, result_clients, nc_t):
        self._updateNeuronCoverage(result_clients)
        for i in range(len(self.fuzz_gen_data)):
            seq = self._findNormalClientsSeqV1(i, nc_t)
            self.all_fedfuzz_seqs.append(seq)

    def _findNormalClientsSeqV1(self, input_id, nc_t):
        client2NC = {cid: self.clients2fuzzinputs_neurons_activations[cid][input_id] > nc_t for cid in self.clients2fuzzinputs_neurons_activations.keys()}
        clients_ids = self.getClientsIDsWithHighestCommonNeurons(client2NC)
        return clients_ids#вывод самой идеальной группы
    def getClientsIDsWithHighestCommonNeurons(self, clients2neurons2boolact ):#вот в этой функции происходит подсчет общих активных нейронов для каждой группы и нахождение самой идеальной группы
        # Найти нейроны, неактивные у всех клиентов (т.е. где все False)
        select_neurons = tf.logical_not(self.torchIntersetion(clients2neurons2boolact))

        clients_neurons2boolact = {
            cid: tf.boolean_mask(t, select_neurons) for cid, t in clients2neurons2boolact.items()
        }
        # Подсчет общего количества активных (True) нейронов среди пересечений по каждой комбинации
        count_of_common_neurons = []
        for comb in self.all_combinations:
            sub_dict = {cid: clients_neurons2boolact[cid] for cid in comb}
            intersection = self.torchIntersetion(sub_dict)
            intersection_true_sum = tf.reduce_sum(tf.cast(intersection, tf.int32)).numpy()
            count_of_common_neurons.append(intersection_true_sum)
        # Поиск комбинации с максимальным количеством общих активных нейронов
        highest_number_of_common_neurons = max(count_of_common_neurons)
        val_index = count_of_common_neurons.index(highest_number_of_common_neurons)
        val_parties_ids = self.all_combinations[val_index]
        return val_parties_ids
    def torchIntersetion(self, client2tensors):
        intersct = True
        for k, v in client2tensors.items():
            intersct = tf.logical_and(intersct, v)
        return intersct
    def _updateNeuronCoverage(self, result_clients):# вычисляет градиент активации нейронов для модели
        for client_id, model in result_clients.items():
            outs = [getNeuronCoverage(model, img) for img in self.fuzz_gen_data]# вычисляет градиент активации нейронов для каждого слоя в модели model для данного входного изображения img
            self.clients2fuzzinputs_neurons_activations[client_id] = [all_acts for all_acts, _ in outs]
            #self.client2layeracts[client_id] = [layer_acts for _, layer_acts in outs]
            gc.collect()# вызывает сборщик мусора для очистки памяти

    def getPotentialMaliciousClients(self, num_test_data):
        client2count = {cid: 0 for cid in self.participating_clients_ids}
        for comb in self.all_fedfuzz_seqs:
            # malicious_ids = malicious_ids.union(participating_clients_ids-comb)
            for cid in comb:
                client2count[cid] += 1
            # for cid in self.participating_clients_ids - comb:
            #     client2count[cid] += 1
        print("client2count", client2count)
        print("num_test_data", num_test_data)
        malicious2freq = {cid: count / num_test_data for cid, count in client2count.items()}  # получаются вероятности вроде как, того что клиент вредоносен, но вообще как бред выглядит
        return malicious2freq
    def zeroing_out_variables(self):
        self.fuzz_gen_data = None
        self.clients2fuzzinputs_neurons_activations = {}
        self.all_fedfuzz_seqs = []
        self.participating_clients_ids = None
        self.all_combinations = None
    def run_filter(self,result_clients, nc_t, part_math_wait):
        self.all_combinations = makeAllSubsetsofSizeN(set(list(result_clients.keys())), len(result_clients) - 1)
        self.participating_clients_ids = set(list(result_clients.keys()))
        self.generation_test_data()
        self.filter(result_clients, nc_t)
        result = self.getPotentialMaliciousClients(num_test_data= len(self.fuzz_gen_data))
        self.zeroing_out_variables()
        return  result, 0
    