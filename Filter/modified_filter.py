import time
from torch.nn.init import (kaiming_normal_, kaiming_uniform_, normal_,
                           orthogonal_, trunc_normal_, uniform_,
                           xavier_normal_, xavier_uniform_)
import torch
import gc
import itertools
import torchvision
from captum.attr import LayerGradientXActivation
from new_code.work_with_datasets.generation_for_filter import FuzzGeneration
from keras import Model
import keras
import tensorflow as tf
from new_code.conf import resualt_work
import numpy as np

# def getAllLayers(model):
#     layers = []
#     for layer in model.layers:
#         # Если слой сам по себе (не вложенная модель)
#         if not hasattr(layer, 'layers') and (isinstance(layer, (keras.layers.Conv2D, keras.layers.Dense))):
#             layers.append(layer)
#         # Если слой содержит вложенные слои (например, модель внутри модели или вложенные блоки)
#         elif hasattr(layer, 'layers'):
#             temp_layers = getAllLayers(layer)
#             layers.extend(temp_layers)
#     return layers
def get_activations(model, img):
    # Преобразуем PyTorch тензор в NumPy массив
    torch_tensor_numpy = img.detach().cpu().numpy()  # Используем .detach() чтобы избежать зависимостей от графа, если это требуется
    # Преобразуем NumPy массив в TensorFlow тензор
    img = tf.convert_to_tensor(torch_tensor_numpy, dtype=tf.float32)
    img = tf.Variable(img)  # нужно, чтобы у img можно было считать градиент
    _ = model(img, training=False)
    # Получаем выходы всех слоев
    outputs = [layer.output for layer in model.layers]
    # Создаем новую модель: вход — такой же, выход — все активации
    activation_model = Model(inputs=model.inputs, outputs=outputs)
    with tf.GradientTape() as tape:
        tape.watch(img)  # Следим за img для вычисления градиента
        activations = activation_model(img, training=False)

    grads = tape.gradient(activations[-1], img)

    grad_times_activations = []
    layer2output = []
    all_activations = []  # Список для всех активаций

    for activation in activations:
        # Получаем количество активированных нейронов для данного слоя
        if len(activation.shape) > 3:  # Если 4D (batch_size, height, width, channels)
            # Для свёрточных слоев
            active_neurons = tf.reduce_sum(tf.cast(activation > 0, tf.float32)).numpy()  # Считаем активированные нейроны
        elif len(activation.shape) == 2:  # Если 2D (batch_size, features)
            # Для полносвязанных слоёв
            active_neurons = tf.reduce_sum(tf.cast(activation > 0, tf.float32)).numpy()  # Считаем активированные нейроны
        else:
            active_neurons = 0

        # Обрабатываем активации для градиентов
        if len(grads.shape) > 1 and len(activation.shape) > 3:  # Если 4D (batch_size, height, width, channels)
            # Изменяем размер градиентов, если необходимо
            if grads.shape[1:3] != activation.shape[1:3]:
                grads_resized = tf.image.resize(grads, size=(activation.shape[1], activation.shape[2]))
            else:
                grads_resized = grads
            grad_times_activation = grads_resized * activation
            grad_times_activations.append(grad_times_activation)
        elif len(activation.shape) == 2:  # Если 2D (batch_size, features)
            # Просто пропускаем или обрабатываем по-другому (зависит от задачи)
            grad_times_activations.append(None)
        else:
            grad_times_activations.append(None)

        # Сплющиваем активацию для каждого слоя
        flattened_activation = tf.reshape(activation, [-1])  # Сплющиваем активацию для данного слоя
        layer2output.append(flattened_activation)  # Добавляем сплющенные активации для данного слоя
        all_activations.append(flattened_activation.numpy())  # Добавляем в список для всех слоев

    # Конкатенируем все активации в одну
    concatenated_activations = np.concatenate(all_activations, axis=0)

    # Возвращаем объединенные активации и список слоёв с их активациями
    return concatenated_activations, layer2output



def getNeuronCoverage(model, img):
    r = get_activations(model=model, img=img)
    return r



class ModifiedNewFilter:
    def __init__(self, round, dname, input_shape):
        self.round = round
        self.dname = dname
        self.input_shape = input_shape
        self.fuzz_gen_data = None
        self.clients2fuzzinputs_neurons_activations = {}
        self.client2layeracts = {}
        self.all_fedfuzz_seqs = []

    def generation_test_data(self):
        min_t = -1
        max_t = 1
        random_generator = kaiming_normal_
        apply_transform = True
        num_inputs_test_data = 10
        fuzz_gen = FuzzGeneration(input_shape=self.input_shape, randomGenerator=random_generator,
                       apply_transform=apply_transform, dname=self.dname, majority_threshold=5, num_test_data=num_inputs_test_data,
                       min_t=min_t, max_t=max_t)
        self.fuzz_gen_data, input_gen_time = fuzz_gen.getFuzzInputs()  # это генерация как раз случайных тестовых данных
    def filter(self, result_clients, nc_t, part_math_wait):
        self._updateNeuronCoverage(result_clients)
        print(f"len(self.fuzz_inputs): {len(self.fuzz_gen_data)}")
        for i in range(len(self.fuzz_gen_data)):
            seq = self._findNormalClientsSeqV1(i, nc_t, part_math_wait)
            self.all_fedfuzz_seqs += seq
    def _findNormalClientsSeqV1(self, input_id, nc_t, part_math_wait):
        client2NC = {cid: self.clients2fuzzinputs_neurons_activations[cid][input_id] > nc_t for cid in self.clients2fuzzinputs_neurons_activations.keys()}
        clients_ids = self.getClientsIDsWithHighestCommonNeurons(client2NC,part_math_wait)
        return clients_ids#вывод самой идеальной группы

    def getClientsIDsWithHighestCommonNeurons(self, clients2neurons2boolact, part_math_wait=0.01):
        select_neurons = self.torchIntersetion(clients2neurons2boolact) == False

        clients_neurons2boolact = {
            cid: tf.boolean_mask(t, select_neurons) for cid, t in clients2neurons2boolact.items()
        }

        sed1 = {}
        total_sum = 0

        for cid in clients_neurons2boolact:
            # Замените .sum() на tf.reduce_sum()
            sed = tf.reduce_sum(tf.cast(self.torchIntersetion({cid: clients_neurons2boolact[cid]}) == True, tf.int32))
            total_sum += sed.numpy()  # Получаем числовое значение из тензора
            sed1[cid] = sed.numpy()  # Сохраняем значение в sed1


        # Среднее значение для всех клиентов
        math_wait = total_sum / len(clients_neurons2boolact)
        pogr = math_wait * part_math_wait
        without_attacks = 0
        print(sed1)
        print(pogr)
        print(math_wait)
        #sed1 количиство нейронов совпадающих для клиентов между собой
        with_attacks_ind = []

        for cid in sed1:
            if sed1[cid] > math_wait - pogr:
                without_attacks += 1
            else:
                with_attacks_ind.append(cid)

        print(f"Without_attacks: {without_attacks}")
        resualt_work.resualt_get_data_for_model[f"part_math_wait_{part_math_wait}"][f"round:{self.round}"] = {}
        resualt_work.resualt_get_data_for_model[f"part_math_wait_{part_math_wait}"][f"round:{self.round}"]["diff_neurons"] = int(sed1[max(sed1, key=sed1.get)] - sed1[min(sed1, key=sed1.get)])
        return with_attacks_ind

    def torchIntersetion(self, client2tensors):
        intersct = True
        for k, v in client2tensors.items():
            intersct = tf.logical_and(intersct, v)
        return intersct
    def _updateNeuronCoverage(self, client2model):# вычисляет градиент активации нейронов для модели
        device = torch.device("cuda")

        for client_id, model in client2model.items():
            outs = [getNeuronCoverage(model, img.to(device)) for img in self.fuzz_gen_data]# вычисляет градиент активации нейронов для каждого слоя в модели model для данного входного изображения img
            self.clients2fuzzinputs_neurons_activations[client_id] = [all_acts for all_acts, _ in outs]
            #self.client2layeracts[client_id] = [layer_acts for _, layer_acts in outs]
            gc.collect()# вызывает сборщик мусора для очистки памяти
    def zeroing_out_variables(self):
        self.fuzz_gen_data = None
        self.clients2fuzzinputs_neurons_activations = {}
        self.all_fedfuzz_seqs = []


    def getPotentialMaliciousClients(self, num_test_data):
        num_ids = {}

        for item in self.all_fedfuzz_seqs:
            if item in num_ids:
                num_ids[item] += 1
            else:
                num_ids[item] = 1
        probability_malicious_ids = {}
        for cid in self.clients2fuzzinputs_neurons_activations.keys():
            probability_malicious_ids[cid] = 1
        for key, value in num_ids.items():
            probability_malicious_ids[key] = 1 - value / num_test_data
        return probability_malicious_ids

    def run_filter(self,result_clients, nc_t, part_math_wait):
        self.generation_test_data()
        self.filter(result_clients, nc_t, part_math_wait)
        result = self.getPotentialMaliciousClients(num_test_data=len(self.fuzz_gen_data))
        self.zeroing_out_variables()
        return result, 0
