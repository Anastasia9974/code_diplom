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
def getNeuronCoverage(model, img):
    r = get_activations(model=model, input_tensor=img)
    return r


#4 fun (генерация тестовых данных, в теории это не меняется но надо посмотреть вдруг надо преобразование производить и в теории это мучше в работу с данными кинуть)
# class FuzzGeneration:
#     #client2models, all_combinations, dname, input_shape, n_fuzz_inputs=10, random_generator=kaiming_normal_, apply_transform=True, nc_thresholds=[0.0], num_bugs=1, use_gpu=True
#     def __init__(self, shape, randomGenerator, apply_transform, dname=None, n_fuzz_inputs=10, majority_threshold=5, time_delta=60, min_t=-1, max_t=1):
#         self.majority_threshold = majority_threshold
#         print(f"Majority Threshold {self.majority_threshold}")
#         self.same_seqs_set = set()
#         self.n_fuzz_inputs = n_fuzz_inputs
#         self.size = 1024
#         self.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
#         self.device = torch.device("cpu")
#         self.fuzz_inputs = []
#         self.input_shape = shape
#         self.time_delta = time_delta
#         self.min_t = min_t
#         self.max_t = max_t
#         self.apply_transform = apply_transform
#         self.randomGenerator = None
#         func_names = [f.__name__ for f in [uniform_, normal_, xavier_uniform_,
#                                            xavier_normal_, kaiming_uniform_, kaiming_normal_, trunc_normal_,
#                                            orthogonal_]]
#         if randomGenerator.__name__ in func_names:
#             self.randomGenerator = randomGenerator
#         else:
#             raise Exception(f"Error: {type(randomGenerator)} not supported")
#         if dname is not None:
#             self.transform = self._getDataSetTransformation(dname)
#             # print("Orignal data transform.")
#         # if use_gpu:
#         #    self.device = torch.device("cuda")
#     def _getDataSetTransformation(self, dname):
#         if dname in ["CIFAR10", "cifar10"]:
#             return torchvision.transforms.Compose([
#                 # torchvision.transforms.ToTensor(),
#                 torchvision.transforms.Normalize(
#                     (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#             ])
#         elif dname in ["femnist", "mnist"]:
#             return torchvision.transforms.Compose([
#                 # torchvision.transforms.ToTensor(),
#                 torchvision.transforms.Normalize((0.1307,), (0.3081,))
#             ])
#         elif dname == "fashionmnist":
#             return torchvision.transforms.Compose([
#                 # torchvision.transforms.ToTensor(),
#                 torchvision.transforms.Normalize((0.5,), (0.5,))
#             ])
#         else:
#             raise Exception(f"Dataset {dname} not supported")
#
#     #5 fun
#     def getFuzzInputs(self):#возвращает n_fuzz_inputs тестовых данных
#         return self._simpleFuzzInputs()
#         # return self._generateFeedBackFuzzInputs1()
#     def _simpleFuzzInputs(self):
#         print("Sime Fuzz inputs")
#         start = time.time()
#         fuzz_inputs = [self._getRandomInput() for _ in range(self.n_fuzz_inputs)]
#         return fuzz_inputs, time.time() - start
#     def _getRandomInput(self):
#         img = torch.empty(self.input_shape)
#         self.randomGenerator(img)
#         if self.apply_transform:
#             return self.transform(img)
#         return img


#2 fun




# 6 fun
# class FedFuzz:
#     def __init__(self, client2model, fuzz_inputs, use_gpu) -> None:
#         self.fuzz_inputs = fuzz_inputs
#         self.use_gpu = use_gpu
#         self.clients2fuzzinputs_neurons_activations = {}
#         self.client2layeracts = {}
#         self._updateNeuronCoverage(client2model)
#         self.clientids = set([c for c in client2model.keys()])
#     def _updateNeuronCoverage(self, client2model):# вычисляет градиент активации нейронов для модели
#         device = torch.device("cpu")
#         # if self.use_gpu:
#         # device = torch.device("cuda")
#
#         for client_id, model in client2model.items():
#             model = model.to(torch.device("cuda"))
#             outs = [getNeuronCoverage(model, img.to(device), device) for img in self.fuzz_inputs]# вычисляет градиент активации нейронов для каждого слоя в модели model для данного входного изображения img
#             self.clients2fuzzinputs_neurons_activations[client_id] = [all_acts for all_acts, _ in outs]
#             self.client2layeracts[client_id] = [layer_acts for _, layer_acts in outs]
#
#
#             model = model.to(torch.device("cuda"))
#             gc.collect()# вызывает сборщик мусора для очистки памяти
#     def runFedFuzz(self, nc_t, round):
#         all_fedfuzz_seqs = []
#         print(f"len(self.fuzz_inputs): {len(self.fuzz_inputs)}")
#         for i in range(len(self.fuzz_inputs)):
#             seq = self._findNormalClientsSeqV1(i, nc_t, round)
#             all_fedfuzz_seqs += seq
#             #all_fedfuzz_seqs.append(seq)
#         return all_fedfuzz_seqs, len(self.fuzz_inputs) # вывод всех наборов групп у которых совпадает максимальное кол-во нейронов(своя группа для каждого тестового изображения)
#     def _findNormalClientsSeqV1(self, input_id, nc_t, round):
#         client2NC = {cid: self.clients2fuzzinputs_neurons_activations[cid][input_id] > nc_t for cid in self.clients2fuzzinputs_neurons_activations.keys()}
#         clients_ids = self.getClientsIDsWithHighestCommonNeurons(client2NC, round)
#         return clients_ids#вывод самой идеальной группы
#     def getClientsIDsWithHighestCommonNeurons(self, clients2neurons2boolact, round):#вот в этой функции происходит подсчет общих активных нейронов для каждой группы и нахождение самой идеальной группы
#         select_neurons = self.torchIntersetion(clients2neurons2boolact) == False
#         clients_neurons2boolact = {cid: t[select_neurons] for cid, t in clients2neurons2boolact.items()}
#         sed1 = {}
#         sum = 0
#         for cid in clients_neurons2boolact:
#             sed = (self.torchIntersetion({cid: clients_neurons2boolact[cid]}) == True).sum().item()
#             sum += sed
#             sed1[cid] = sed
#         math_wait = (sum / len(clients_neurons2boolact))
#         qwe = 0.1
#         if round >= 6:
#              qwe=0.05
#         pogr = math_wait*qwe
#
#         without_attacks = 0
#         print(sed1)
#         print(pogr)
#         print(math_wait)
#         with_attacks_ind = []
#         for cid in sed1:
#             if sed1[cid] > math_wait-pogr:
#                 without_attacks +=1
#             else:
#                 with_attacks_ind.append(cid)
#         print(f"With attacks: {without_attacks}")
#         return with_attacks_ind
#
#     def torchIntersetion(self, client2tensors):
#         intersct = True
#         for k, v in client2tensors.items():
#             intersct = intersct * v
#         return intersct

# # 3 fun
# def runFedfuzz(client2models, dname, input_shape, round, n_fuzz_inputs=10, random_generator=kaiming_normal_, apply_transform=True, nc_thresholds=[0.0],  use_gpu=True):
#     min_t = -1
#     max_t = 1
#
#     fuzz_gen = FuzzGeneration(client2models, input_shape, use_gpu, randomGenerator=random_generator, apply_transform=apply_transform, dname=dname, majority_threshold=5, num_test_data=n_fuzz_inputs, min_t=min_t, max_t=max_t)
#     fuzz_inputs, input_gen_time = fuzz_gen.getFuzzInputs()  # это генерация как раз случайных тестовых данных
#
#     total_time = 0
#     start = time.time()
#     fedfuzz = FedFuzz(client2models, fuzz_inputs, use_gpu=use_gpu)
#     total_time += (time.time() - start)
#     start = time.time()
#     fedfuzz_results, num_test_data = fedfuzz.runFedFuzz(0.0, round)# используются для поиска группы клиентов или идентификаторов клиентов, которые имеют наибольшее количество общих нейронов,
#
#     fedfuzz_time = total_time + ((time.time() - start) / len(nc_thresholds))
#     return fedfuzz_results, input_gen_time, fedfuzz_time, num_test_data
#
#
# def getPotentialMaliciousClients(fedfuzz_clients_combs,num_test_data):
#     num_ids = {}
#     for item in fedfuzz_clients_combs:
#         if item in num_ids:
#             num_ids[item] += 1
#         else:
#             num_ids[item] = 1
#     probability_malicious_ids = {}
#     for key, value in num_ids.items():
#         probability_malicious_ids[key] = value / num_test_data
#     return probability_malicious_ids
#
#
#
# #1 fun
# # модели клиентов, количество тестовых входных данных, датасет для обучения, и id вредоносных клиентов для проверки(проверка сторонняя, на метод не влияет)
# def fedFuzzDefense(client2model, input_shape, dname: str, round):
#     fedfuzz_acc = -1
#     print(f"len(input_shape): {len(input_shape)}")
#     fedfuzz_combs, _, _, num_test_data = runFedfuzz(client2model, dname=dname, input_shape=input_shape, round =round)  # находит подмножество клиентов с общими нейронами(своя группа для каждого тестового изображения)
#     '''len(client2model)'''
#     malacious2confidence = getPotentialMaliciousClients(fedfuzz_combs, num_test_data)  # получение вероятности насколько слиент вредоносен
#     return malacious2confidence, fedfuzz_acc


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
        #return self.all_fedfuzz_seqs, len(self.fuzz_gen_data)
    def _findNormalClientsSeqV1(self, input_id, nc_t, part_math_wait):
        client2NC = {cid: self.clients2fuzzinputs_neurons_activations[cid][input_id] > nc_t for cid in self.clients2fuzzinputs_neurons_activations.keys()}
        clients_ids = self.getClientsIDsWithHighestCommonNeurons(client2NC,part_math_wait)
        return clients_ids#вывод самой идеальной группы

    def getClientsIDsWithHighestCommonNeurons(self, clients2neurons2boolact, part_math_wait):
        select_neurons = tf.logical_not(self.torchIntersetion(clients2neurons2boolact))

        clients_neurons2boolact = {
            cid: tf.boolean_mask(t, select_neurons) for cid, t in clients2neurons2boolact.items()
        }

        sed1 = {}
        total_sum = 0

        for cid in clients_neurons2boolact:
            # Замените .sum() на tf.reduce_sum()
            sed = tf.reduce_sum(tf.cast(self.torchIntersetion({cid: clients_neurons2boolact[cid]}), tf.int32))
            total_sum += sed.numpy()  # Получаем числовое значение из тензора
            sed1[cid] = sed.numpy()  # Сохраняем значение в sed1


        # Среднее значение для всех клиентов
        math_wait = total_sum / len(clients_neurons2boolact)

        #!!!!!!!!!!!!!!!!вот начиная от сюда надо менять, надо как-то модель написать чтобы параметр pogr высчитывала
        #на основе сходимости модели и номера раунда (возможно числа клиентов)
        # qwe = 0.3
        # if self.round >= 6:
        #     qwe = 0.05

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
