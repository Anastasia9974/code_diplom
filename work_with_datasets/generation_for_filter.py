import time
import torchvision
from torchvision import transforms
import torch
from torch.nn.init import (kaiming_normal_, kaiming_uniform_, normal_,
                           orthogonal_, trunc_normal_, uniform_,
                           xavier_normal_, xavier_uniform_)



class FuzzGeneration:
    #client2models, all_combinations, dname, input_shape, n_fuzz_inputs=10, random_generator=kaiming_normal_, apply_transform=True, nc_thresholds=[0.0], num_bugs=1, use_gpu=True
    def __init__(self, input_shape, randomGenerator, apply_transform, dname=None, num_test_data=10, majority_threshold=5, time_delta=60, min_t=-1, max_t=1):
        self.majority_threshold = majority_threshold
        print(f"Majority Threshold {self.majority_threshold}")
        self.same_seqs_set = set()
        self.num_test_data = num_test_data
        self.size = 1024
        self.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        self.device = torch.device("cpu")
        self.fuzz_inputs = []
        self.input_shape = input_shape
        self.time_delta = time_delta
        self.min_t = min_t
        self.max_t = max_t
        self.apply_transform = apply_transform
        self.randomGenerator = None
        func_names = [f.__name__ for f in [uniform_, normal_, xavier_uniform_,
                                           xavier_normal_, kaiming_uniform_, kaiming_normal_, trunc_normal_,
                                           orthogonal_]]
        if randomGenerator.__name__ in func_names:
            self.randomGenerator = randomGenerator
        else:
            raise Exception(f"Error: {type(randomGenerator)} not supported")
        if dname is not None:
            self.transform = self._getDataSetTransformation(dname)

    def _getDataSetTransformation(self, dname):
        if dname in ["CIFAR10", "cifar10"]:
            return transforms.Compose([
                transforms.Lambda(lambda x: x.permute(0, 2, 1)),  # Меняем оси (H, W, C) и в NumPy
                transforms.Lambda(lambda x: x / 255.0)  # Нормализация 0-1
            ])
        #elif dname in ["femnist", "mnist"]:
         #   return torchvision.transforms.Compose([
         #       # torchvision.transforms.ToTensor(),
         #       torchvision.transforms.Normalize((0.1307,), (0.3081,))
         #   ])
        elif dname == "fashionmnist":
            return torchvision.transforms.Compose([
                torchvision.transforms.Normalize((0.5,), (0.5,))
            ])
        else:
            raise Exception(f"Dataset {dname} not supported")

    #5 fun
    def getFuzzInputs(self):#возвращает n_fuzz_inputs тестовых данных
        return self._simpleFuzzInputs()

    def _simpleFuzzInputs(self):
        print("Sime Fuzz inputs")
        start = time.time()
        fuzz_inputs = [self._getRandomInput() for _ in range(self.num_test_data)]
        return fuzz_inputs, time.time() - start

    def _getRandomInput(self):
        print("Shape:", self.input_shape, type(self.input_shape), type(self.input_shape[0]))
        img = torch.empty(self.input_shape)
        self.randomGenerator(img)

        if self.apply_transform:
            img = self.transform(img)  # применим пользовательскую трансформацию (например, нормализация)

        # Преобразуем из (C, H, W) → (H, W, C)
        img = img.permute(0, 2, 1)

        # Добавим батч-дименсию: (1, H, W, C)
        img = img.unsqueeze(0)

        # Приведем к типу, ожидаемому Keras
        img = img.to(dtype=torch.float32)
        print(img.shape)
        return img