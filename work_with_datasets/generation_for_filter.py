import torch
import torchvision.transforms.functional as F
import time

class FuzzGeneration:
    def __init__(self, input_shape, randomGenerator, apply_transform, dname=None, num_test_data=10, majority_threshold=5, time_delta=60, min_t=-1, max_t=1):
        self.majority_threshold = majority_threshold
        self.dname = dname
        print(f"Majority Threshold {self.majority_threshold}")
        self.same_seqs_set = set()
        self.num_test_data = num_test_data
        self.transform = None
        self.device = torch.device("cpu")
        self.fuzz_inputs = []
        self.input_shape = input_shape  # Ожидаемая форма должна быть (H, W, C) теперь!
        self.time_delta = time_delta
        self.min_t = min_t
        self.max_t = max_t
        self.apply_transform = apply_transform
        self.randomGenerator = None
        func_names = [f.__name__ for f in [torch.nn.init.uniform_, torch.nn.init.normal_, torch.nn.init.xavier_uniform_,
                                           torch.nn.init.xavier_normal_, torch.nn.init.kaiming_uniform_, torch.nn.init.kaiming_normal_,
                                           torch.nn.init.trunc_normal_, torch.nn.init.orthogonal_]]
        if randomGenerator.__name__ in func_names:
            self.randomGenerator = randomGenerator
        else:
            raise Exception(f"Error: {type(randomGenerator)} not supported")

        if dname is not None:
            self._setup_normalization(dname)

    def _setup_normalization(self, dname):
        if dname.lower() == "cifar10":
            self.mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32)
            self.std = torch.tensor([0.2023, 0.1994, 0.2010], dtype=torch.float32)
        elif dname.lower() == "mnist":
            self.mean = torch.tensor([0.1307], dtype=torch.float32)
            self.std = torch.tensor([0.3081], dtype=torch.float32)
        else:
            raise Exception(f"Dataset {dname} not supported")

    def _normalize(self, img):
        # img ожидается (H, W, C)
        img = img.permute(2, 0, 1)  # (C, H, W) для Normalize
        img = F.normalize(img, self.mean, self.std)
        img = img.permute(1, 2, 0)  # обратно (H, W, C)
        return img

    def getFuzzInputs(self):
        return self._simpleFuzzInputs()

    def _simpleFuzzInputs(self):
        print("Simple Fuzz inputs")
        start = time.time()
        fuzz_inputs = [self._getRandomInput() for _ in range(self.num_test_data)]
        return fuzz_inputs, time.time() - start

    def _getRandomInput(self):
        #print("Shape:", self.input_shape, type(self.input_shape), type(self.input_shape[0]))
        img = torch.empty((self.input_shape[0], self.input_shape[1], self.input_shape[2]))  # (H, W, C)
        self.randomGenerator(img)

        if self.apply_transform and self.dname is not None:
            img = self._normalize(img)

        # Добавляем батч размерность (1, H, W, C)
        img = img.unsqueeze(0)

        img = img.to(dtype=torch.float32)
        return img
