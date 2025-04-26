from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
import torch
import tensorflow as tf
import numpy as np
class WorkWithDataset:
    # Сначала ведется получение датасетов
    def __init__(self):
        self.test_ds = None
        self.train_ds = None
        self.dataset = None
        self. trainloader_client = []
        self.testloader_client = []
        self.testloader_server = None
        self.trainloader_server = None
    def download_dataset(self, data_name: str):
        path = f"/home/anvi/code_diplom/new_code/training_data/{data_name}/"
        if data_name == 'cifar10':
            print("Downloading CIFAR10 dataset")
            transform = transforms.Compose([
                transforms.ToTensor(),  # Преобразуем изображение в тензор (C, H, W)
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                transforms.Lambda(lambda x: x.permute(1, 2, 0))  # Меняем оси (H, W, C) и в NumPy
            ])
            self.train_ds = datasets.CIFAR10(root=path, download=True, train=True, transform=transform)
            self.test_ds = datasets.CIFAR10(root=path, download=True, train=False, transform=transform)

        elif data_name == 'mnist':
            print("Downloading MNIST dataset")
            transform = transforms.Compose([
                transforms.ToTensor(),  # (1, 28, 28)
                transforms.Normalize((0.1307,), (0.3081,)),  # Сначала нормализация
                transforms.Lambda(lambda x: x.permute(1, 2, 0)),  # Потом (28, 28, 1)
            ])
            self.train_ds = datasets.MNIST(root=path, download=True, train=True, transform=transform)
            self.test_ds = datasets.MNIST(root=path, download=True, train=False, transform=transform)

        else:
            print("Not a valid dataset")
    # обработка данных
    def data_processing(self, input_shape, batch_size:int = 512):
        def dataset_to_tf(dataset, batch_size, input_shape):
            def generator():
                for image, label in dataset:
                    yield image.numpy(), np.int32(label)  # Преобразуем PyTorch-данные

            return tf.data.Dataset.from_generator(
                generator,
                output_signature=(
                    tf.TensorSpec(shape=input_shape, dtype=tf.float32),
                    tf.TensorSpec(shape=(), dtype=tf.int32)
                )
            ).batch(batch_size)

        self.trainloader_server = dataset_to_tf(self.train_ds,batch_size, input_shape=input_shape)
        self.testloader_server = dataset_to_tf(self.test_ds,batch_size, input_shape=input_shape)
        for i in range(len(self.trainloader_client)):
            self.trainloader_client[i] = dataset_to_tf(self.trainloader_client[i],batch_size, input_shape=input_shape)
            self.testloader_client[i] = dataset_to_tf(self.testloader_client[i],batch_size, input_shape=input_shape)
    # деление датасета на всех клиентов
    def data_division(self, num_client: int, test_percent: int):
        partition_size = int(len(self.train_ds) / num_client)
        lengths = [partition_size]*num_client
        self.dataset = random_split(self.train_ds, lengths, torch.Generator().manual_seed(42))
        for ds in self.dataset:
            len_test_client = len(ds)//test_percent
            len_train_client = len(ds) - len_test_client
            lengths = [len_train_client, len_test_client]
            ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
            self.trainloader_client.append(ds_train)
            self.testloader_client.append(ds_val)
        self.trainloader_server = self.train_ds
        self.testloader_server = self.test_ds

    # выводится информация что за датасет
    def show_dataset(self):

        ...
    def get_dataset(self):
        return self.trainloader_client, self.testloader_client, self.trainloader_server, self.testloader_server
    def set_dataset(self, trainloader_client, testloader_client, trainloader_server, testloader_server):
        self.trainloader_client = trainloader_client
        self.testloader_client = testloader_client
        self.trainloader_server = trainloader_server
        self.testloader_server = testloader_server

#тестирование функкционала
#db = WorkWithDataset()
#db.download_dataset(data_name ='cifar10')
#db.data_division(num_client=10)
#db.data_processing(data_name = 'cifar10')
