import torch
from torch.utils.data import Dataset

class AttackLabelFlipping(Dataset):
    def __init__(self, dataset, class_ids_to_flip, flip_target_class_id):
        """
        dataset: оригинальный датасет
        class_ids_to_flip: список исходных классов, которые надо изменить
        flip_target_class_id: метка, на которую будет заменяться
        """
        self.dataset = dataset
        self.class_ids = class_ids_to_flip
        self.target_class_id = flip_target_class_id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if y in self.class_ids:
            y = self.target_class_id
        return x, y

def label_flipping_attack(train_ds, client_id, classes_to_flip, flip_target):
    print(f"Injecting label flipping attack to client {client_id}")
    flipped_dataset = AttackLabelFlipping(dataset=train_ds, class_ids_to_flip=classes_to_flip, flip_target_class_id=flip_target)
    return flipped_dataset