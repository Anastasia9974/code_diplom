import torch
from torch.utils.data import Dataset
class AttackBackdoor(Dataset):
    def __init__(self, dataset, class_ids_to_poison, attack_pattern, backdoor_target_class_id):
        self.dataset = dataset
        self.class_ids = class_ids_to_poison
        self.attack_pattern = attack_pattern
        self.target_class_id = backdoor_target_class_id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if y in self.class_ids:
            y = self.target_class_id
            x += self.attack_pattern
        return x, y
def getBackDoorPatterGrey(shape):
    pattern = torch.zeros(shape)
    # pattern[22:,22:] = 255
    pattern[30:,30:] = 246
    return pattern



def backdoor(train_ds, clients_id):
    print(f"Injecting backdoor to client {clients_id}")
    s = train_ds[0][0].shape
    backdor_dataset = AttackBackdoor(dataset=train_ds, class_ids_to_poison=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                                     attack_pattern=getBackDoorPatterGrey(s), backdoor_target_class_id=9)
    train_ds = backdor_dataset
    return train_ds