from new_code.attacks.backdoor_atack import backdoor
from new_code.attacks.poisoning_attack import label_flipping_attack
def get_attacks(attacks: str, client_id, data_for_cl, bad_clients):
    if attacks == "backdoor" and (int(client_id) in bad_clients):
        print(f"bad_clients: {bad_clients}")
        data_for_cl["train_data"][int(client_id)] = backdoor(train_ds=data_for_cl["train_data"][int(client_id)],
                                                             clients_id=client_id)
        data_for_cl["test_data"][int(client_id)] = backdoor(train_ds=data_for_cl["test_data"][int(client_id)],
                                                            clients_id=client_id)
    elif attacks == "label_flipping" and (int(client_id) in bad_clients):
        # тут какая нибудь другая атака должна быть
        print(f"bad_clients label_flipping: {bad_clients}")
        data_for_cl["train_data"][int(client_id)] = label_flipping_attack(
            train_ds=data_for_cl["train_data"][int(client_id)], client_id=client_id,
            classes_to_flip=[0, 1, 2, 3, 4, 5, 6, 7, 8], flip_target=9)
        data_for_cl["test_data"][int(client_id)] = label_flipping_attack(
            train_ds=data_for_cl["test_data"][int(client_id)], client_id=client_id,
            classes_to_flip=[0, 1, 2, 3, 4, 5, 6, 7, 8], flip_target=9)
    return data_for_cl