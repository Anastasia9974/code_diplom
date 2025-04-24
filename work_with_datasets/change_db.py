from new_code.work_with_datasets.get_dataset import WorkWithDataset
def change_db(db_cl, batch_size:int, cid:int):
    db = WorkWithDataset()
    db.set_dataset(db_cl["train_data"], db_cl["test_data"], db_cl["all_train_data"], db_cl["all_test_data"])
    db.data_processing(batch_size=batch_size)
    train_data, test_data, _, _ = db.get_dataset()
    return train_data[cid], test_data[cid]