from keras import optimizers
from keras import losses
def testing_model(model, test_ds):
    print("Testing model...")
    optimizer = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    model.model.compile(optimizer=optimizer,
                        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])
    # Оценка модели на тестовых данных
    test_loss, test_acc = model.model.evaluate(test_ds)
    print(f"Test loss: {test_loss}, Test accuracy: {test_acc}")
    return test_loss, test_acc