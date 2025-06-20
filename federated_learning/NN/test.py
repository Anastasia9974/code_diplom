from keras import optimizers
from keras import losses
def testing_model(model, test_ds):
    print("Testing model...")
    model.model.compile(optimizer= "adam",
                        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])
    # Оценка модели на тестовых данных
    test_loss, test_acc = model.model.evaluate(test_ds,verbose=2)
    print(f"Test loss: {test_loss}, Test accuracy: {test_acc}")
    return test_loss, test_acc