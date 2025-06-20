from keras import optimizers
from keras import losses
def training_model(model, epochs, train_ds, test_ds):
    print("Training model...")
    model.model.compile(optimizer="adam",
                  loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.model.fit(train_ds, epochs=epochs, validation_data=test_ds)

    return model