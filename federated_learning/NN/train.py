from keras import optimizers
from keras import losses
def training_model(model, epochs, train_ds, test_ds):
    print("Training model...")
    optimizer = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    model.model.compile(optimizer=optimizer,
                  loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.model.fit(train_ds, epochs=epochs, validation_data=test_ds)

    return model