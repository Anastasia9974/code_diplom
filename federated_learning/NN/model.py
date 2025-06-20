#сделаны классы для определения моделей
from keras import Sequential
from keras import layers
class CNN():
    def __init__(self, input_shape):
        self.model = Sequential()

        # Conv1
        self.model.add(layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=input_shape))
        self.model.add(layers.MaxPooling2D(pool_size=2))

        # Conv2
        self.model.add(layers.Conv2D(64, kernel_size=3, activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size=2))

        self.model.add(layers.Conv2D(64, kernel_size=3, activation='relu'))
        # Flatten
        self.model.add(layers.Flatten())

        # Fully connected layers
        self.model.add(layers.Dense(64, activation='relu'))

        self.model.add(layers.Dense(10))

    def show_arhitecture(self):
        print(self.model.summary())

#тестирование данного кода
#model = CNN()
#model.show_arhitecture()

###self.model.add(layers.Dense(256))
###self.model.add(layers.LeakyReLU(alpha=0.1))
###self.model.add(layers.Dropout(0.50))