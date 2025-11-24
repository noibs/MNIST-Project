import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import layers, models


# 1. Indlæs og forbehandl data

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)


# 2. CNN med data augmentation
def create_model():
    randomness =  0.2
    model = models.Sequential([
        # Data augmentation
        layers.RandomRotation(randomness),
        layers.RandomTranslation(randomness, randomness),
        layers.RandomZoom(randomness),

        # CNN lag
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# 3. Plot kurver
def plot_history(history):
    plt.plot(history.history['accuracy'], label='Training accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


# 4. Træn og gem model
def train_model():
    (x_train, y_train), (x_test, y_test) = load_data()
    model = create_model()
    model.summary()

    history = model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=15,
        validation_data=(x_test, y_test),
        verbose=1
    )

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_acc*100:.2f}%")
    model.save("mnist_model.keras")
    print("✅ Model gemt!")

    return model, history


# Run
if __name__ == "__main__":
    np.random.seed(42)
    train_model()
