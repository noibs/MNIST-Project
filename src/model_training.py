import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import layers, models
from keras.optimizers import Adam


# 1. Indlæs og forbehandl data
def load_data():
    # Hent MNIST data fra Keras
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Normaliser og reshaping. X er billeder, Y er one-hot encoded labels
    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)


# 2. CNN med data augmentation
def create_model(augmentation=0.2, dropout=0.5, learning_rate=0):
    model = models.Sequential([
        # Definer input lag
        layers.Input(shape=(28, 28, 1)),
        # Data augmentation
        layers.RandomRotation(augmentation),
        layers.RandomTranslation(augmentation, augmentation),
        layers.RandomZoom(augmentation),

        # CNN lag
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),

        # Fully connected lag
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout),
        # Output lag
        layers.Dense(10, activation='softmax')
    ])

    # Optimizer
    # Brug brugerdefineret learning rate hvis angivet, ellers standard Adam
    if learning_rate > 0:
        optimizer = Adam(learning_rate=learning_rate)
    else:
        optimizer = 'adam'  # Standard Adam

    # Compile model
    model.compile(
        optimizer=optimizer,
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
        epochs=5,
        validation_data=(x_test, y_test),
        verbose=1
    )

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_acc*100:.2f}%")
    model.save("mnist_model_baseline.keras")
    print("✅ Model gemt!")

    # Vis træningshistorik
    plot_history(history)
    return model, history

if __name__ == "__main__":
    np.random.seed(42)
    train_model()
