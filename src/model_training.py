# model_training.py - KERAS 3 VERSION
import numpy as np
import keras
from keras import layers
from keras.models import Sequential
from keras.datasets import mnist


def load_and_preprocess_data():
    """Indlæs og forbehandl MNIST data"""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape til (samples, 28, 28, 1)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255.0

    # One-hot encode labels
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)


def create_augmentation_layers():
    """
    Opret data augmentation som Keras layers (Keras 3 metode)

    Disse layers bliver en del af modellen og kører automatisk
    under træning (men ikke under evaluering/prediction)
    """
    augmentation = Sequential([
        layers.RandomRotation(0.1),  # Roter op til 10% (0.1 * 2π ≈ 36 grader)
        layers.RandomTranslation(  # Flyt billede op/ned og venstre/højre
            height_factor=0.1,
            width_factor=0.1
        ),
        layers.RandomZoom(0.1),  # Zoom ind/ud op til 10%
    ], name='data_augmentation')

    return augmentation


def create_model():
    """Opret CNN model MED augmentation layers inkluderet"""

    # Data augmentation layers
    augmentation = create_augmentation_layers()

    # Byg modellen
    model = Sequential([
        # Data augmentation (kun aktiv under træning!)
        augmentation,

        # CNN layers
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model


def train_model():
    """Træn modellen"""
    print("Indlæser data...")
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

    print("Opretter model med data augmentation...")
    model = create_model()
    model.summary()

    print("\nTræner model...")
    print("Note: Augmentation sker automatisk under træning!")

    history = model.fit(
        x_train,
        y_train,
        batch_size=128,
        epochs=15,
        validation_data=(x_test, y_test),
        verbose=1
    )

    print("\nEvaluerer model...")
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_acc * 100:.2f}%")

    print("Gemmer model...")
    model.save('data/mnist_model.keras')
    print("✅ Færdig!")

    return model, history


if __name__ == "__main__":
    np.random.seed(42)
    train_model()
