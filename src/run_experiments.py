# run_experiments.py
import numpy as np
from model_training import create_model, load_data
import json
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def run_single_experiment(config):
    """Kør ét eksperiment med given konfiguration"""
    print(f"\n🚀 Running: {config['name']}")

    # Load data
    (x_train, y_train), (x_test, y_test) = load_data()

    # Create model with config
    model = create_model(
        augmentation=config['aug_strength'],
        dropout=config['dropout']
    )

    # Train
    history = model.fit(
        x_train, y_train,
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        validation_data=(x_test, y_test),
        verbose=1
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    # Predictions
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {config["name"]}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'results/cm_{config["name"]}.png')
    plt.close()

    # Classification Report
    report = classification_report(y_true, y_pred, output_dict=True)

    # Save results
    results = {
        'config': config,
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'history': {
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']]
        },
        'classification_report': report
    }

    with open(f'experiments/exp_{config["name"]}.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save model
    model.save(f'experiments/model_{config["name"]}.keras')

    print(f"✅ {config['name']}: Test Accuracy = {test_acc * 100:.2f}%")
    return results


# Define experiments
experiments = [
    {'name': 'baseline_no_aug', 'aug_strength': 0.0, 'dropout': 0.5, 'epochs': 15, 'batch_size': 128},
    {'name': 'low_aug', 'aug_strength': 0.1, 'dropout': 0.5, 'epochs': 15, 'batch_size': 128},
    {'name': 'medium_aug', 'aug_strength': 0.2, 'dropout': 0.5, 'epochs': 15, 'batch_size': 128},
    {'name': 'high_aug', 'aug_strength': 0.3, 'dropout': 0.5, 'epochs': 15, 'batch_size': 128},
]

if __name__ == "__main__":
    import os

    os.makedirs('experiments', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    for config in experiments:
        run_single_experiment(config)