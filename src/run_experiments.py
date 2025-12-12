# run_experiments.py
import numpy as np
from model_training import create_model, load_data
import json
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import timedelta


def run_single_experiment(config):
    """Kør ét eksperiment med given konfiguration"""
    print(f"\n🚀 Running: {config['name']}")

    # Start total tid
    total_start = time.time()

    # Load data
    data_start = time.time()
    (x_train, y_train), (x_test, y_test) = load_data()
    data_load_time = time.time() - data_start

    # Create model with config
    model_start = time.time()
    model = create_model(
        augmentation=config['aug_strength'],
        dropout=config['dropout'],
        learning_rate=config['learning_rate']
    )
    model_creation_time = time.time() - model_start

    # Train med tidstagning
    training_start = time.time()
    history = model.fit(
        x_train, y_train,
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        validation_data=(x_test, y_test),
        verbose=1
    )
    training_time = time.time() - training_start

    #Evaluate med tidstagning
    eval_start = time.time()
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    eval_time = time.time() - eval_start

    # Predictions
    pred_start = time.time()
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)
    prediction_time = time.time() - pred_start

    # Beregn gennemsnitstid per epoch
    time_per_epoch = training_time / config['epochs']

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

    # ✅ Total tid
    total_time = time.time() - total_start

    # ✅ Save results med timing information
    results = {
        'config': config,
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'timing': {  # ✅ NY SEKTION
            'total_time_seconds': round(total_time, 2),
            'data_load_time_seconds': round(data_load_time, 2),
            'model_creation_time_seconds': round(model_creation_time, 2),
            'training_time_seconds': round(training_time, 2),
            'training_time_formatted': str(timedelta(seconds=int(training_time))),
            'time_per_epoch_seconds': round(time_per_epoch, 2),
            'eval_time_seconds': round(eval_time, 2),
            'prediction_time_seconds': round(prediction_time, 2)
        },
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

    # ✅ Print detaljeret timing
    print(f"✅ {config['name']}:")
    print(f"   📊 Test Accuracy: {test_acc * 100:.2f}%")
    print(f"   ⏱️  Training Time: {training_time:.2f}s ({str(timedelta(seconds=int(training_time)))})")
    print(f"   ⏱️  Time per Epoch: {time_per_epoch:.2f}s")
    print(f"   ⏱️  Total Time: {total_time:.2f}s")

    return results


# Define experiments
experiments = [
    {'name': 'baseline', 'aug_strength': 0, 'learning_rate': 0, 'dropout': 0.5, 'epochs': 5, 'batch_size': 128},

]

if __name__ == "__main__":
    import os

    os.makedirs('experiments', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # ✅ Track alle eksperimenter
    all_results = []
    experiment_start = time.time()

    for config in experiments:
        result = run_single_experiment(config)
        all_results.append(result)

    total_experiment_time = time.time() - experiment_start

    # ✅ Sammenfatning af alle eksperimenter
    print("\n" + "=" * 60)
    print("📈 EXPERIMENT SUMMARY")
    print("=" * 60)

    for result in all_results:
        name = result['config']['name']
        acc = result['test_accuracy'] * 100
        train_time = result['timing']['training_time_seconds']
        time_per_epoch = result['timing']['time_per_epoch_seconds']

        print(f"{name:20} | Acc: {acc:5.2f}% | Train: {train_time:6.1f}s | Per Epoch: {time_per_epoch:5.2f}s")

    print(f"\n⏱️  Total Experiment Time: {str(timedelta(seconds=int(total_experiment_time)))}")
    print("=" * 60)

    # ✅ Gem samlet oversigt
    summary = {
        'total_experiment_time_seconds': round(total_experiment_time, 2),
        'experiments': all_results
    }

    with open('experiments/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
