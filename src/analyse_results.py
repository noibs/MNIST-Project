"""
analyze_results.py
Analyserer og visualiserer resultater fra alle eksperimenter
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Sæt pæn plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['font.size'] = 11


def load_all_experiments(experiments_dir='experiments'):
    """Indlæs alle eksperiment-resultater fra JSON filer"""
    experiments = []

    for file in sorted(Path(experiments_dir).glob('exp_*.json')):
        with open(file, 'r') as f:
            data = json.load(f)
            experiments.append(data)

    print(f"✅ Loaded {len(experiments)} experiments")
    return experiments


def create_comparison_table(experiments):
    """Lav sammenligningstabel af alle eksperimenter"""
    print("\n" + "=" * 80)
    print("📊 EXPERIMENT COMPARISON TABLE")
    print("=" * 80)

    # Header
    print(f"{'Experiment':<25} {'Aug':>6} {'Drop':>6} {'Epochs':>7} {'Train Acc':>10} {'Val Acc':>10} {'Test Acc':>10}")
    print("-" * 80)

    # Data
    for exp in experiments:
        name = exp['config']['name']
        aug = exp['config']['aug_strength']
        dropout = exp['config']['dropout']
        epochs = exp['config']['epochs']
        train_acc = exp['history']['accuracy'][-1] * 100
        val_acc = exp['history']['val_accuracy'][-1] * 100
        test_acc = exp['test_accuracy'] * 100

        print(
            f"{name:<25} {aug:>6.1f} {dropout:>6.2f} {epochs:>7d} {train_acc:>9.2f}% {val_acc:>9.2f}% {test_acc:>9.2f}%")

    print("=" * 80 + "\n")


def plot_accuracy_comparison(experiments, save_dir='results'):
    """Plot sammenligning af accuracy for alle eksperimenter"""
    plt.figure(figsize=(7, 7))

    for exp in experiments:
        name = exp['config']['name']
        epochs = range(1, len(exp['history']['accuracy']) + 1)
        val_acc = [x * 100 for x in exp['history']['val_accuracy']]

        plt.plot(epochs, val_acc, marker='o', label=name, linewidth=2)

    plt.title('Validation Accuracy Comparison - All Experiments', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=13)
    plt.ylabel('Validation Accuracy (%)', fontsize=13)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(f'{save_dir}/accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_dir}/accuracy_comparison.png")
    plt.close()


def plot_loss_comparison(experiments, save_dir='results'):
    """Plot sammenligning af loss for alle eksperimenter"""
    plt.figure(figsize=(8, 8))

    for exp in experiments:
        name = exp['config']['name']
        epochs = range(1, len(exp['history']['loss']) + 1)
        val_loss = exp['history']['val_loss']

        plt.plot(epochs, val_loss, marker='o', label=name, linewidth=2)

    plt.title('Validation Loss Comparison - All Experiments', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=13)
    plt.ylabel('Validation Loss', fontsize=13)
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(f'{save_dir}/loss_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_dir}/loss_comparison.png")
    plt.close()


def plot_train_vs_val(experiments, save_dir='results'):
    """Plot training vs validation accuracy (overfitting check)"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, exp in enumerate(experiments[:4]):  # Max 4 plots
        name = exp['config']['name']
        epochs = range(1, len(exp['history']['accuracy']) + 1)
        train_acc = [x * 100 for x in exp['history']['accuracy']]
        val_acc = [x * 100 for x in exp['history']['val_accuracy']]

        ax = axes[idx]
        ax.plot(epochs, train_acc, marker='o', label='Training', linewidth=2)
        ax.plot(epochs, val_acc, marker='s', label='Validation', linewidth=2)
        ax.set_title(f'{name}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Accuracy (%)', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Calculate overfitting gap
        gap = train_acc[-1] - val_acc[-1]
        ax.text(0.02, 0.98, f'Gap: {gap:.2f}%',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'{save_dir}/train_vs_val_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_dir}/train_vs_val_comparison.png")
    plt.close()


def plot_final_metrics_bars(experiments, save_dir='results'):
    """Bar plot af finale test accuracies"""
    names = [exp['config']['name'] for exp in experiments]
    accuracies = [exp['test_accuracy'] * 100 for exp in experiments]

    plt.figure(figsize=(7, 7))
    bars = plt.bar(range(len(names)), accuracies, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'][:len(names)])

    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                 f'{acc:.2f}%',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.xlabel('Experiment', fontsize=13)
    plt.ylabel('Test Accuracy (%)', fontsize=13)
    plt.title('Final Test Accuracy Comparison', fontsize=16, fontweight='bold')
    plt.xticks(range(len(names)), names, rotation=45, ha='right')
    plt.ylim(min(accuracies) - 1, max(accuracies) + 2)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    plt.savefig(f'{save_dir}/final_accuracy_bars.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_dir}/final_accuracy_bars.png")
    plt.close()


def analyze_per_class_performance(experiments, save_dir='results'):
    """Analysér per-class performance"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, exp in enumerate(experiments[:4]):
        name = exp['config']['name']
        report = exp['classification_report']

        # Extract per-class metrics
        classes = [str(i) for i in range(10)]
        precision = [report[c]['precision'] * 100 for c in classes]
        recall = [report[c]['recall'] * 100 for c in classes]
        f1 = [report[c]['f1-score'] * 100 for c in classes]

        ax = axes[idx]
        x = np.arange(len(classes))
        width = 0.25

        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)

        ax.set_xlabel('Digit Class', fontsize=11)
        ax.set_ylabel('Score (%)', fontsize=11)
        ax.set_title(f'{name} - Per-Class Metrics', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend(fontsize=9)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(90, 100)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/per_class_metrics.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_dir}/per_class_metrics.png")
    plt.close()


def find_worst_performing_classes(experiments):
    """Find hvilke cifre der er sværest at genkende"""
    print("\n" + "=" * 80)
    print("🔍 WORST PERFORMING CLASSES (Lowest F1-Scores)")
    print("=" * 80)

    for exp in experiments:
        name = exp['config']['name']
        report = exp['classification_report']

        # Extract F1-scores per class
        class_f1 = []
        for digit in range(10):
            f1 = report[str(digit)]['f1-score'] * 100
            class_f1.append((digit, f1))

        # Sort by F1-score
        class_f1.sort(key=lambda x: x[1])

        print(f"\n{name}:")
        print("  Worst 3 classes:")
        for digit, f1 in class_f1[:3]:
            print(f"    Digit {digit}: F1={f1:.2f}%")

    print("=" * 80 + "\n")


def plot_augmentation_effect(experiments, save_dir='results'):
    """Visualiser effekten af forskellige augmentation levels"""

    # Filter eksperimenter med samme dropout (0.5) men forskellig augmentation
    aug_experiments = [exp for exp in experiments if exp['config']['dropout'] == 0.5]

    if len(aug_experiments) < 2:
        print("⚠️  Not enough experiments to compare augmentation effect")
        return

    aug_levels = [exp['config']['aug_strength'] for exp in aug_experiments]
    test_accs = [exp['test_accuracy'] * 100 for exp in aug_experiments]
    train_accs = [exp['history']['accuracy'][-1] * 100 for exp in aug_experiments]
    overfitting_gaps = [train - test for train, test in zip(train_accs, test_accs)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Accuracy vs Augmentation
    ax1.plot(aug_levels, train_accs, marker='o', linewidth=2, markersize=8, label='Training Acc')
    ax1.plot(aug_levels, test_accs, marker='s', linewidth=2, markersize=8, label='Test Acc')
    ax1.set_xlabel('Augmentation Strength', fontsize=13)
    ax1.set_ylabel('Accuracy (%)', fontsize=13)
    ax1.set_title('Effect of Data Augmentation on Accuracy', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Overfitting Gap vs Augmentation
    ax2.plot(aug_levels, overfitting_gaps, marker='o', linewidth=2, markersize=8, color='red')
    ax2.set_xlabel('Augmentation Strength', fontsize=13)
    ax2.set_ylabel('Overfitting Gap (Train - Test) %', fontsize=13)
    ax2.set_title('Effect of Augmentation on Overfitting', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/augmentation_effect.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_dir}/augmentation_effect.png")
    plt.close()


def generate_summary_report(experiments, save_dir='results'):
    """Generer tekstbaseret sammenfatning"""
    report_path = f'{save_dir}/summary_report.txt'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("MNIST EXPERIMENT SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")

        # Best model
        best_exp = max(experiments, key=lambda x: x['test_accuracy'])
        f.write(f"🏆 BEST MODEL: {best_exp['config']['name']}\n")
        f.write(f"   Test Accuracy: {best_exp['test_accuracy'] * 100:.2f}%\n")
        f.write(
            f"   Configuration: aug={best_exp['config']['aug_strength']}, dropout={best_exp['config']['dropout']}\n\n")

        # All results
        f.write("📊 ALL RESULTS:\n")
        f.write("-" * 80 + "\n")
        for exp in sorted(experiments, key=lambda x: x['test_accuracy'], reverse=True):
            name = exp['config']['name']
            acc = exp['test_accuracy'] * 100
            f.write(f"  {name:<25} {acc:.2f}%\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("=" * 80 + "\n\n")

        for exp in experiments:
            f.write(f"\n{'=' * 60}\n")
            f.write(f"Experiment: {exp['config']['name']}\n")
            f.write(f"{'=' * 60}\n")
            f.write(f"Configuration:\n")
            f.write(f"  - Augmentation strength: {exp['config']['aug_strength']}\n")
            f.write(f"  - Dropout: {exp['config']['dropout']}\n")
            f.write(f"  - Epochs: {exp['config']['epochs']}\n")
            f.write(f"  - Batch size: {exp['config']['batch_size']}\n\n")

            f.write(f"Results:\n")
            f.write(f"  - Final Training Accuracy: {exp['history']['accuracy'][-1] * 100:.2f}%\n")
            f.write(f"  - Final Validation Accuracy: {exp['history']['val_accuracy'][-1] * 100:.2f}%\n")
            f.write(f"  - Test Accuracy: {exp['test_accuracy'] * 100:.2f}%\n")
            f.write(f"  - Test Loss: {exp['test_loss']:.4f}\n")
            f.write(f"  - Overfitting Gap: {(exp['history']['accuracy'][-1] - exp['test_accuracy']) * 100:.2f}%\n\n")

            # Worst performing classes
            report = exp['classification_report']
            class_f1 = [(i, report[str(i)]['f1-score'] * 100) for i in range(10)]
            class_f1.sort(key=lambda x: x[1])
            f.write(f"  Worst 3 classes (by F1-score):\n")
            for digit, f1 in class_f1[:3]:
                f.write(f"    - Digit {digit}: {f1:.2f}%\n")

    print(f"✅ Saved: {report_path}")


def main():
    """Main analysis function"""
    print("\n" + "=" * 80)
    print("🔬 MNIST EXPERIMENT ANALYSIS")
    print("=" * 80 + "\n")

    # Load experiments
    experiments = load_all_experiments()

    if not experiments:
        print("❌ No experiments found! Run experiments first.")
        return

    # Create results directory
    os.makedirs('results', exist_ok=True)

    # Run all analyses
    print("\n📊 Generating analyses...")

    create_comparison_table(experiments)
    plot_accuracy_comparison(experiments)
    plot_loss_comparison(experiments)
    plot_train_vs_val(experiments)
    plot_final_metrics_bars(experiments)
    analyze_per_class_performance(experiments)
    find_worst_performing_classes(experiments)
    plot_augmentation_effect(experiments)
    generate_summary_report(experiments)

    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  📁 results/")
    print("    📊 accuracy_comparison.png")
    print("    📊 loss_comparison.png")
    print("    📊 train_vs_val_comparison.png")
    print("    📊 final_accuracy_bars.png")
    print("    📊 per_class_metrics.png")
    print("    📊 augmentation_effect.png")
    print("    📄 summary_report.txt")
    print("\n💡 Tip: Use these plots in your SOP report!\n")


if __name__ == "__main__":
    main()
