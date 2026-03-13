import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import (confusion_matrix, roc_curve, auc, 
                             precision_recall_curve, f1_score, 
                             classification_report, roc_auc_score)
import pandas as pd

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

def load_training_data():
    """Load training history and model"""
    history = joblib.load('ml/models/training_history.pkl')
    model = load_model('ml/models/lstm_model.keras')
    return history, model

def plot_training_history(history):
    """Plot accuracy and loss curves"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    epochs = range(1, len(history['loss']) + 1)
    
    # Accuracy
    axes[0].plot(epochs, history['accuracy'], 'b-', linewidth=2.5, label='Train Accuracy')
    axes[0].plot(epochs, history['val_accuracy'], 'r-', linewidth=2.5, label='Val Accuracy')
    axes[0].fill_between(epochs, history['accuracy'], history['val_accuracy'], alpha=0.2)
    axes[0].set_title('Model Accuracy Over Epochs', fontsize=16, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0.4, 1.0])
    
    # Loss
    axes[1].plot(epochs, history['loss'], 'g-', linewidth=2.5, label='Train Loss')
    axes[1].plot(epochs, history['val_loss'], 'orange', linewidth=2.5, label='Val Loss')
    axes[1].fill_between(epochs, history['loss'], history['val_loss'], alpha=0.2, color='yellow')
    axes[1].set_title('Model Loss Over Epochs', fontsize=16, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ml/graphs/1_training_history.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 1_training_history.png")

def plot_epoch_metrics(history):
    """Plot detailed metrics for each epoch"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    epochs = range(1, len(history['loss']) + 1)
    
    # Train vs Val Loss
    axes[0, 0].plot(epochs, history['loss'], 'b-', marker='o', linewidth=2, markersize=6, label='Train')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', marker='s', linewidth=2, markersize=6, label='Validation')
    axes[0, 0].set_title('Training & Validation Loss by Epoch', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Train vs Val Accuracy
    axes[0, 1].plot(epochs, history['accuracy'], 'g-', marker='o', linewidth=2, markersize=6, label='Train')
    axes[0, 1].plot(epochs, history['val_accuracy'], 'orange', marker='s', linewidth=2, markersize=6, label='Validation')
    axes[0, 1].set_title('Training & Validation Accuracy by Epoch', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Loss difference (overfitting indicator)
    loss_diff = np.array(history['val_loss']) - np.array(history['loss'])
    colors = ['green' if x < 0.05 else 'orange' if x < 0.1 else 'red' for x in loss_diff]
    axes[1, 0].bar(epochs, loss_diff, color=colors, alpha=0.7)
    axes[1, 0].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[1, 0].set_title('Overfitting Indicator (Val Loss - Train Loss)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss Difference')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Best epoch highlight
    best_epoch = np.argmin(history['val_loss']) + 1
    best_val_acc = history['val_accuracy'][best_epoch - 1]
    
    axes[1, 1].text(0.5, 0.7, f'Best Epoch: {best_epoch}', 
                   ha='center', fontsize=20, fontweight='bold', transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.5, 0.55, f'Validation Accuracy: {best_val_acc:.2%}', 
                   ha='center', fontsize=16, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.5, 0.40, f'Validation Loss: {history["val_loss"][best_epoch-1]:.4f}', 
                   ha='center', fontsize=16, transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.5, 0.25, f'Total Epochs: {len(history["loss"])}', 
                   ha='center', fontsize=14, transform=axes[1, 1].transAxes)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('ml/graphs/2_epoch_metrics.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 2_epoch_metrics.png")

def plot_convergence_analysis(history):
    """Analyze model convergence"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = range(1, len(history['loss']) + 1)
    
    # Convergence speed
    axes[0].plot(epochs, history['loss'], 'b-', linewidth=2, label='Train Loss')
    axes[0].axhline(y=min(history['val_loss']), color='r', linestyle='--', 
                   label=f'Best Val Loss ({min(history["val_loss"]):.4f})')
    axes[0].fill_between(epochs, history['loss'], min(history['val_loss']), alpha=0.2)
    axes[0].set_title('Loss Convergence', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plateau
    axes[1].plot(epochs, history['val_accuracy'], 'g-', linewidth=2.5, marker='o', markersize=5)
    best_acc_epoch = np.argmax(history['val_accuracy']) + 1
    axes[1].axvline(x=best_acc_epoch, color='r', linestyle='--', alpha=0.7, label=f'Peak: Epoch {best_acc_epoch}')
    axes[1].axhline(y=max(history['val_accuracy']), color='orange', linestyle='--', alpha=0.7)
    axes[1].set_title('Validation Accuracy Plateau', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning curve smoothness
    train_acc_diff = np.diff(history['accuracy'])
    axes[2].plot(epochs[1:], train_acc_diff, 'purple', linewidth=2, marker='s', markersize=5)
    axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[2].fill_between(epochs[1:], train_acc_diff, 0, alpha=0.3, color='purple')
    axes[2].set_title('Training Accuracy Change per Epoch', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy Difference')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ml/graphs/3_convergence_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 3_convergence_analysis.png")

def plot_performance_summary(history):
    """Summary statistics dashboard - Improved"""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.35)
    
    best_epoch = np.argmin(history['val_loss']) + 1
    best_val_acc = max(history['val_accuracy'])
    best_val_loss = min(history['val_loss'])
    
    # Title
    fig.suptitle('LSTM Model Performance Summary Dashboard', fontsize=18, fontweight='bold', y=0.98)
    
    # ===== TOP SECTION: Key Metrics =====
    ax_key = fig.add_subplot(gs[0, :2])
    ax_key.axis('off')
    
    # Best Epoch Box
    best_epoch_text = (
        f"BEST EPOCH: {best_epoch}\n"
        f"Val Accuracy: {best_val_acc:.2%}\n"
        f"Val Loss: {best_val_loss:.4f}"
    )
    ax_key.text(0.25, 0.5, best_epoch_text, fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round,pad=1', facecolor='#4CAF50', alpha=0.7, edgecolor='black', linewidth=2),
               ha='center', va='center', transform=ax_key.transAxes, color='white')
    
    # Final Performance Box
    final_text = (
        f"FINAL EPOCH ({len(history['loss'])})\n"
        f"Train Acc: {history['accuracy'][-1]:.2%} | Val Acc: {history['val_accuracy'][-1]:.2%}\n"
        f"Train Loss: {history['loss'][-1]:.4f} | Val Loss: {history['val_loss'][-1]:.4f}"
    )
    ax_key.text(0.75, 0.5, final_text, fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.9', facecolor='#2196F3', alpha=0.7, edgecolor='black', linewidth=2),
               ha='center', va='center', transform=ax_key.transAxes, color='white')
    
    # Improvement Box
    ax_improvement = fig.add_subplot(gs[0, 2:])
    ax_improvement.axis('off')
    
    acc_improvement = (history['accuracy'][-1] - history['accuracy'][0]) * 100
    val_acc_improvement = (history['val_accuracy'][-1] - history['val_accuracy'][0]) * 100
    loss_reduction = ((history['loss'][0] - history['loss'][-1]) / history['loss'][0]) * 100
    
    improvement_text = (
        f"TRAINING IMPROVEMENTS\n\n"
        f"Train Accuracy: +{acc_improvement:.1f}%\n"
        f"Val Accuracy: +{val_acc_improvement:.1f}%\n"
        f"Loss Reduction: {loss_reduction:.1f}%"
    )
    ax_improvement.text(0.5, 0.5, improvement_text, fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.9', facecolor='#FF9800', alpha=0.7, edgecolor='black', linewidth=2),
                       ha='center', va='center', transform=ax_improvement.transAxes, color='white')
    
    # ===== MIDDLE SECTION: Core Metrics Bar Charts =====
    ax1 = fig.add_subplot(gs[1, :2])
    metrics_best = {
        'Train\nLoss': history['loss'][best_epoch-1],
        'Val\nLoss': history['val_loss'][best_epoch-1],
    }
    colors1 = ['#FF6B6B', '#FF8C42']
    bars1 = ax1.bar(metrics_best.keys(), metrics_best.values(), color=colors1, alpha=0.8, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax1.set_title(f'Loss at Best Epoch ({best_epoch})', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax2 = fig.add_subplot(gs[1, 2:])
    metrics_acc = {
        'Train\nAccuracy': history['accuracy'][best_epoch-1],
        'Val\nAccuracy': history['val_accuracy'][best_epoch-1],
    }
    colors2 = ['#4CAF50', '#8BC34A']
    bars2 = ax2.bar(metrics_acc.keys(), metrics_acc.values(), color=colors2, alpha=0.8, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax2.set_title(f'Accuracy at Best Epoch ({best_epoch})', fontsize=13, fontweight='bold')
    ax2.set_ylim([min(history['accuracy']) - 0.05, 1.05])
    ax2.grid(axis='y', alpha=0.3)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2%}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # ===== STATISTICS TABLE =====
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis('off')
    
    stats = [
        ['Metric', 'Best Epoch Value', 'Final Epoch Value', 'Overall Best'],
        ['Loss', f"{history['loss'][best_epoch-1]:.4f}", f"{history['loss'][-1]:.4f}", f"{min(history['loss']):.4f}"],
        ['Val Loss', f"{history['val_loss'][best_epoch-1]:.4f}", f"{history['val_loss'][-1]:.4f}", f"{min(history['val_loss']):.4f}"],
        ['Accuracy', f"{history['accuracy'][best_epoch-1]:.2%}", f"{history['accuracy'][-1]:.2%}", f"{max(history['accuracy']):.2%}"],
        ['Val Accuracy', f"{history['val_accuracy'][best_epoch-1]:.2%}", f"{history['val_accuracy'][-1]:.2%}", f"{max(history['val_accuracy']):.2%}"],
        ['Epochs', '', str(len(history['loss'])), ''],
    ]
    
    table = ax_table.table(cellText=stats, cellLoc='center', loc='center',
                          colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.8)
    
    # Header styling
    for i in range(4):
        table[(0, i)].set_facecolor('#1976D2')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=11)
    
    # Content styling
    for i in range(1, len(stats)):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E3F2FD')
            else:
                table[(i, j)].set_facecolor('#BBDEFB')
            table[(i, j)].set_text_props(fontsize=10)
    
    # ===== BOTTOM SECTION: Distribution Plots =====
    ax3 = fig.add_subplot(gs[3, 0])
    ax3.hist(history['loss'], bins=12, color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.axvline(history['loss'][best_epoch-1], color='darkred', linestyle='--', linewidth=2.5, label='Best Epoch')
    ax3.set_title('Train Loss Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Loss', fontsize=10)
    ax3.set_ylabel('Frequency', fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)
    
    ax4 = fig.add_subplot(gs[3, 1])
    ax4.hist(history['val_loss'], bins=12, color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.axvline(history['val_loss'][best_epoch-1], color='darkblue', linestyle='--', linewidth=2.5, label='Best Epoch')
    ax4.set_title('Val Loss Distribution', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Loss', fontsize=10)
    ax4.set_ylabel('Frequency', fontsize=10)
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)
    
    ax5 = fig.add_subplot(gs[3, 2])
    ax5.hist(history['accuracy'], bins=12, color='#95E1D3', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax5.axvline(history['accuracy'][best_epoch-1], color='darkgreen', linestyle='--', linewidth=2.5, label='Best Epoch')
    ax5.set_title('Train Accuracy Distribution', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Accuracy', fontsize=10)
    ax5.set_ylabel('Frequency', fontsize=10)
    ax5.legend(fontsize=9)
    ax5.grid(alpha=0.3)
    
    ax6 = fig.add_subplot(gs[3, 3])
    ax6.hist(history['val_accuracy'], bins=12, color='#FFE082', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax6.axvline(history['val_accuracy'][best_epoch-1], color='darkgoldenrod', linestyle='--', linewidth=2.5, label='Best Epoch')
    ax6.set_title('Val Accuracy Distribution', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Accuracy', fontsize=10)
    ax6.set_ylabel('Frequency', fontsize=10)
    ax6.legend(fontsize=9)
    ax6.grid(alpha=0.3)
    
    plt.savefig('ml/graphs/4_performance_summary.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 4_performance_summary.png")

def plot_comparison_dashboard(history):
    """Compare train vs validation metrics side by side"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    epochs = range(1, len(history['loss']) + 1)
    
    # Loss comparison
    axes[0, 0].plot(epochs, history['loss'], 'b-', linewidth=3, label='Train Loss', marker='o', markersize=4)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', linewidth=3, label='Val Loss', marker='s', markersize=4)
    axes[0, 0].fill_between(epochs, history['loss'], history['val_loss'], alpha=0.2, color='gray')
    axes[0, 0].set_title('Loss Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy comparison
    axes[0, 1].plot(epochs, history['accuracy'], 'g-', linewidth=3, label='Train Accuracy', marker='o', markersize=4)
    axes[0, 1].plot(epochs, history['val_accuracy'], 'orange', linewidth=3, label='Val Accuracy', marker='s', markersize=4)
    axes[0, 1].fill_between(epochs, history['accuracy'], history['val_accuracy'], alpha=0.2, color='yellow')
    axes[0, 1].set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Accuracy', fontsize=12)
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Loss ratio
    loss_ratio = np.array(history['val_loss']) / (np.array(history['loss']) + 1e-6)
    axes[1, 0].plot(epochs, loss_ratio, 'purple', linewidth=3, marker='D', markersize=5)
    axes[1, 0].axhline(y=1, color='black', linestyle='--', linewidth=2, alpha=0.7, label='1:1 Ratio')
    axes[1, 0].fill_between(epochs, loss_ratio, 1, alpha=0.2, color='purple')
    axes[1, 0].set_title('Val/Train Loss Ratio (Lower = Better)', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Ratio', fontsize=12)
    axes[1, 0].set_ylim([0.5, 2.0])
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Improvement metrics
    acc_improvement = np.array(history['accuracy']) - history['accuracy'][0]
    val_acc_improvement = np.array(history['val_accuracy']) - history['val_accuracy'][0]
    
    axes[1, 1].plot(epochs, acc_improvement * 100, 'b-', linewidth=3, label='Train Improvement', marker='o', markersize=4)
    axes[1, 1].plot(epochs, val_acc_improvement * 100, 'r-', linewidth=3, label='Val Improvement', marker='s', markersize=4)
    axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1, 1].fill_between(epochs, acc_improvement * 100, val_acc_improvement * 100, alpha=0.2, color='cyan')
    axes[1, 1].set_title('Accuracy Improvement from Epoch 1 (%)', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Improvement (%)', fontsize=12)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add epoch labels
    for ax in axes.flat:
        ax.set_xlabel('Epoch', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('ml/graphs/5_comparison_dashboard.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 5_comparison_dashboard.png")

if __name__ == "__main__":
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE TRAINING VISUALIZATIONS")
    print("="*80 + "\n")
    
    # Load data
    history, model = load_training_data()
    
    # Generate all graphs
    plot_training_history(history)
    plot_epoch_metrics(history)
    plot_convergence_analysis(history)
    plot_performance_summary(history)
    plot_comparison_dashboard(history)
    
    print("\n" + "="*80)
    print("✓ ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
    print("="*80)
    print("\nGenerated files:")
    print("  1. 1_training_history.png - Basic accuracy and loss curves")
    print("  2. 2_epoch_metrics.png - Detailed epoch-by-epoch analysis")
    print("  3. 3_convergence_analysis.png - Convergence speed analysis")
    print("  4. 4_performance_summary.png - Statistics dashboard")
    print("  5. 5_comparison_dashboard.png - Train vs Validation comparison")
    print("\n")
