import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
import os
from datetime import datetime


class ModelEvaluator:
    """
    Comprehensive model evaluation with metrics and visualizations
    """

    def __init__(self, model, device, tag_names, save_dir='outputs/metrics'):
        self.model = model
        self.device = device
        self.tag_names = tag_names
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Results storage
        self.y_true = []
        self.y_pred = []
        self.confidences = []
        self.predictions_per_class = {tag: [] for tag in tag_names}

    def evaluate_batch(self, data_loader):
        """
        Evaluate model on a data loader

        Args:
            data_loader: PyTorch DataLoader
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        self.model.eval()

        y_true_batch = []
        y_pred_batch = []
        confidences_batch = []

        with torch.no_grad():
            for inputs, labels, lengths in data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                lengths = lengths.to(self.device)

                # Forward pass
                if hasattr(self.model, 'use_attention') and self.model.use_attention:
                    outputs, _ = self.model(inputs, lengths)
                else:
                    outputs = self.model(inputs, lengths)

                # Get predictions
                probs = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probs, dim=1)

                # Store results
                y_true_batch.extend(labels.cpu().numpy())
                y_pred_batch.extend(predicted.cpu().numpy())
                confidences_batch.extend(confidence.cpu().numpy())

        # Update global results
        self.y_true.extend(y_true_batch)
        self.y_pred.extend(y_pred_batch)
        self.confidences.extend(confidences_batch)

        # Calculate metrics
        metrics = self.calculate_metrics(y_true_batch, y_pred_batch)

        return metrics

    def calculate_metrics(self, y_true, y_pred):
        """
        Calculate classification metrics

        Args:
            y_true: True labels
            y_pred: Predicted labels
        Returns:
            metrics: Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }

        return metrics

    def print_metrics(self, metrics, dataset_name='Validation'):
        """Print formatted metrics"""
        print(f"\n{'=' * 60}")
        print(f"{dataset_name} Set Metrics")
        print(f"{'=' * 60}")
        print(f"Accuracy:           {metrics['accuracy']:.4f}")
        print(f"Precision (macro):  {metrics['precision_macro']:.4f}")
        print(f"Recall (macro):     {metrics['recall_macro']:.4f}")
        print(f"F1 Score (macro):   {metrics['f1_macro']:.4f}")
        print(f"F1 Score (weighted): {metrics['f1_weighted']:.4f}")
        print(f"{'=' * 60}\n")

    def plot_confusion_matrix(self, normalize=False, save_name='confusion_matrix.png'):
        """
        Plot confusion matrix

        Args:
            normalize: Normalize values
            save_name: Filename to save plot
        """
        # Get unique labels that actually appear
        unique_labels = np.unique(np.concatenate([self.y_true, self.y_pred]))

        cm = confusion_matrix(self.y_true, self.y_pred, labels=unique_labels)

        if normalize:
            # Avoid division by zero
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            cm = cm.astype('float') / row_sums
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'

        # Use only tag names for classes that appear
        tag_labels = [self.tag_names[i] for i in unique_labels]

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=tag_labels,
            yticklabels=tag_labels,
            cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
        )
        plt.title(title, fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
        plt.close()

    def plot_per_class_metrics(self, save_name='per_class_metrics.png'):
        """
        Plot precision, recall, F1 for each class
        """
        # Get unique labels present in predictions
        unique_labels = np.unique(np.concatenate([self.y_true, self.y_pred]))

        report = classification_report(
            self.y_true, self.y_pred,
            labels=unique_labels,
            target_names=[self.tag_names[i] for i in unique_labels],
            output_dict=True,
            zero_division=0
        )

        # Extract per-class metrics (only for classes that exist)
        classes = [self.tag_names[i] for i in unique_labels]
        precision = [report[c]['precision'] for c in classes]
        recall = [report[c]['recall'] for c in classes]
        f1 = [report[c]['f1-score'] for c in classes]

        # Plot
        x = np.arange(len(classes))
        width = 0.25

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)

        ax.set_xlabel('Intent Class', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Metrics', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Per-class metrics saved to {save_path}")
        plt.close()

    def plot_confidence_distribution(self, save_name='confidence_distribution.png'):
        """
        Plot distribution of prediction confidences
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Overall confidence distribution
        ax1.hist(self.confidences, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(np.mean(self.confidences), color='red', linestyle='--',
                    label=f'Mean: {np.mean(self.confidences):.3f}')
        ax1.set_xlabel('Confidence Score', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Confidence by correctness
        correct_mask = np.array(self.y_true) == np.array(self.y_pred)
        correct_conf = np.array(self.confidences)[correct_mask]
        incorrect_conf = np.array(self.confidences)[~correct_mask]

        ax2.hist(correct_conf, bins=30, alpha=0.6, color='green',
                 label=f'Correct ({len(correct_conf)})', edgecolor='black')
        ax2.hist(incorrect_conf, bins=30, alpha=0.6, color='red',
                 label=f'Incorrect ({len(incorrect_conf)})', edgecolor='black')
        ax2.set_xlabel('Confidence Score', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Confidence by Prediction Correctness', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confidence distribution saved to {save_path}")
        plt.close()

    def generate_classification_report(self, save_name='classification_report.txt'):
        """
        Generate and save detailed classification report
        """
        # Get unique labels present in predictions
        unique_labels = np.unique(np.concatenate([self.y_true, self.y_pred]))

        report = classification_report(
            self.y_true, self.y_pred,
            labels=unique_labels,
            target_names=[self.tag_names[i] for i in unique_labels],
            zero_division=0
        )

        save_path = os.path.join(self.save_dir, save_name)
        with open(save_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("CLASSIFICATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(report)
            f.write("\n\n" + "=" * 60 + "\n")
            f.write(f"Average Confidence: {np.mean(self.confidences):.4f}\n")
            f.write(f"Total Samples: {len(self.y_true)}\n")
            f.write(f"Classes in test set: {len(unique_labels)}/{len(self.tag_names)}\n")
            f.write("=" * 60 + "\n")

        print(f"Classification report saved to {save_path}")
        print(report)

    def generate_full_report(self):
        """
        Generate all evaluation plots and reports
        """
        print("\nGenerating evaluation reports...")

        self.plot_confusion_matrix(normalize=False)
        self.plot_confusion_matrix(normalize=True, save_name='confusion_matrix_normalized.png')
        self.plot_per_class_metrics()
        self.plot_confidence_distribution()
        self.generate_classification_report()

        print("\nAll evaluation reports generated successfully!")


class TrainingMonitor:
    """
    Monitor and visualize training progress
    """

    def __init__(self, save_dir='outputs/metrics'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.epochs = []

    def update(self, epoch, train_loss, val_loss, train_acc, val_acc):
        """
        Update training metrics
        """
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)

    def plot_training_history(self, save_name='training_history.png'):
        """
        Plot training and validation metrics
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss plot
        ax1.plot(self.epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        ax1.plot(self.epochs, self.val_losses, 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Accuracy plot
        ax2.plot(self.epochs, self.train_accuracies, 'b-', label='Train Accuracy', linewidth=2)
        ax2.plot(self.epochs, self.val_accuracies, 'r-', label='Val Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
        plt.close()


def split_data(X, y, lengths, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    """
    Split data into train/val/test sets with automatic stratify detection

    Args:
        X: Input features
        y: Labels
        lengths: Sequence lengths
        train_size: Training set proportion
        val_size: Validation set proportion
        test_size: Test set proportion
        random_state: Random seed
    Returns:
        Train, validation, test splits
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Splits must sum to 1.0"

    # Check class distribution
    unique, counts = np.unique(y, return_counts=True)
    min_samples = counts.min()

    # Disable stratify if any class has fewer than 2 samples
    use_stratify = min_samples >= 2

    if not use_stratify:
        print(f"Warning: Some classes have only {min_samples} sample(s).")
        print(f"    Using random split instead of stratified split.")
        stratify_first = None
    else:
        stratify_first = y

    # First split: train vs (val + test)
    X_train, X_temp, y_train, y_temp, len_train, len_temp = train_test_split(
        X, y, lengths,
        test_size=(val_size + test_size),
        random_state=random_state,
        shuffle=True,
        stratify=stratify_first
    )

    # Second split: val vs test
    # Recalculate stratify for second split
    if use_stratify:
        unique_temp, counts_temp = np.unique(y_temp, return_counts=True)
        use_stratify_temp = counts_temp.min() >= 2
        stratify_second = y_temp if use_stratify_temp else None
    else:
        stratify_second = None

    val_ratio = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test, len_val, len_test = train_test_split(
        X_temp, y_temp, len_temp,
        test_size=(1 - val_ratio),
        random_state=random_state,
        shuffle=True,
        stratify=stratify_second
    )

    print(f"\nData split completed:")
    print(f"  Train: {len(X_train)} samples ({train_size * 100:.1f}%)")
    print(f"  Val:   {len(X_val)} samples ({val_size * 100:.1f}%)")
    print(f"  Test:  {len(X_test)} samples ({test_size * 100:.1f}%)")

    return (X_train, y_train, len_train), (X_val, y_val, len_val), (X_test, y_test, len_test)


def evaluate_model(model, data_loader, device):
    """
    Quick evaluation function

    Args:
        model: PyTorch model
        data_loader: DataLoader
        device: torch.device
    Returns:
        accuracy: Float accuracy
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels, lengths in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)

            if hasattr(model, 'use_attention') and model.use_attention:
                outputs, _ = model(inputs, lengths)
            else:
                outputs = model(inputs, lengths)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy
