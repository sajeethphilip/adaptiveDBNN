import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
import json
from collections import defaultdict
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import copy
import glob

from dbnn_optimised import GPUDBNN  # Import the original DBNN class

class AdaptiveDBNN:
    """Wrapper for DBNN that implements adaptive learning with enhanced statistics"""

    def __init__(self, dataset_name: str, config: Dict = None):
        self.dataset_name = dataset_name
        self.config = config or self._load_config(dataset_name)
        self.adaptive_config = self.config.get('adaptive_learning', {
            "enable_adaptive": True,
            "initial_samples_per_class": 10,
            "margin": 0.1,
            "max_adaptive_rounds": 10
        })
        self.stats_config = self.config.get('statistics', {
            'enable_confusion_matrix': True,
            'enable_progress_plots': True,
            'color_progress': 'green',
            'color_regression': 'red',
            'save_plots': True
        })

        # Initialize the base DBNN model
        self.model = GPUDBNN(dataset_name)

        # Adaptive learning state
        self.training_indices = []  # Indices of samples in training set
        self.test_indices = []      # Indices of samples in test set
        self.best_accuracy = 0.0
        self.best_training_indices = []
        self.adaptive_round = 0

        # Statistics tracking
        self.round_stats = []
        self.previous_confusion = None
        self.start_time = datetime.now()

        # Store the full dataset for adaptive learning
        self.X_full = None
        self.y_full = None

        # Update config file with default settings if they don't exist
        self._update_config_file()

        # Show current settings
        self.show_adaptive_settings()

    def show_adaptive_settings(self):
        """Display the current adaptive learning settings"""
        print("\n🔧 Current Adaptive Learning Settings:")
        print("=" * 50)
        for key, value in self.adaptive_config.items():
            print(f"  {key:30}: {value}")

        print("\n📊 Current Statistics Settings:")
        print("=" * 50)
        for key, value in self.stats_config.items():
            print(f"  {key:30}: {value}")
        print()

    def configure_adaptive_learning(self):
        """Interactively configure adaptive learning settings"""
        print("\n🎛️  Configure Adaptive Learning Settings")
        print("=" * 50)

        # Get new values from user
        try:
            initial_samples = int(input(f"Initial samples per class [{self.adaptive_config['initial_samples_per_class']}]: ")
                                or self.adaptive_config['initial_samples_per_class'])
            margin = float(input(f"Margin threshold [{self.adaptive_config['margin']}]: ")
                          or self.adaptive_config['margin'])
            max_rounds = int(input(f"Maximum adaptive rounds [{self.adaptive_config['max_adaptive_rounds']}]: ")
                            or self.adaptive_config['max_adaptive_rounds'])

            # Update settings
            self.adaptive_config.update({
                'initial_samples_per_class': initial_samples,
                'margin': margin,
                'max_adaptive_rounds': max_rounds
            })

            # Update config file
            self._update_config_file()

            print("✅ Settings updated successfully!")
            self.show_adaptive_settings()

        except ValueError:
            print("❌ Invalid input. Settings not changed.")

    def _update_config_file(self):
        """Update the dataset configuration file with adaptive learning settings"""
        config_path = f"{self.dataset_name}.conf"

        try:
            # Load existing config
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
            else:
                config = {}

            # Ensure adaptive_learning section exists
            if 'adaptive_learning' not in config:
                config['adaptive_learning'] = {}

            # Update with current settings
            config['adaptive_learning'].update({
                'enable_adaptive': True,
                'initial_samples_per_class': self.adaptive_config.get('initial_samples_per_class', 5),
                'margin': self.adaptive_config.get('margin', 0.1),
                'max_adaptive_rounds': self.adaptive_config.get('max_adaptive_rounds', 10)
            })

            # Ensure statistics section exists
            if 'statistics' not in config:
                config['statistics'] = {}

            # Update statistics settings
            config['statistics'].update({
                'enable_confusion_matrix': self.stats_config.get('enable_confusion_matrix', True),
                'enable_progress_plots': self.stats_config.get('enable_progress_plots', True),
                'color_progress': self.stats_config.get('color_progress', 'green'),
                'color_regression': self.stats_config.get('color_regression', 'red'),
                'save_plots': self.stats_config.get('save_plots', True)
            })

            # Save updated config
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)

            print(f"✅ Updated configuration file: {config_path}")

        except Exception as e:
            print(f"⚠️  Warning: Could not update config file: {str(e)}")

    def _load_config(self, dataset_name: str) -> Dict:
        """Load configuration from file"""
        config_path = f"{dataset_name}.conf"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}

    def prepare_adaptive_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare initial training data with distinct samples based on cardinality"""
        initial_samples = self.adaptive_config.get('initial_samples_per_class', 5)

        # Get unique classes
        unique_classes = np.unique(y)

        # Initialize indices
        all_indices = np.arange(len(X))
        self.training_indices = []
        self.test_indices = list(all_indices)

        # For each class, select the most distinct samples
        for class_id in unique_classes:
            class_indices = np.where(y == class_id)[0]

            if len(class_indices) > initial_samples:
                # Calculate pairwise distances to find diverse samples
                class_data = X[class_indices]

                # Use k-means++ initialization to select diverse samples
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=initial_samples, init='k-means++', n_init=1, random_state=42)
                kmeans.fit(class_data)

                # Find samples closest to cluster centers
                distances = kmeans.transform(class_data)
                closest_indices = np.argmin(distances, axis=0)

                selected_indices = class_indices[closest_indices]
            else:
                # Use all available samples if fewer than requested
                selected_indices = class_indices

            self.training_indices.extend(selected_indices)
            # Remove from test indices
            self.test_indices = [idx for idx in self.test_indices if idx not in selected_indices]

        # Create initial datasets
        X_train = X[self.training_indices]
        y_train = y[self.training_indices]
        X_test = X[self.test_indices]
        y_test = y[self.test_indices]

        print(f"Initial training set: {len(X_train)} samples")
        print(f"Initial test set: {len(X_test)} samples")
        print(f"Class distribution in training: {np.bincount(y_train)}")

        # Record initial statistics
        self._record_round_statistics(0, X_train, y_train, X_test, y_test, 0.0, [])

        return X_train, y_train, X_test, y_test

    def _record_round_statistics(self, round_num, X_train, y_train, X_test, y_test,
                               accuracy, samples_added):
        """Record statistics for the current round"""
        round_stat = {
            'round': round_num,
            'training_size': len(X_train),
            'test_size': len(X_test),
            'accuracy': accuracy,
            'samples_added': len(samples_added),
            'class_distribution_train': np.bincount(y_train).tolist(),
            'class_distribution_test': np.bincount(y_test).tolist(),
            'timestamp': datetime.now().isoformat(),
            'elapsed_time': (datetime.now() - self.start_time).total_seconds()
        }
        self.round_stats.append(round_stat)

        # Print round statistics
        print(f"\n📊 Round {round_num} Statistics:")
        print(f"   Training samples: {round_stat['training_size']}")
        print(f"   Test samples: {round_stat['test_size']}")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Samples added: {round_stat['samples_added']}")
        print(f"   Class distribution (train): {round_stat['class_distribution_train']}")
        print(f"   Class distribution (test): {round_stat['class_distribution_test']}")
        print(f"   Elapsed time: {round_stat['elapsed_time']:.2f}s")

    def _plot_confusion_matrix_comparison(self, y_true, y_pred, round_num, previous_cm=None):
        """Plot confusion matrix with progress indicators"""
        if not self.stats_config.get('enable_confusion_matrix', True):
            return

        current_cm = confusion_matrix(y_true, y_pred)
        n_classes = len(np.unique(y_true))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Current confusion matrix
        sns.heatmap(current_cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_title(f'Round {round_num} - Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('True')

        # Comparison matrix (if previous exists)
        if previous_cm is not None and previous_cm.shape == current_cm.shape:
            comparison = current_cm - previous_cm
            progress_color = self.stats_config.get('color_progress', 'green')
            regression_color = self.stats_config.get('color_regression', 'red')

            # Create a colormap that shows improvements in green and regressions in red
            cmap = sns.diverging_palette(220, 20, as_cmap=True)

            sns.heatmap(comparison, annot=True, fmt='d', cmap=cmap,
                       center=0, ax=ax2)
            ax2.set_title('Change from Previous Round\n(Green: Improvement, Red: Regression)')
            ax2.set_xlabel('Predicted')
            ax2.set_ylabel('True')
        else:
            ax2.text(0.5, 0.5, 'No previous matrix\nfor comparison',
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Comparison Matrix')

        plt.tight_layout()

        if self.stats_config.get('save_plots', True):
            plt.savefig(f'{self.dataset_name}_round_{round_num}_confusion.png',
                       dpi=300, bbox_inches='tight')
        plt.close()

        return current_cm

    def _plot_training_progress(self):
        """Plot overall training progress across rounds"""
        if not self.stats_config.get('enable_progress_plots', True) or len(self.round_stats) < 2:
            return

        rounds = [stat['round'] for stat in self.round_stats]
        accuracies = [stat['accuracy'] for stat in self.round_stats]
        training_sizes = [stat['training_size'] for stat in self.round_stats]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Accuracy progression
        ax1.plot(rounds, accuracies, 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy Progression')
        ax1.grid(True, alpha=0.3)

        # Training size progression
        ax2.plot(rounds, training_sizes, 's-', color='orange', linewidth=2, markersize=8)
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Training Samples')
        ax2.set_title('Training Set Growth')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if self.stats_config.get('save_plots', True):
            plt.savefig(f'{self.dataset_name}_training_progress.png',
                       dpi=300, bbox_inches='tight')
        plt.close()

    def _print_detailed_classification_report(self, y_true, y_pred, round_num):
        """Print detailed classification report"""
        print(f"\n📋 Round {round_num} - Detailed Classification Report:")
        print("=" * 60)
        report = classification_report(y_true, y_pred, zero_division=0)
        print(report)
        print("=" * 60)

    def _find_margin_based_samples(self, X: np.ndarray, y: np.ndarray,
                                 predictions: np.ndarray, posteriors: np.ndarray) -> List[int]:
        """Find samples to add based on margin criteria"""
        margin = self.adaptive_config.get('margin', 0.1)
        samples_to_add = []

        # Get the test set labels (not the full dataset labels)
        y_test = y[self.test_indices]

        # Get failed predictions (compare with test set labels, not full dataset)
        failed_mask = predictions != y_test
        failed_indices = np.where(failed_mask)[0]

        if len(failed_indices) == 0:
            print("No misclassified samples found. Model may be predicting only one class.")
            # Fallback: add diverse samples from all classes
            return self._get_diverse_fallback_samples(X, y, y_test, n_samples=10)

        # Check if posteriors has the correct shape
        n_classes = len(np.unique(y))
        if posteriors.shape[1] != n_classes:
            print(f"Warning: Posterior matrix has shape {posteriors.shape}, expected {posteriors.shape[0]} x {n_classes}")
            print("Model is likely predicting only one class. Adding diverse samples as fallback.")
            return self._get_diverse_fallback_samples(X, y, y_test, n_samples=10)

        # For each failed sample, calculate margin
        margins = []
        for i, idx_in_test in enumerate(failed_indices):
            # Convert test index back to original dataset index
            original_idx = self.test_indices[idx_in_test]
            true_class = y_test[idx_in_test]  # Use test set label
            pred_class = predictions[idx_in_test]

            # Ensure indices are within bounds
            if true_class >= posteriors.shape[1] or pred_class >= posteriors.shape[1]:
                continue

            # Calculate margin (difference between true and predicted class posteriors)
            true_posterior = posteriors[idx_in_test, true_class]
            pred_posterior = posteriors[idx_in_test, pred_class]
            margin_value = pred_posterior - true_posterior
            margins.append((original_idx, margin_value, true_class, pred_class))

        # Find maximum and minimum margin failures
        if margins:
            # Maximum margin failure (most confident wrong prediction)
            max_margin_idx, max_margin, max_true_class, max_pred_class = max(margins, key=lambda x: x[1])

            # Minimum margin failure (least confident wrong prediction)
            min_margin_idx, min_margin, min_true_class, min_pred_class = min(margins, key=lambda x: x[1])

            # Add samples based on maximum margin criteria
            max_margin_samples = self._get_margin_based_samples(
                X, y, max_true_class, max_pred_class, max_margin, margin, 'max'
            )

            # Add samples based on minimum margin criteria
            min_margin_samples = self._get_margin_based_samples(
                X, y, min_true_class, min_pred_class, min_margin, margin, 'min'
            )

            samples_to_add = list(set(max_margin_samples + min_margin_samples))

        return samples_to_add

    def _get_diverse_fallback_samples(self, X: np.ndarray, y: np.ndarray, y_test: np.ndarray, n_samples: int = 10) -> List[int]:
        """Get diverse samples when no misclassified samples are found"""
        samples_to_add = []

        # Get class distribution in test set
        class_counts = np.bincount(y_test)
        n_classes = len(class_counts)

        # Select samples proportionally from each class
        samples_per_class = max(1, n_samples // n_classes)

        for class_id in range(n_classes):
            if class_counts[class_id] > 0:  # If class exists in test set
                class_indices = np.where(y_test == class_id)[0]

                if len(class_indices) > 0:
                    # Select diverse samples using distance-based selection
                    class_data = X[self.test_indices][class_indices]

                    if len(class_indices) > samples_per_class:
                        # Use k-means to select diverse samples
                        from sklearn.cluster import KMeans
                        kmeans = KMeans(n_clusters=samples_per_class, init='k-means++', n_init=1, random_state=42)
                        kmeans.fit(class_data)

                        # Find samples closest to cluster centers
                        distances = kmeans.transform(class_data)
                        closest_indices = np.argmin(distances, axis=0)

                        selected_test_indices = class_indices[closest_indices]
                    else:
                        selected_test_indices = class_indices

                    # Convert test indices to original indices
                    samples_to_add.extend([self.test_indices[idx] for idx in selected_test_indices])

        return samples_to_add[:n_samples]  # Return at most n_samples

    def _get_margin_based_samples(self, X: np.ndarray, y: np.ndarray, true_class: int,
                                pred_class: int, reference_margin: float, margin: float,
                                margin_type: str) -> List[int]:
        """Get samples within margin of reference margin"""
        samples_to_add = []

        # Get all samples of the true class that were misclassified as pred_class
        class_mask = (y == true_class)
        misclassified_mask = np.zeros_like(y, dtype=bool)

        # For the current adaptive round, we need to get predictions on test set
        test_predictions = self.model.predict(X[self.test_indices])
        test_posteriors = self._compute_batch_posterior_for_indices(X, self.test_indices)

        # Find misclassified samples of this class
        for i, idx in enumerate(self.test_indices):
            if y[idx] == true_class and test_predictions[i] == pred_class:
                misclassified_mask[idx] = True

        misclassified_indices = np.where(misclassified_mask)[0]

        if len(misclassified_indices) == 0:
            return samples_to_add

        # Calculate margins for all misclassified samples
        margins = []
        for idx in misclassified_indices:
            # Find the position in test_indices
            test_pos = np.where(self.test_indices == idx)[0][0]
            true_posterior = test_posteriors[test_pos, true_class]
            pred_posterior = test_posteriors[test_pos, pred_class]
            sample_margin = pred_posterior - true_posterior
            margins.append((idx, sample_margin))

        # Select samples based on margin criteria
        if margin_type == 'max':
            # For maximum margin: select samples with margin close to the maximum
            threshold = reference_margin - margin
            samples_to_add = [idx for idx, m in margins if m >= threshold]
        else:  # min margin
            # For minimum margin: select samples with margin close to the minimum
            threshold = reference_margin + margin
            samples_to_add = [idx for idx, m in margins if m <= threshold]

        return samples_to_add

    def _compute_batch_posterior_for_indices(self, X: np.ndarray, indices: List[int]) -> np.ndarray:
        """Compute posteriors for specific indices"""
        X_subset = X[indices]
        return self.model._compute_batch_posterior(X_subset)

    def adaptive_learning_cycle(self, X: np.ndarray, y: np.ndarray):
        """Main adaptive learning cycle with enhanced statistics"""
        max_rounds = self.adaptive_config.get('max_adaptive_rounds', 10)

        # Store the full dataset for later use
        self.X_full = X
        self.y_full = y

        # Prepare initial data
        X_train, y_train, X_test, y_test = self.prepare_adaptive_data(X, y)

        for round_num in range(1, max_rounds + 1):
            self.adaptive_round = round_num
            print(f"\n{'='*60}")
            print(f"=== Adaptive Learning Round {round_num}/{max_rounds} ===")
            print(f"{'='*60}")

            # Train model on current training set using the original method
            print("Training model...")
            # We need to temporarily modify the model's internal data to use our training set
            self._train_with_custom_data(X_train, y_train, X_test, y_test)

            # Evaluate on test set
            test_posteriors = self.model._compute_batch_posterior(X_test)
            test_predictions = np.argmax(test_posteriors, axis=1)  # Convert posteriors to class labels

            # Ensure y_test is integer type (not float)
            y_test = y_test.astype(int)

            accuracy = accuracy_score(y_test, test_predictions)

            # Save best model
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_training_indices = self.training_indices.copy()
                self.model._save_best_weights()

            # Record statistics
            self._record_round_statistics(round_num, X_train, y_train, X_test, y_test,
                                        accuracy, [])

            # Print detailed classification report
            self._print_detailed_classification_report(y_test, test_predictions, round_num)

            # Plot confusion matrix comparison
            current_confusion = self._plot_confusion_matrix_comparison(
                y_test, test_predictions, round_num, self.previous_confusion
            )
            self.previous_confusion = current_confusion

            # Check stopping condition
            if accuracy >= 1.0 or len(self.test_indices) == 0:
                print("🎯 Stopping condition met!")
                break

            # Get posteriors for test set
            test_posteriors = self._compute_batch_posterior_for_indices(X, self.test_indices)

            # Find samples to add based on margin criteria
            samples_to_add = self._find_margin_based_samples(
                X, y, test_predictions, test_posteriors  # Use test_predictions (class labels), not posteriors
            )

            if not samples_to_add:
                print("⏹️  No more informative samples found. Stopping.")
                break

            # Add samples to training set
            print(f"📥 Adding {len(samples_to_add)} informative samples to training set")
            self.training_indices.extend(samples_to_add)
            self.test_indices = [idx for idx in self.test_indices if idx not in samples_to_add]

            # Update datasets
            X_train = X[self.training_indices]
            y_train = y[self.training_indices]
            X_test = X[self.test_indices]
            y_test = y[self.test_indices]

            # Update statistics with samples added
            self.round_stats[-1]['samples_added'] = len(samples_to_add)

            print(f"📊 New training set size: {len(X_train)}")
            print(f"📊 New test set size: {len(X_test)}")

        # Final evaluation with best model
        print(f"\n{'='*60}")
        print("=== Final Results ===")
        print(f"{'='*60}")
        print(f"🏆 Best accuracy achieved: {self.best_accuracy:.4f}")
        print(f"📦 Final training set size: {len(self.best_training_indices)}")
        print(f"⏱️  Total training time: {(datetime.now() - self.start_time).total_seconds():.2f}s")

        # Plot overall progress
        self._plot_training_progress()

        # Save final statistics
        self._save_final_statistics()

        # Load best weights
        self.model._load_best_weights()

        return self.best_accuracy, self.best_training_indices

    def _train_with_custom_data(self, X_train, y_train, X_test, y_test):
        """Temporarily modify the model's internal data for custom training"""
        # Store original data
        original_data = self.model.data.copy()

        # Create a temporary DataFrame with our custom training data
        # We need to reconstruct the DataFrame structure that GPUDBNN expects

        # Disable pruning during adaptive learning
        self.model.set_pruning_enabled(False)

        # Get the original column names from the model's data
        column_names = list(self.model.data.columns)

        # Create a temporary DataFrame with the same structure
        # For training data
        train_df = pd.DataFrame(X_train, columns=column_names[:-1])  # All columns except target
        train_df[column_names[-1]] = y_train  # Add target column

        # For test data (if needed for evaluation)
        test_df = pd.DataFrame(X_test, columns=column_names[:-1])
        test_df[column_names[-1]] = y_test

        # Temporarily replace the model's data
        self.model.data = train_df

        # Train the model using its original method
        self.model.train()

        # Restore original data and re-enable pruning for final training
        self.model.data = original_data
        self.model.set_pruning_enabled(True)

    def _save_final_statistics(self):
        """Save final statistics to file"""
        stats_file = f"{self.dataset_name}_adaptive_stats.json"

        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            else:
                return obj

        final_stats = {
            'dataset': self.dataset_name,
            'best_accuracy': float(self.best_accuracy),  # Convert to float
            'final_training_size': int(len(self.best_training_indices)),  # Convert to int
            'total_rounds': int(len(self.round_stats)),  # Convert to int
            'total_time_seconds': float((datetime.now() - self.start_time).total_seconds()),  # Convert to float
            'round_statistics': convert_numpy_types(self.round_stats),  # Convert all numpy types
            'config': self.adaptive_config
        }

        with open(stats_file, 'w') as f:
            json.dump(final_stats, f, indent=4)

        print(f"📊 Statistics saved to {stats_file}")

    def prepare_full_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare the full dataset without train/test split"""
        # Use the original DBNN's data preparation but return full dataset
        df = self.model.data

        # Encode categorical features
        df_encoded = self.model._encode_categorical_features(df)

        # Separate features and target
        if isinstance(self.model.target_column, int):
            X = df_encoded.drop(df_encoded.columns[self.model.target_column], axis=1).values
            y = df_encoded.iloc[:, self.model.target_column].values
        else:
            X = df_encoded.drop(self.model.target_column, axis=1).values
            y = df_encoded[self.model.target_column].values

        # Encode target labels
        y_encoded = self.model.label_encoder.fit_transform(y)

        # Scale features
        X_scaled = self.model.scaler.fit_transform(X)

        return X_scaled, y_encoded

def list_conf_files():
    """List all .conf files in the current directory"""
    conf_files = glob.glob("*.conf")
    return conf_files

def select_dataset():
    """Allow user to select a dataset by listing available conf files"""
    conf_files = list_conf_files()

    if not conf_files:
        print("No .conf files found in the current directory.")
        print("Please enter the dataset name manually.")
        return input("Enter dataset name: ").strip()

    print("\nAvailable configuration files:")
    print("=" * 40)
    for i, conf_file in enumerate(conf_files, 1):
        dataset_name = conf_file.replace('.conf', '')
        print(f"{i}. {dataset_name}")
    print("=" * 40)

    while True:
        try:
            choice = input("\nEnter the number or name of the dataset to use: ").strip()

            # Check if input is a number
            if choice.isdigit():
                index = int(choice) - 1
                if 0 <= index < len(conf_files):
                    return conf_files[index].replace('.conf', '')
                else:
                    print(f"Please enter a number between 1 and {len(conf_files)}")

            # Check if input is a valid dataset name (with or without .conf extension)
            else:
                # Remove .conf extension if present
                if choice.endswith('.conf'):
                    choice = choice[:-5]

                # Check if this dataset exists
                if f"{choice}.conf" in conf_files:
                    return choice
                else:
                    print(f"Dataset '{choice}' not found. Available datasets: {[f.replace('.conf', '') for f in conf_files]}")

        except (ValueError, KeyboardInterrupt):
            print("\nInvalid input. Please try again or press Ctrl+C to exit.")

def main():
    """Main function for adaptive learning"""
    # List available conf files and let user select one
    dataset_name = select_dataset()

    # Initialize adaptive learning model
    adaptive_model = AdaptiveDBNN(dataset_name)

    # Check if adaptive learning is enabled
    if not adaptive_model.adaptive_config.get('enable_adaptive', False):
        print("Adaptive learning not enabled in config. Using standard training.")
        # Fall back to standard DBNN training
        model = GPUDBNN(dataset_name)
        model.train()
        X_test, y_test = model._prepare_data()[1:3]  # Get test data for evaluation
        model.predict(X_test)
        return

    # Ask if user wants to configure settings
    configure = input("\nConfigure adaptive learning settings? (y/N): ").strip().lower()
    if configure == 'y':
        adaptive_model.configure_adaptive_learning()

    # Prepare full dataset
    print("Preparing full dataset...")
    X_full, y_full = adaptive_model.prepare_full_data()

    # Run adaptive learning
    best_accuracy, best_indices = adaptive_model.adaptive_learning_cycle(X_full, y_full)

    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj

    # Save final results
    results = {
        'best_accuracy': float(best_accuracy),  # Convert to float
        'training_indices': [int(idx) for idx in best_indices],  # Convert to list of ints
        'training_set_size': int(len(best_indices)),  # Convert to int
        'total_samples': int(len(X_full)),  # Convert to int
        'adaptive_config': adaptive_model.adaptive_config
    }

    results_file = f"{dataset_name}_adaptive_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"✅ Results saved to {results_file}")
    # Final config update
    adaptive_model._update_config_file()

if __name__ == "__main__":
    main()
