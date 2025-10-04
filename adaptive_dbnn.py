import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
import json
from collections import defaultdict
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import copy
import glob
import time
import torch
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Import the optimized DBNN implementation
import dbnn_optimised

class DatasetConfig:
    """Dataset configuration handler"""

    @staticmethod
    def get_available_datasets():
        """Get list of available datasets from configuration files"""
        config_files = glob.glob("*.conf")
        datasets = [f.replace('.conf', '') for f in config_files]
        return datasets

    @staticmethod
    def load_config(dataset_name):
        """Load configuration for a dataset"""
        config_path = f"{dataset_name}.conf"
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

class DBNNVisualizer:
    """Visualization system for DBNN"""

    def __init__(self, model, output_dir='visualizations', enabled=True):
        self.model = model
        self.output_dir = output_dir
        self.enabled = enabled
        os.makedirs(output_dir, exist_ok=True)

    def create_visualizations(self, X, y, predictions=None):
        """Create various visualizations"""
        if not self.enabled:
            return

        print("Creating visualizations...")

        # Create some basic plots
        plt.figure(figsize=(10, 6))
        plt.hist(y, bins=20, alpha=0.7, color='blue')
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Frequency')
        plt.savefig(f'{self.output_dir}/class_distribution.png')
        plt.close()

class AdaptiveDBNN:
    """Wrapper for DBNN that implements sophisticated adaptive learning with comprehensive analysis"""

    def __init__(self, dataset_name: str = None, config: Dict = None):
        # Handle dataset selection if not provided
        if dataset_name is None:
            dataset_name = self._select_dataset()

        self.dataset_name = dataset_name
        self.config = config or self._load_config(dataset_name)

        # Ensure config has required fields by using DatasetConfig
        if 'target_column' not in self.config:
            self.config = DatasetConfig.load_config(self.dataset_name)

        # Enhanced adaptive learning configuration with proper defaults
        self.adaptive_config = self.config.get('adaptive_learning', {})
        # Set defaults for any missing parameters
        default_config = {
            "enable_adaptive": True,
            "initial_samples_per_class": 5,
            "max_margin_samples_per_class": 3,
            "margin_tolerance": 0.15,
            "kl_threshold": 0.1,
            "max_adaptive_rounds": 20,
            "patience": 10,  # Fixed: Now properly set to 10
            "min_improvement": 0.001,
            "training_convergence_epochs": 50,
            "min_training_accuracy": 0.95,
            "min_samples_to_add_per_class": 5,
            "adaptive_margin_relaxation": 0.1,
            "max_divergence_samples_per_class": 5,  # New: prioritize divergence
            "exhaust_all_failed": True,  # NEW: Continue until all failed examples are exhausted
            "min_failed_threshold": 10,  # NEW: Stop when failed examples below this threshold
        }
        for key, default_value in default_config.items():
            if key not in self.adaptive_config:
                self.adaptive_config[key] = default_value

        self.stats_config = self.config.get('statistics', {
            'enable_confusion_matrix': True,
            'enable_progress_plots': True,
            'color_progress': 'green',
            'color_regression': 'red',
            'save_plots': True,
            'create_interactive_plots': True,
            'create_sample_analysis': True
        })

        # Visualization configuration
        self.viz_config = self.config.get('visualization_config', {
            'enabled': True,
            'output_dir': 'adaptive_visualizations',
            'create_animations': False,
            'create_reports': True,
            'create_3d_visualizations': True
        })

        # Initialize the base DBNN model with proper config using the imported optimized version
        self.model = dbnn_optimised.GPUDBNN(dataset_name, config=self.config)

        # Adaptive learning state
        self.training_indices = []
        self.test_indices = []
        self.best_accuracy = 0.0
        self.best_training_indices = []
        self.best_test_indices = []
        self.best_round = 0
        self.adaptive_round = 0
        self.patience_counter = 0
        self.best_weights = None

        # Statistics tracking
        self.round_stats = []
        self.previous_confusion = None
        self.start_time = datetime.now()
        self.adaptive_start_time = None
        self.device_type = self._get_device_type()

        # Store the full dataset for adaptive learning
        self.X_full = None
        self.y_full = None
        self.y_full_original = None
        self.original_data_shape = None

        # Track all selected samples for analysis
        self.all_selected_samples = defaultdict(list)
        self.sample_selection_history = []

        # Initialize label encoder for adaptive learning
        self.label_encoder = LabelEncoder()

        # Initialize visualizers
        self.adaptive_visualizer = None
        self._initialize_visualizers()

        # Update config file with default settings if they don't exist
        self._update_config_file()

        # Show current settings
        self.show_adaptive_settings()

    def _load_config(self, dataset_name: str) -> Dict:
        """Load configuration from file"""
        config_path = f"{dataset_name}.conf"
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _select_dataset(self) -> str:
        """Select dataset from available configuration files"""
        available_datasets = DatasetConfig.get_available_datasets()

        if available_datasets:
            print("Available datasets:")
            for i, dataset in enumerate(available_datasets, 1):
                print(f"{i}. {dataset}")

            choice = input(f"\nSelect a dataset (1-{len(available_datasets)}): ").strip()
            try:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(available_datasets):
                    return available_datasets[choice_idx]
                else:
                    print("Invalid selection, using default dataset name")
                    return input("Enter dataset name: ").strip()
            except ValueError:
                print("Invalid input, using default dataset name")
                return input("Enter dataset name: ").strip()
        else:
            print("No existing dataset configurations found.")
            return input("Enter dataset name: ").strip()

    def _get_device_type(self) -> str:
        """Get the device type (CPU/GPU)"""
        try:
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown GPU"
                return f"GPU: {gpu_name}"
            else:
                return "CPU"
        except:
            return "Unknown Device"

    def _initialize_visualizers(self):
        """Initialize visualization systems"""
        # Initialize adaptive visualizer
        if self.viz_config.get('enabled', True):
            try:
                self.adaptive_visualizer = DBNNVisualizer(
                    self.model,
                    output_dir=self.viz_config.get('output_dir', 'adaptive_visualizations'),
                    enabled=True
                )
                print("✓ Adaptive visualizer initialized")
            except Exception as e:
                print(f"⚠️ Could not initialize adaptive visualizer: {e}")
                self.adaptive_visualizer = None

        # Create output directory
        os.makedirs(self.viz_config.get('output_dir', 'adaptive_visualizations'), exist_ok=True)

    def show_adaptive_settings(self):
        """Display the current adaptive learning settings"""
        print("\n🔧 Advanced Adaptive Learning Settings:")
        print("=" * 60)
        for key, value in self.adaptive_config.items():
            print(f"  {key:30}: {value}")
        print(f"\n💻 Device: {self.device_type}")
        print()

    def configure_adaptive_learning(self):
        """Interactively configure adaptive learning settings"""
        print("\n🎛️  Configure Advanced Adaptive Learning Settings")
        print("=" * 60)

        try:
            initial_samples = int(input(f"Initial samples per class [{self.adaptive_config['initial_samples_per_class']}]: ")
                                or self.adaptive_config['initial_samples_per_class'])
            max_margin_samples = int(input(f"Max margin samples per class [{self.adaptive_config['max_margin_samples_per_class']}]: ")
                                   or self.adaptive_config['max_margin_samples_per_class'])
            min_samples_to_add = int(input(f"Min samples to add per class [{self.adaptive_config['min_samples_to_add_per_class']}]: ")
                                   or self.adaptive_config['min_samples_to_add_per_class'])
            margin_tol = float(input(f"Margin tolerance [{self.adaptive_config['margin_tolerance']}]: ")
                            or self.adaptive_config['margin_tolerance'])
            max_rounds = int(input(f"Maximum adaptive rounds [{self.adaptive_config['max_adaptive_rounds']}]: ")
                            or self.adaptive_config['max_adaptive_rounds'])
            patience = int(input(f"Patience for early stopping [{self.adaptive_config['patience']}]: ")
                         or self.adaptive_config['patience'])
            convergence_epochs = int(input(f"Training convergence epochs [{self.adaptive_config['training_convergence_epochs']}]: ")
                                   or self.adaptive_config['training_convergence_epochs'])

            # NEW: Exhaust all failed examples option
            exhaust_all = input(f"Exhaust all failed examples? (y/N) [{ 'y' if self.adaptive_config['exhaust_all_failed'] else 'n' }]: ").strip().lower()
            exhaust_all_failed = exhaust_all == 'y' if exhaust_all else self.adaptive_config['exhaust_all_failed']

            min_failed_threshold = int(input(f"Min failed threshold to stop [{self.adaptive_config['min_failed_threshold']}]: ")
                                     or self.adaptive_config['min_failed_threshold'])

            # Update settings
            self.adaptive_config.update({
                'initial_samples_per_class': initial_samples,
                'max_margin_samples_per_class': max_margin_samples,
                'min_samples_to_add_per_class': min_samples_to_add,
                'margin_tolerance': margin_tol,
                'max_adaptive_rounds': max_rounds,
                'patience': patience,
                'training_convergence_epochs': convergence_epochs,
                'exhaust_all_failed': exhaust_all_failed,
                'min_failed_threshold': min_failed_threshold
            })

            self._update_config_file()
            print("✅ Advanced settings updated successfully!")
            self.show_adaptive_settings()

        except ValueError:
            print("❌ Invalid input. Settings not changed.")

    def _update_config_file(self):
        """Update the dataset configuration file with adaptive learning settings"""
        config_path = f"{self.dataset_name}.conf"
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
            else:
                config = {}

            if 'adaptive_learning' not in config:
                config['adaptive_learning'] = {}

            config['adaptive_learning'].update(self.adaptive_config)

            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)

            print(f"✅ Updated configuration file: {config_path}")
        except Exception as e:
            print(f"⚠️  Warning: Could not update config file: {str(e)}")

    def _train_until_convergence(self, X_train: np.ndarray, y_train: np.ndarray) -> bool:
        """
        Train the model until training accuracy converges
        Returns True if training was successful
        """
        print("🎯 Training model with current training set...")

        try:
            # Store original training configuration
            original_max_epochs = self.model.max_epochs
            original_trials = self.model.trials

            # Set convergence parameters
            self.model.max_epochs = 1000
            self.model.trials = self.adaptive_config['training_convergence_epochs']

            # Train with custom data - we'll use the model's internal training method
            # For adaptive learning, we need to temporarily replace the model's data
            self._train_with_custom_data(X_train, y_train)

            # Restore original parameters
            self.model.max_epochs = original_max_epochs
            self.model.trials = original_trials

            return True

        except Exception as e:
            print(f"⚠️ Training error: {e}")
            return False

    def _select_informative_samples(self, X: np.ndarray, y: np.ndarray,
                                  predictions: np.ndarray, posteriors: np.ndarray) -> List[int]:
        """
        AGGRESSIVE: Select informative samples within margin tolerance
        with maximum limit of 10% of failed examples per class
        """
        samples_to_add = []
        unique_classes = np.unique(y)
        margin_tol = 0.10  # 10% of maximum margin as tolerance
        min_samples_per_class = self.adaptive_config['min_samples_to_add_per_class']
        max_percentage_per_class = 0.10  # Maximum 10% of failed examples per class

        print("🔍 Selecting informative samples with 10% maximum per class...")

        # Get test set predictions and true labels
        y_test = y[self.test_indices]

        # Find all misclassified samples
        misclassified_mask = predictions != y_test
        misclassified_indices = np.where(misclassified_mask)[0]

        if len(misclassified_indices) == 0:
            print("✅ No misclassified samples found - model is performing well!")
            return samples_to_add

        print(f"📊 Found {len(misclassified_indices)} misclassified samples")

        # Calculate margins for all misclassified samples and group by true class
        class_margins = defaultdict(list)
        total_failed_by_class = defaultdict(int)

        for i, idx_in_test in enumerate(misclassified_indices):
            original_idx = self.test_indices[idx_in_test]
            true_class = y_test[idx_in_test]
            pred_class = predictions[idx_in_test]

            # Get original class labels for proper comparison
            true_class_original = self._get_original_class_label(true_class)
            pred_class_original = self._get_original_class_label(pred_class)

            # Ensure indices are within bounds
            if true_class < posteriors.shape[1] and pred_class < posteriors.shape[1]:
                true_posterior = posteriors[idx_in_test, true_class]
                pred_posterior = posteriors[idx_in_test, pred_class]
                margin = pred_posterior - true_posterior

                # Store only the essential information
                class_margins[true_class].append((
                    original_idx, margin, true_class, pred_class,
                    true_class_original, pred_class_original
                ))

            total_failed_by_class[true_class] += 1

        # Print class-wise failed statistics
        print("📊 Class-wise failed samples:")
        for class_id in unique_classes:
            failed_count = total_failed_by_class.get(class_id, 0)
            print(f"   Class {class_id}: {failed_count} failed samples")

        # Calculate maximum margin for tolerance calculation
        all_margins = []
        for class_id, margins in class_margins.items():
            all_margins.extend([m[1] for m in margins])

        if not all_margins:
            print("⚠️ No valid margins calculated - using random selection")
            n_to_select = min(len(misclassified_indices), min_samples_per_class * len(unique_classes))
            random_indices = np.random.choice(misclassified_indices, n_to_select, replace=False)
            return [self.test_indices[idx] for idx in random_indices]

        max_abs_margin = max(abs(min(all_margins)), abs(max(all_margins)))
        dynamic_tolerance = margin_tol * max_abs_margin
        print(f"📏 Using dynamic tolerance: {dynamic_tolerance:.4f} (10% of max margin {max_abs_margin:.4f})")

        # STEP 1: For each class, select samples within margin tolerance up to 10% maximum
        print("📈 Selecting samples within tolerance (max 10% per class)...")
        for class_id in unique_classes:
            if class_id not in class_margins or not class_margins[class_id]:
                print(f"   ⚠️ Class {class_id}: No failed samples available")
                continue

            class_failed_count = total_failed_by_class[class_id]
            class_samples = class_margins[class_id]

            # Calculate maximum samples for this class (10% of failed examples)
            max_samples_for_class = max(
                min_samples_per_class,  # At least minimum
                int(class_failed_count * max_percentage_per_class)  # Up to 10%
            )

            # Sort by absolute margin (most informative first)
            class_samples_sorted = sorted(class_samples, key=lambda x: abs(x[1]), reverse=True)

            # Take reference margins from top candidates
            if class_samples_sorted:
                # Use the top candidate's margin as reference
                reference_margin = class_samples_sorted[0][1]

                # Select samples within tolerance of the reference
                candidates_within_tolerance = [
                    candidate for candidate in class_samples_sorted
                    if abs(candidate[1] - reference_margin) <= dynamic_tolerance
                ]

                # Limit to maximum samples for this class
                selected_candidates = candidates_within_tolerance[:max_samples_for_class]

                # If we have fewer than min_samples_per_class, take more from sorted list
                if len(selected_candidates) < min_samples_per_class:
                    selected_candidates = class_samples_sorted[:min_samples_per_class]

                print(f"   Class {class_id}: Selected {len(selected_candidates)}/{class_failed_count} samples "
                      f"({len(selected_candidates)/class_failed_count*100:.1f}%, max: {max_samples_for_class})")

                for candidate in selected_candidates:
                    candidate_idx, candidate_margin, true_class, pred_class, true_orig, pred_orig = candidate

                    if candidate_idx not in samples_to_add and candidate_idx not in self.training_indices:
                        samples_to_add.append(candidate_idx)

                        # Track selection reason
                        margin_type = 'max_margin' if candidate_margin > 0 else 'min_margin'
                        self.all_selected_samples[true_orig].append({
                            'index': candidate_idx,
                            'margin': candidate_margin,
                            'selection_type': f'{margin_type}',
                            'round': self.adaptive_round,
                            'true_class_original': true_orig,
                            'pred_class_original': pred_orig,
                            'tolerance_used': dynamic_tolerance,
                            'class_failed_count': class_failed_count,
                            'max_percentage': max_percentage_per_class,
                            'relaxation_applied': False
                        })

        # Final selection summary
        selected_by_class = defaultdict(int)
        for idx in samples_to_add:
            for class_id, margins in class_margins.items():
                if any(m[0] == idx for m in margins):
                    selected_by_class[class_id] += 1
                    break

        print("📊 Final selection summary:")
        total_selected = len(samples_to_add)
        for class_id in unique_classes:
            selected_count = selected_by_class.get(class_id, 0)
            failed_count = total_failed_by_class.get(class_id, 0)
            percentage = (selected_count / failed_count * 100) if failed_count > 0 else 0
            print(f"   Class {class_id}: {selected_count}/{failed_count} selected ({percentage:.1f}%)")

        print(f"🎯 Selected {total_selected} informative samples total")

        return samples_to_add

    def _get_original_class_label(self, encoded_label: int) -> Any:
        """Convert encoded label back to original class label"""
        if hasattr(self, 'label_encoder') and hasattr(self.label_encoder, 'classes_'):
            if encoded_label < len(self.label_encoder.classes_):
                return self.label_encoder.classes_[encoded_label]
        return encoded_label

    def prepare_adaptive_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare initial training data with diverse samples from each class"""
        initial_samples = self.adaptive_config['initial_samples_per_class']
        unique_classes = np.unique(y)

        # Initialize indices
        all_indices = np.arange(len(X))
        self.training_indices = []
        self.test_indices = list(all_indices)

        print("🎯 Selecting initial diverse training samples...")

        for class_id in unique_classes:
            class_indices = np.where(y == class_id)[0]

            if len(class_indices) > initial_samples:
                # Use k-means++ to select diverse samples
                class_data = X[class_indices]
                kmeans = KMeans(n_clusters=initial_samples, init='k-means++', n_init=1, random_state=42)
                kmeans.fit(class_data)

                # Find samples closest to cluster centers
                distances = kmeans.transform(class_data)
                closest_indices = np.argmin(distances, axis=0)
                selected_indices = class_indices[closest_indices]
            else:
                # Use all available samples
                selected_indices = class_indices

            self.training_indices.extend(selected_indices)
            self.test_indices = [idx for idx in self.test_indices if idx not in selected_indices]

        # Create datasets
        X_train = X[self.training_indices]
        y_train = y[self.training_indices]
        X_test = X[self.test_indices]
        y_test = y[self.test_indices]

        # Get original class labels for display
        original_class_labels = [self._get_original_class_label(cls) for cls in np.unique(y_train)]

        print(f"📊 Initial training set: {len(X_train)} samples")
        print(f"📊 Initial test set: {len(X_test)} samples")
        print(f"🎯 Class distribution (original labels): {original_class_labels}")
        print(f"🎯 Class counts: {np.bincount(y_train)}")

        # Initialize best indices
        self.best_training_indices = self.training_indices.copy()
        self.best_test_indices = self.test_indices.copy()
        self.best_accuracy = 0.0

        return X_train, y_train, X_test, y_test

    def adaptive_learn(self, X: np.ndarray = None, y: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Main adaptive learning loop with optional exhaustive mode"""
        print("\n🚀 Starting Advanced Adaptive Learning Process")
        print("=" * 70)
        print(f"Dataset: {self.dataset_name}")
        print(f"Device: {self.device_type}")
        print("=" * 70)

        # Use base model's data if not provided
        if X is None or y is None:
            print("Using base model's dataset...")
            X, y, y_original = self.prepare_full_data()

        # Store the full dataset with original labels
        self.X_full = X.copy()
        self.y_full = y.copy()
        self.y_full_original = y_original
        self.original_data_shape = X.shape

        print(f"📦 Total samples: {len(X)}")
        print(f"🎯 Classes (original): {np.unique(self.y_full_original)}")
        print(f"🎯 Classes (encoded): {np.unique(y)}")

        exhaust_all_failed = self.adaptive_config['exhaust_all_failed']
        min_failed_threshold = self.adaptive_config['min_failed_threshold']

        if exhaust_all_failed:
            print("🔁 MODE: EXHAUSTIVE - Continuing until failed examples are addressed")
            print(f"🎯 Min failed threshold: {min_failed_threshold}")
            # Set max_rounds to a very high number for exhaustive mode
            effective_max_rounds = 1000  # Essentially unlimited
        else:
            print(f"🔄 MODE: PATIENCE - Max rounds: {self.adaptive_config['max_adaptive_rounds']}")
            print(f"⏳ Patience: {self.adaptive_config['patience']}")
            effective_max_rounds = self.adaptive_config['max_adaptive_rounds']

        # Initialize adaptive learning
        X_train, y_train, X_test, y_test = self.prepare_adaptive_data(X, y)
        self.adaptive_start_time = datetime.now()

        # Track failed examples history
        failed_history = []

        # Main adaptive learning loop
        patience = self.adaptive_config['patience']

        # Use while loop for exhaustive mode to avoid round limits
        round_num = 0
        while True:
            round_num += 1
            self.adaptive_round = round_num

            if exhaust_all_failed:
                print(f"\n🔄 Adaptive Learning Round {round_num} (Exhaustive Mode)")
            else:
                print(f"\n🔄 Adaptive Learning Round {round_num}/{effective_max_rounds}")
            print("=" * 50)

            # Safety check: prevent infinite loops
            if round_num > 1000:  # Absolute maximum safety limit
                print("🛑 Safety limit reached: 1000 rounds")
                break

            # Step 1: Train until convergence
            round_start_time = datetime.now()
            training_success = self._train_until_convergence(X_train, y_train)

            if not training_success:
                print("❌ Training failed, skipping round")
                continue

            # Step 2: Predict on test set
            print("🎯 Evaluating on test set...")
            predictions = self._predict_with_current_model(X_test)
            posteriors = self._get_posteriors_with_current_model(X_test)

            # Ensure consistent lengths
            if len(predictions) != len(y_test):
                min_len = min(len(predictions), len(y_test))
                predictions = predictions[:min_len]
                y_test = y_test[:min_len]

            # Calculate accuracy and count failed examples
            accuracy = accuracy_score(y_test, predictions)
            misclassified_mask = predictions != y_test
            failed_count = np.sum(misclassified_mask)
            failed_history.append(failed_count)

            print(f"📊 Round {round_num} Test Accuracy: {accuracy:.4f}")
            print(f"❌ Failed examples: {failed_count}")

            # Create round visualizations (only for first few rounds to save time)
            if self.stats_config.get('enable_confusion_matrix', True) and round_num <= 20:
                self._create_round_visualizations(round_num, y_test, predictions)

            # Step 3: Save best model if accuracy improved
            improvement_threshold = self.adaptive_config['min_improvement']
            if accuracy > self.best_accuracy + improvement_threshold:
                improvement = accuracy - self.best_accuracy
                self.best_accuracy = accuracy
                self.best_round = round_num
                self.best_training_indices = self.training_indices.copy()
                self.best_test_indices = self.test_indices.copy()
                self.best_weights = self.model.current_W.copy() if hasattr(self.model, 'current_W') else None
                self.patience_counter = 0  # Reset patience when accuracy improves

                print(f"🎯 NEW BEST ACCURACY: {accuracy:.4f} (improvement: {improvement:.4f})")
                print(f"📦 Best training set size: {len(self.best_training_indices)}")

                # Save best weights
                if self.best_weights is not None:
                    self.model._save_best_weights()
            else:
                self.patience_counter += 1
                if not exhaust_all_failed:  # Only show patience counter in patience mode
                    print(f"⏳ No improvement - Patience: {self.patience_counter}/{patience}")

            # Step 4: Select informative samples for next round
            print("🎯 Selecting informative samples...")
            samples_to_add = self._select_informative_samples(X, y, predictions, posteriors)

            # Step 5: Check stopping conditions
            stop_reason = self._check_stopping_conditions(
                round_num, samples_to_add, failed_count, min_failed_threshold,
                patience, exhaust_all_failed, effective_max_rounds
            )

            if stop_reason:
                print(f"🛑 {stop_reason}")
                break

            # Step 6: Update training set
            print("🎯 Updating training set...")
            added_count = len(samples_to_add)
            self.training_indices.extend(samples_to_add)
            self.test_indices = [idx for idx in self.test_indices if idx not in samples_to_add]

            # Update datasets
            X_train = X[self.training_indices]
            y_train = y[self.training_indices]
            X_test = X[self.test_indices]
            y_test = y[self.test_indices]

            # Record statistics
            round_duration = (datetime.now() - round_start_time).total_seconds()
            self._record_round_statistics(round_num, X_train, y_train, X_test, y_test,
                                        accuracy, samples_to_add, round_duration, failed_count)

            print(f"📥 Added {added_count} samples to training set")
            print(f"📊 New training size: {len(X_train)}")
            print(f"📊 New test size: {len(X_test)}")
            print(f"❌ Remaining failed examples: {failed_count}")
            print(f"⏱️ Round duration: {round_duration:.2f}s")

            # Progress indicator for exhaustive mode
            if exhaust_all_failed and round_num % 10 == 0:
                initial_test_size = len(X) - len(self.best_training_indices)
                progress = ((initial_test_size - len(X_test)) / initial_test_size) * 100
                print(f"📈 Progress: {progress:.1f}% of test set processed")

        # Final processing
        print(f"\n🏁 Adaptive Learning Completed after {round_num} rounds")
        print("=" * 70)

        # Use best configuration
        X_train_best = X[self.best_training_indices]
        y_train_best = y[self.best_training_indices]
        X_test_best = X[self.best_test_indices]
        y_test_best = y[self.best_test_indices]

        print(f"🎯 Best accuracy: {self.best_accuracy:.4f} (achieved at round {self.best_round})")
        print(f"📦 Optimal training set size: {len(X_train_best)}")
        print(f"📊 Final test set size: {len(X_test_best)}")
        print(f"❌ Final failed examples: {failed_history[-1] if failed_history else 'N/A'}")
        print(f"⏱️ Total training time: {(datetime.now() - self.adaptive_start_time).total_seconds():.2f}s")

        # Retrain final model with best configuration
        print("\n🔧 Training final model with optimal configuration...")
        self._train_with_custom_data(X_train_best, y_train_best)

        # Create comprehensive visualizations and analysis
        self._create_comprehensive_analysis(X_train_best, y_train_best, X_test_best, y_test_best)

        # Save final results
        self._save_final_results(X_train_best, y_train_best, X_test_best, y_test_best)

        return X_train_best, y_train_best, X_test_best, y_test_best

    def _check_stopping_conditions(self, round_num: int, samples_to_add: List[int],
                                 failed_count: int, min_failed_threshold: int,
                                 patience: int, exhaust_all_failed: bool, max_rounds: int) -> str:
        """Check all stopping conditions and return reason if should stop"""

        # 1. No more samples to add
        if not samples_to_add:
            return "No more informative samples found!"

        # 2. EXHAUSTIVE MODE: Only stop when failed examples are effectively addressed
        if exhaust_all_failed:
            if failed_count <= min_failed_threshold:
                return f"Failed examples ({failed_count}) below threshold ({min_failed_threshold})"

            # In exhaustive mode, also stop if test set is too small to be meaningful
            if len(self.test_indices) <= min_failed_threshold * 2:
                return f"Test set too small ({len(self.test_indices)} samples) for meaningful evaluation"

            # Continue learning regardless of rounds or patience
            return ""

        # 3. Max rounds
        if round_num >= max_rounds:
            return f"Reached maximum rounds ({max_rounds})"

        # 4. Patience-based stopping (only if not in exhaustive mode)
        if not exhaust_all_failed and self.patience_counter >= patience:
            return f"Early stopping after {patience} rounds without improvement"

        # 5. Exhaustive mode: stop when failed examples are below threshold
        if exhaust_all_failed and failed_count <= min_failed_threshold:
            return f"Failed examples ({failed_count}) below threshold ({min_failed_threshold})"

        # 5. Exhaustive mode: stop when test set is too small
        if exhaust_all_failed and len(self.test_indices) <= min_failed_threshold * 2:
            return f"Test set too small ({len(self.test_indices)} samples)"

        return ""  # Continue learning

    def _record_round_statistics(self, round_num: int, X_train: np.ndarray, y_train: np.ndarray,
                               X_test: np.ndarray, y_test: np.ndarray, accuracy: float,
                               samples_added: List[int], duration: float, failed_count: int):
        """Record comprehensive statistics for the current round"""
        # Get class distribution with original labels
        y_train_original = np.array([self._get_original_class_label(cls) for cls in y_train])
        unique_classes, class_counts = np.unique(y_train_original, return_counts=True)
        class_distribution = dict(zip(unique_classes, class_counts))

        stats = {
            'round': round_num,
            'training_size': len(X_train),
            'test_size': len(X_test),
            'accuracy': accuracy,
            'samples_added': len(samples_added),
            'class_distribution': class_distribution,
            'duration': duration,
            'failed_count': failed_count,  # NEW: Track failed examples
            'patience_counter': self.patience_counter,
            'best_accuracy': self.best_accuracy,
            'timestamp': datetime.now().isoformat()
        }

        self.round_stats.append(stats)

    def _create_round_visualizations(self, round_num: int, y_true: np.ndarray, y_pred: np.ndarray):
        """Create visualizations for the current round"""
        if not self.stats_config.get('enable_confusion_matrix', True):
            return

        # Convert to original labels for confusion matrix
        y_true_original = np.array([self._get_original_class_label(cls) for cls in y_true])
        y_pred_original = np.array([self._get_original_class_label(cls) for cls in y_pred])

        # Create confusion matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true_original, y_pred_original)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Round {round_num} - Confusion Matrix\nAccuracy: {accuracy_score(y_true, y_pred):.4f}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        # Save with round number
        output_dir = self.viz_config.get('output_dir', 'adaptive_visualizations')
        plt.savefig(f'{output_dir}/confusion_matrix_round_{round_num:03d}.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _create_comprehensive_analysis(self, X_train: np.ndarray, y_train: np.ndarray,
                                    X_test: np.ndarray, y_test: np.ndarray):
        """Create comprehensive analysis of the adaptive learning process"""
        print("\n📊 Creating comprehensive analysis...")

        # 1. Create learning curve
        self._create_learning_curve()

        # 2. Create sample selection analysis
        self._create_sample_selection_analysis()

        # 3. Create final model evaluation
        self._create_final_evaluation(X_test, y_test)

        # 4. Create adaptive learning summary
        self._create_adaptive_summary()

        print("✅ Comprehensive analysis completed!")

    def _create_learning_curve(self):
        """Create learning curve showing accuracy and training size over rounds"""
        if len(self.round_stats) < 2:
            print("⚠️ Not enough rounds for learning curve")
            return

        rounds = [stats['round'] for stats in self.round_stats]
        accuracies = [stats['accuracy'] for stats in self.round_stats]
        training_sizes = [stats['training_size'] for stats in self.round_stats]
        failed_counts = [stats['failed_count'] for stats in self.round_stats]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Accuracy progression
        ax1.plot(rounds, accuracies, 'b-o', linewidth=2, markersize=6)
        ax1.set_xlabel('Adaptive Round')
        ax1.set_ylabel('Test Accuracy')
        ax1.set_title('Adaptive Learning - Accuracy Progression')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=self.best_accuracy, color='r', linestyle='--', label=f'Best: {self.best_accuracy:.4f}')
        ax1.legend()

        # Plot 2: Training size progression
        ax2.plot(rounds, training_sizes, 'g-o', linewidth=2, markersize=6)
        ax2.set_xlabel('Adaptive Round')
        ax2.set_ylabel('Training Set Size')
        ax2.set_title('Training Set Growth')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Failed examples progression
        ax3.plot(rounds, failed_counts, 'r-o', linewidth=2, markersize=6)
        ax3.set_xlabel('Adaptive Round')
        ax3.set_ylabel('Failed Examples')
        ax3.set_title('Failed Examples Reduction')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Samples added per round
        samples_added = [stats['samples_added'] for stats in self.round_stats]
        ax4.bar(rounds, samples_added, alpha=0.7, color='orange')
        ax4.set_xlabel('Adaptive Round')
        ax4.set_ylabel('Samples Added')
        ax4.set_title('Samples Added per Round')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        output_dir = self.viz_config.get('output_dir', 'adaptive_visualizations')
        plt.savefig(f'{output_dir}/adaptive_learning_curve.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _create_sample_selection_analysis(self):
        """Create analysis of sample selection patterns"""
        if not self.all_selected_samples:
            print("⚠️ No sample selection data available")
            return

        # Analyze selection patterns by class and round
        selection_data = []
        for class_label, samples in self.all_selected_samples.items():
            for sample_info in samples:
                selection_data.append({
                    'class': class_label,
                    'round': sample_info['round'],
                    'margin': sample_info['margin'],
                    'selection_type': sample_info['selection_type']
                })

        if not selection_data:
            return

        df = pd.DataFrame(selection_data)

        plt.figure(figsize=(12, 8))

        # Plot 1: Samples selected by class and round
        plt.subplot(2, 2, 1)
        class_round_counts = df.groupby(['class', 'round']).size().unstack(fill_value=0)
        class_round_counts.T.plot(kind='bar', stacked=True, ax=plt.gca())
        plt.title('Samples Selected by Class and Round')
        plt.xlabel('Adaptive Round')
        plt.ylabel('Number of Samples')
        plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')

        # Plot 2: Margin distribution by class
        plt.subplot(2, 2, 2)
        df.boxplot(column='margin', by='class', ax=plt.gca())
        plt.title('Margin Distribution by Class')
        plt.suptitle('')  # Remove automatic title
        plt.xlabel('Class')
        plt.ylabel('Margin')

        # Plot 3: Selection types
        plt.subplot(2, 2, 3)
        selection_counts = df['selection_type'].value_counts()
        plt.pie(selection_counts.values, labels=selection_counts.index, autopct='%1.1f%%')
        plt.title('Sample Selection Types')

        # Plot 4: Rounds with most selections
        plt.subplot(2, 2, 4)
        round_counts = df['round'].value_counts().sort_index()
        plt.bar(round_counts.index, round_counts.values)
        plt.title('Samples Selected per Round')
        plt.xlabel('Round')
        plt.ylabel('Samples Selected')

        plt.tight_layout()
        output_dir = self.viz_config.get('output_dir', 'adaptive_visualizations')
        plt.savefig(f'{output_dir}/sample_selection_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _create_final_evaluation(self, X_test: np.ndarray, y_test: np.ndarray):
        """Create final model evaluation"""
        print("📈 Creating final model evaluation...")

        # Get predictions
        y_pred = self._predict_with_current_model(X_test)
        y_test_original = np.array([self._get_original_class_label(cls) for cls in y_test])
        y_pred_original = np.array([self._get_original_class_label(cls) for cls in y_pred])

        # Final accuracy
        final_accuracy = accuracy_score(y_test, y_pred)
        print(f"🎯 Final Test Accuracy: {final_accuracy:.4f}")

        # Detailed classification report
        print("\n📋 Detailed Classification Report:")
        print(classification_report(y_test_original, y_pred_original))

        # Final confusion matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test_original, y_pred_original)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Final Model - Confusion Matrix\nAccuracy: {final_accuracy:.4f}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        output_dir = self.viz_config.get('output_dir', 'adaptive_visualizations')
        plt.savefig(f'{output_dir}/final_confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _create_adaptive_summary(self):
        """Create comprehensive adaptive learning summary"""
        summary = {
            'dataset': self.dataset_name,
            'device': self.device_type,
            'total_rounds': len(self.round_stats),
            'best_round': self.best_round,
            'best_accuracy': self.best_accuracy,
            'final_training_size': len(self.best_training_indices),
            'adaptive_config': self.adaptive_config,
            'start_time': self.adaptive_start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_duration': (datetime.now() - self.adaptive_start_time).total_seconds(),
            'round_statistics': self.round_stats,
            'sample_selection_summary': {
                class_label: len(samples)
                for class_label, samples in self.all_selected_samples.items()
            }
        }

        # Save summary as JSON
        output_dir = self.viz_config.get('output_dir', 'adaptive_visualizations')
        with open(f'{output_dir}/adaptive_learning_summary.json', 'w') as f:
            json.dump(summary, f, indent=4, default=str)

        # Print summary to console
        print("\n📊 ADAPTIVE LEARNING SUMMARY")
        print("=" * 50)
        print(f"Dataset: {summary['dataset']}")
        print(f"Device: {summary['device']}")
        print(f"Total Rounds: {summary['total_rounds']}")
        print(f"Best Round: {summary['best_round']}")
        print(f"Best Accuracy: {summary['best_accuracy']:.4f}")
        print(f"Final Training Size: {summary['final_training_size']}")
        print(f"Total Duration: {summary['total_duration']:.2f}s")
        print(f"Sample Selection Summary: {summary['sample_selection_summary']}")

    def _save_final_results(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray):
        """Save final results and model"""
        print("\n💾 Saving final results...")

        output_dir = self.viz_config.get('output_dir', 'adaptive_visualizations')

        # Save indices
        results = {
            'best_training_indices': self.best_training_indices,
            'best_test_indices': self.best_test_indices,
            'best_accuracy': self.best_accuracy,
            'best_round': self.best_round,
            'adaptive_config': self.adaptive_config,
            'round_statistics': self.round_stats
        }

        with open(f'{output_dir}/adaptive_results.json', 'w') as f:
            json.dump(results, f, indent=4, default=str)

        # Save the trained model
        self.model.save_model()

        print("✅ Final results saved!")

    def prepare_full_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare full dataset for adaptive learning"""
        # Use the base model's data preparation
        df = self.model.data
        feature_cols = [col for col in df.columns if col != self.model.target_column]
        X = df[feature_cols].values
        y_original = df[self.model.target_column].values

        # Fit label encoder for adaptive learning
        self.label_encoder.fit(y_original)
        y = self.label_encoder.transform(y_original)
        # Fit label encoder for adaptive learning
        self.label_encoder.fit(y_original)

        return X, y, y_original

    def _train_with_custom_data(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the model with custom data"""
        # This method needs to be implemented to handle custom training data
        # For now, we'll use the model's internal training method
        try:
            # Convert to appropriate format if needed
            original_data = self.model.data.copy()
            # Create temporary DataFrame with custom training data
            feature_cols = [col for col in original_data.columns if col != self.model.target_column]
            temp_df = pd.DataFrame(X_train, columns=feature_cols)
            temp_df[self.model.target_column] = self.label_encoder.inverse_transform(y_train)

            # Replace model data temporarily
            self.model.data = temp_df

            # Train the model
            self.model.train()

            # Restore original data
            self.model.data = original_data
        except Exception as e:
            print(f"⚠️ Custom training error: {e}")

    def _predict_with_current_model(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from current model"""
        try:
            return self.model.predict(X)
        except Exception as e:
            print(f"⚠️ Prediction error: {e}")
            # Return random predictions as fallback
            return np.random.randint(0, len(np.unique(self.y_full)), len(X))

    def _get_posteriors_with_current_model(self, X: np.ndarray) -> np.ndarray:
        """Get posterior probabilities from current model"""
        try:
            return self.model._compute_batch_posterior(X)
        except Exception as e:
            print(f"⚠️ Posterior error: {e}")
            n_classes = len(np.unique(self.y_full))
            return np.ones((len(X), n_classes)) / n_classes

def main():
    """Main function to demonstrate adaptive DBNN"""
    print("🎯 Advanced Adaptive DBNN System")
    print("=" * 50)

    # Create and run adaptive DBNN
    adaptive_model = AdaptiveDBNN()

    # Optionally configure settings
    configure = input("\nConfigure adaptive learning settings? (y/N): ").strip().lower()
    if configure == 'y':
        adaptive_model.configure_adaptive_learning()

    # Run adaptive learning
    print("\n🚀 Starting adaptive learning process...")
    X_train, y_train, X_test, y_test = adaptive_model.adaptive_learn()

    print(f"\n✅ Adaptive learning completed!")
    print(f"📦 Final training set size: {len(X_train)}")
    print(f"📊 Final test set size: {len(X_test)}")
    print(f"🎯 Best accuracy achieved: {adaptive_model.best_accuracy:.4f}")

if __name__ == "__main__":
    main()
