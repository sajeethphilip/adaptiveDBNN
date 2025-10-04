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
import time
import torch

from dbnn_optimised import GPUDBNN  # Import the original DBNN class
from dbnn_visualizer import DBNNVisualizer  # Import the visualizer

class AdaptiveDBNN:
    """Wrapper for DBNN that implements adaptive learning with enhanced statistics"""

    def __init__(self, dataset_name: str, config: Dict = None):
        self.dataset_name = dataset_name
        # Only change model name, keep data file name same
        self.adaptive_model_name = f"{dataset_name}_adaptive"
        self.config = config or self._load_config(dataset_name)

        # Use the same adaptive config settings as before
        self.adaptive_config = self.config.get('adaptive_learning', {
            "enable_adaptive": True,
            "initial_samples_per_class": 10,
            "margin": 0.1,
            "max_adaptive_rounds": 10,
            "patience": 3,
            "min_improvement": 0.001
        })

        # FIRST: Load the original model to get all its configuration
        print("📋 Loading original model configuration...")
        original_model = GPUDBNN(dataset_name)

        # NOW initialize the adaptive model with the same configuration
        # But pass the existing config to avoid re-prompting
        self.model = GPUDBNN(self.adaptive_model_name, config=self.config)

        # Copy critical configuration from original model to avoid re-prompting
        self._copy_model_configuration(original_model, self.model)

        # If the model still doesn't have data loaded, try to load it from original config
        if not hasattr(self.model, 'data') or self.model.data is None:
            self._load_data_from_config()

        # Rest of initialization
        self.stats_config = self.config.get('statistics', {
            'enable_confusion_matrix': True,
            'enable_progress_plots': True,
            'color_progress': 'green',
            'color_regression': 'red',
            'save_plots': True
        })

        self.adaptive_viz_config = self.config.get('adaptive_visualization', {
            'enabled': True,
            'output_dir': 'adaptive_visualizations',
            'create_final_visualizations': True,
            'save_indices_on_improvement': True
        })

        # Adaptive learning state
        self.training_indices = []  # Indices of samples in training set
        self.test_indices = []      # Indices of samples in test set
        self.best_accuracy = 0.0
        self.best_training_indices = []
        self.best_test_indices = []
        self.adaptive_round = 0
        self.patience_counter = 0
        self.best_accuracy_round = 0
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

        # Initialize adaptive visualizer
        self.adaptive_visualizer = None
        self._initialize_adaptive_visualizer()

        # Update config file with default settings if they don't exist
        self._update_config_file()

        # Show current settings
        self.show_adaptive_settings()

    def _copy_model_configuration(self, source_model, target_model):
        """Copy configuration from source model to target model to avoid re-prompting"""
        try:
            # Copy data-related attributes
            if hasattr(source_model, 'data') and source_model.data is not None:
                target_model.data = source_model.data.copy()
                print("✅ Copied dataset configuration")

            # Copy file paths
            if hasattr(source_model, 'csv_file_path'):
                target_model.csv_file_path = source_model.csv_file_path
                print(f"✅ Set CSV file path: {source_model.csv_file_path}")

            # Copy target column
            if hasattr(source_model, 'target_column'):
                target_model.target_column = source_model.target_column
                print(f"✅ Set target column: {source_model.target_column}")

            # Copy action mode
            if hasattr(source_model, 'action'):
                target_model.action = source_model.action
                print(f"✅ Set action mode: {source_model.action}")

            # Copy other critical configuration
            if hasattr(source_model, 'model_filename'):
                # Keep adaptive model name but ensure other config is copied
                print(f"✅ Using model filename: {target_model.model_filename}")

            # Copy preprocessing objects if they exist
            if hasattr(source_model, 'scaler') and source_model.scaler is not None:
                target_model.scaler = source_model.scaler
                print("✅ Copied scaler configuration")

            if hasattr(source_model, 'label_encoder') and source_model.label_encoder is not None:
                target_model.label_encoder = source_model.label_encoder
                print("✅ Copied label encoder")

            if hasattr(source_model, 'categorical_encoders'):
                target_model.categorical_encoders = source_model.categorical_encoders.copy()
                print("✅ Copied categorical encoders")

        except Exception as e:
            print(f"⚠️ Warning: Could not copy some model configuration: {e}")

    def _load_data_from_config(self):
        """Load data directly from configuration to avoid interactive prompts"""
        try:
            if 'csv_file_path' in self.config and os.path.exists(self.config['csv_file_path']):
                print(f"📁 Loading data from: {self.config['csv_file_path']}")
                self.model.data = pd.read_csv(self.config['csv_file_path'])
                print(f"✅ Loaded dataset with shape: {self.model.data.shape}")

                # Set other critical parameters from config
                if 'target_column' in self.config:
                    self.model.target_column = self.config['target_column']
                if 'action' in self.config:
                    self.model.action = self.config['action']
                if 'test_size' in self.config:
                    self.model.test_size = self.config['test_size']

                print(f"✅ Configured from existing config: action={getattr(self.model, 'action', 'Unknown')}, "
                      f"target={getattr(self.model, 'target_column', 'Unknown')}")
            else:
                print("❌ Could not load data from config - CSV file path not found or invalid")

        except Exception as e:
            print(f"❌ Error loading data from config: {e}")

    def _load_config(self, dataset_name: str) -> Dict:
        """Load configuration from file - uses original dataset name"""
        config_path = f"{dataset_name}.conf"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"✅ Loaded configuration: {config_path}")

            # Debug: Show what config we found
            if 'csv_file_path' in config:
                print(f"   CSV file: {config['csv_file_path']}")
            if 'action' in config:
                print(f"   Action: {config['action']}")
            if 'target_column' in config:
                print(f"   Target column: {config['target_column']}")

            return config

        print(f"⚠️ No configuration file found: {config_path}")
        return {}

    def _update_config_file(self):
        """Update the dataset configuration file with adaptive learning settings - keep original file"""
        config_path = f"{self.dataset_name}.conf"  # Use original config file name

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
                'max_adaptive_rounds': self.adaptive_config.get('max_adaptive_rounds', 10),
                'patience': self.adaptive_config.get('patience', 3),
                'min_improvement': self.adaptive_config.get('min_improvement', 0.001)
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

            # Ensure adaptive_visualization section exists
            if 'adaptive_visualization' not in config:
                config['adaptive_visualization'] = {}

            # Update adaptive visualization settings
            config['adaptive_visualization'].update({
                'enabled': self.adaptive_viz_config.get('enabled', True),
                'output_dir': self.adaptive_viz_config.get('output_dir', 'adaptive_visualizations'),
                'create_final_visualizations': self.adaptive_viz_config.get('create_final_visualizations', True),
                'save_indices_on_improvement': self.adaptive_viz_config.get('save_indices_on_improvement', True)
            })

            # Save updated config to original file
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)

            print(f"✅ Updated configuration file: {config_path}")

        except Exception as e:
            print(f"⚠️  Warning: Could not update config file: {str(e)}")

    def _get_device_type(self) -> str:
        """Get the device type (CPU/GPU) and specific model"""
        try:
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown GPU"
                return f"GPU: {gpu_name}"
            else:
                return "CPU"
        except:
            return "Unknown Device"

    def _initialize_adaptive_visualizer(self):
        """Initialize the adaptive visualizer if enabled"""
        if self.adaptive_viz_config.get('enabled', True):
            try:
                self.adaptive_visualizer = DBNNVisualizer(
                    self.model,
                    output_dir=self.adaptive_viz_config.get('output_dir', 'adaptive_visualizations'),
                    enabled=True
                )
                # Configure with adaptive-specific settings
                viz_config = {
                    'epoch_interval': 1,
                    'max_feature_combinations': 10,
                    'create_animations': False,
                    'interactive_mode': True,
                    'auto_open_browser': False
                }
                self.adaptive_visualizer.configure(viz_config)
                print("✓ Adaptive visualizer initialized")
            except Exception as e:
                print(f"⚠️ Could not initialize adaptive visualizer: {e}")
                self.adaptive_visualizer = None

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

        print("\n🎨 Adaptive Visualization Settings:")
        print("=" * 50)
        for key, value in self.adaptive_viz_config.items():
            print(f"  {key:30}: {value}")

        print("\n💻 Device Information:")
        print("=" * 50)
        print(f"  {'Device Type':30}: {self.device_type}")
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
            patience = int(input(f"Patience for early stopping [{self.adaptive_config.get('patience', 3)}]: ")
                         or self.adaptive_config.get('patience', 3))
            min_improvement = float(input(f"Minimum improvement to reset patience [{self.adaptive_config.get('min_improvement', 0.001)}]: ")
                                  or self.adaptive_config.get('min_improvement', 0.001))

            # Update settings
            self.adaptive_config.update({
                'initial_samples_per_class': initial_samples,
                'margin': margin,
                'max_adaptive_rounds': max_rounds,
                'patience': patience,
                'min_improvement': min_improvement
            })

            # Update config file
            self._update_config_file()

            print("✅ Settings updated successfully!")
            self.show_adaptive_settings()

        except ValueError:
            print("❌ Invalid input. Settings not changed.")

    def _save_best_indices(self, round_num: int):
        """Save the best training and test indices to file"""
        if not self.adaptive_viz_config.get('save_indices_on_improvement', True):
            return

        indices_file = f"{self.dataset_name}_best_indices_round_{round_num}.json"

        indices_data = {
            'round': round_num,
            'best_accuracy': float(self.best_accuracy),
            'training_indices': [int(idx) for idx in self.best_training_indices],
            'test_indices': [int(idx) for idx in self.best_test_indices],
            'training_size': len(self.best_training_indices),
            'test_size': len(self.best_test_indices),
            'timestamp': datetime.now().isoformat(),
            'class_distribution_training': np.bincount(self.y_full[self.best_training_indices]).tolist(),
            'class_distribution_test': np.bincount(self.y_full[self.best_test_indices]).tolist(),
            'device_type': self.device_type
        }

        with open(indices_file, 'w') as f:
            json.dump(indices_data, f, indent=4)

        print(f"💾 Saved best indices to: {indices_file}")

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

        # Initialize best indices
        self.best_training_indices = self.training_indices.copy()
        self.best_test_indices = self.test_indices.copy()

        # Record initial statistics
        self._record_round_statistics(0, X_train, y_train, X_test, y_test, 0.0, [])

        return X_train, y_train, X_test, y_test

    def _record_round_statistics(self, round_num, X_train, y_train, X_test, y_test,
                               accuracy, samples_added):
        """Record statistics for the current round with timing information"""
        current_time = datetime.now()
        round_start_time = getattr(self, 'round_start_time', self.start_time)
        round_elapsed = (current_time - round_start_time).total_seconds()

        round_stat = {
            'round': round_num,
            'training_size': len(X_train),
            'test_size': len(X_test),
            'accuracy': accuracy,
            'samples_added': len(samples_added),
            'class_distribution_train': np.bincount(y_train).tolist(),
            'class_distribution_test': np.bincount(y_test).tolist(),
            'timestamp': current_time.isoformat(),
            'round_start_time': round_start_time.isoformat(),
            'round_end_time': current_time.isoformat(),
            'round_duration_seconds': round_elapsed,
            'total_elapsed_time': (current_time - self.start_time).total_seconds(),
            'patience_counter': self.patience_counter,
            'device_type': self.device_type
        }
        self.round_stats.append(round_stat)

        # Print round statistics
        print(f"\n📊 Round {round_num} Statistics:")
        print(f"   Training samples: {round_stat['training_size']}")
        print(f"   Test samples: {round_stat['test_size']}")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Samples added: {round_stat['samples_added']}")
        print(f"   Round duration: {round_elapsed:.2f}s")
        print(f"   Total elapsed time: {round_stat['total_elapsed_time']:.2f}s")
        print(f"   Device: {self.device_type}")
        print(f"   Patience counter: {self.patience_counter}/{self.adaptive_config.get('patience', 3)}")

    def _check_early_stopping(self, current_accuracy: float, round_num: int) -> bool:
        """Check if early stopping conditions are met using adaptive patience"""
        patience = self.adaptive_config.get('patience', 3)
        min_improvement = self.adaptive_config.get('min_improvement', 0.001)

        # For all subsequent rounds, check for improvement
        if current_accuracy > self.best_accuracy + min_improvement:
            # Significant improvement - reset patience
            improvement = current_accuracy - self.best_accuracy
            self.best_accuracy = current_accuracy
            self.best_accuracy_round = round_num
            self.patience_counter = 0
            print(f"🎯 New best accuracy: {current_accuracy:.4f} (improvement: {improvement:.4f})")

            # Save best indices
            self.best_training_indices = self.training_indices.copy()
            self.best_test_indices = self.test_indices.copy()
            self._save_best_indices(round_num)

            return False
        else:
            # No significant improvement - increment patience
            self.patience_counter += 1
            improvement_needed = (self.best_accuracy + min_improvement) - current_accuracy
            print(f"⏳ No improvement - Patience: {self.patience_counter}/{patience}")
            print(f"   Need improvement of {improvement_needed:.4f} to reset patience")

            if self.patience_counter >= patience:
                print(f"🛑 Early stopping triggered after {patience} rounds without improvement")
                print(f"🏆 Best accuracy {self.best_accuracy:.4f} achieved at round {self.best_accuracy_round}")
                return True

        return False

    def _create_final_visualizations(self, X_train_best: np.ndarray, y_train_best: np.ndarray,
                                   X_test_best: np.ndarray, y_test_best: np.ndarray):
        """Create final visualizations using the adaptive visualizer"""
        if not self.adaptive_viz_config.get('create_final_visualizations', True):
            return

        if self.adaptive_visualizer is None:
            print("⚠️ Adaptive visualizer not available for final visualizations")
            return

        try:
            print("\n🎨 Creating final adaptive learning visualizations...")

            # Set data context for the visualizer
            self.adaptive_visualizer.set_data_context(
                X_train=X_train_best,
                y_train=y_train_best,
                feature_names=[f'Feature_{i}' for i in range(X_train_best.shape[1])],
                class_names=[str(cls) for cls in np.unique(y_train_best)]
            )

            # Extract training history from round statistics
            training_accuracies = [stat['accuracy'] for stat in self.round_stats if 'accuracy' in stat]
            training_errors = [1 - acc for acc in training_accuracies] if training_accuracies else []

            # Use current weights for visualization
            if hasattr(self.model, 'current_W') and self.model.current_W is not None:
                self.adaptive_visualizer.finalize_visualizations(
                    self.model.current_W,
                    training_errors,
                    training_accuracies
                )

            print("✅ Final adaptive learning visualizations completed!")

        except Exception as e:
            print(f"⚠️ Error creating final visualizations: {e}")

    def _generate_adaptive_learning_report(self, final_weights: np.ndarray):
        """Generate comprehensive visualization report for adaptive learning"""
        if not self.adaptive_viz_config.get('enabled', True):
            print("⚠️ Visualization disabled - skipping report generation")
            return

        if self.adaptive_visualizer is None:
            print("⚠️ Adaptive visualizer not available - skipping report generation")
            return

        try:
            print("\n📊 Generating comprehensive adaptive learning report...")

            # Extract training history from round statistics
            training_accuracies = [stat['accuracy'] for stat in self.round_stats if 'accuracy' in stat]
            training_errors = [1 - acc for acc in training_accuracies] if training_accuracies else []

            # Set data context for the visualizer if not already set
            if self.X_full is not None and self.y_full is not None:
                X_train_best = self.X_full[self.best_training_indices]
                y_train_best = self.y_full[self.best_training_indices]

                self.adaptive_visualizer.set_data_context(
                    X_train=X_train_best,
                    y_train=y_train_best,
                    feature_names=[f'Feature_{i}' for i in range(X_train_best.shape[1])],
                    class_names=[str(cls) for cls in np.unique(y_train_best)]
                )

            # Finalize visualizations with comprehensive report
            self.adaptive_visualizer.finalize_visualizations(
                final_weights,
                training_errors,
                training_accuracies
            )

            # Generate additional adaptive-specific visualizations
            self._create_adaptive_specific_visualizations(final_weights, training_accuracies)

            print("✅ Comprehensive adaptive learning report generated!")

        except Exception as e:
            print(f"⚠️ Error generating adaptive learning report: {e}")

    def _create_adaptive_specific_visualizations(self, final_weights: np.ndarray, training_accuracies: List[float]):
        """Create visualizations specific to adaptive learning process"""
        if not self.adaptive_viz_config.get('enabled', True) or not self.adaptive_visualizer:
            return

        try:
            print("Creating adaptive learning specific visualizations...")

            # Create adaptive learning progression plot
            self._plot_adaptive_learning_progression()

            # Create sample selection analysis
            self._plot_sample_selection_analysis()

            # Create round-by-round comparison
            self._create_round_comparison_visualization(final_weights)

        except Exception as e:
            print(f"⚠️ Error creating adaptive-specific visualizations: {e}")

    def _plot_adaptive_learning_progression(self):
        """Create detailed plots showing adaptive learning progression"""
        if len(self.round_stats) < 2:
            return

        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            rounds = [stat['round'] for stat in self.round_stats]
            accuracies = [stat['accuracy'] for stat in self.round_stats]
            training_sizes = [stat['training_size'] for stat in self.round_stats]
            round_durations = [stat.get('round_duration_seconds', 0) for stat in self.round_stats]
            samples_added = [stat.get('samples_added', 0) for stat in self.round_stats]

            # Create comprehensive adaptive learning dashboard
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    'Accuracy Progression by Round',
                    'Training Set Growth',
                    'Round Duration',
                    'Samples Added per Round',
                    'Cumulative Training Time',
                    'Accuracy vs Training Size'
                ),
                specs=[[{"type": "scatter"}, {"type": "scatter"}],
                       [{"type": "bar"}, {"type": "bar"}],
                       [{"type": "scatter"}, {"type": "scatter"}]]
            )

            # Accuracy progression
            fig.add_trace(go.Scatter(
                x=rounds, y=accuracies, mode='lines+markers', name='Accuracy',
                line=dict(color='blue', width=3), marker=dict(size=8)
            ), 1, 1)

            # Training set growth
            fig.add_trace(go.Scatter(
                x=rounds, y=training_sizes, mode='lines+markers', name='Training Size',
                line=dict(color='green', width=3), marker=dict(size=8)
            ), 1, 2)

            # Round duration
            fig.add_trace(go.Bar(
                x=rounds, y=round_durations, name='Round Duration',
                marker_color='orange'
            ), 2, 1)

            # Samples added
            fig.add_trace(go.Bar(
                x=rounds, y=samples_added, name='Samples Added',
                marker_color='red'
            ), 2, 2)

            # Cumulative time
            cumulative_time = [stat.get('total_elapsed_time', 0) for stat in self.round_stats]
            fig.add_trace(go.Scatter(
                x=rounds, y=cumulative_time, mode='lines', name='Cumulative Time',
                line=dict(color='purple', width=3)
            ), 3, 1)

            # Accuracy vs training size
            fig.add_trace(go.Scatter(
                x=training_sizes, y=accuracies, mode='lines+markers', name='Accuracy vs Size',
                line=dict(color='brown', width=2), marker=dict(size=6)
            ), 3, 2)

            fig.update_layout(
                height=1000,
                title_text=f"Adaptive Learning Analysis - {self.dataset_name}",
                showlegend=True
            )

            # Save the plot
            import plotly.io as pio
            filepath = os.path.join(self.adaptive_viz_config.get('output_dir', 'adaptive_visualizations'),
                                   'adaptive_learning_dashboard.html')
            pio.write_html(fig, filepath)
            print(f"✓ Adaptive learning dashboard saved: {filepath}")

        except Exception as e:
            print(f"⚠️ Error creating adaptive progression plot: {e}")

    def _plot_sample_selection_analysis(self):
        """Analyze and visualize sample selection patterns"""
        if not hasattr(self, 'X_full') or self.X_full is None:
            return

        try:
            import plotly.graph_objects as go
            from sklearn.decomposition import PCA

            # Use PCA to visualize sample selection in 2D
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(self.X_full)

            fig = go.Figure()

            # Plot all samples
            fig.add_trace(go.Scatter(
                x=X_pca[:, 0], y=X_pca[:, 1],
                mode='markers',
                marker=dict(size=3, color='lightgray', opacity=0.3),
                name='All Samples',
                text=[f"Sample {i}" for i in range(len(X_pca))]
            ))

            # Plot initial training samples
            if hasattr(self, 'best_training_indices') and self.best_training_indices:
                initial_indices = self.best_training_indices[:self.adaptive_config.get('initial_samples_per_class', 10)]
                fig.add_trace(go.Scatter(
                    x=X_pca[initial_indices, 0], y=X_pca[initial_indices, 1],
                    mode='markers',
                    marker=dict(size=8, color='blue', symbol='circle'),
                    name='Initial Training Samples',
                    text=[f"Initial Sample {i}" for i in initial_indices]
                ))

            # Plot final training samples
            if hasattr(self, 'best_training_indices') and self.best_training_indices:
                fig.add_trace(go.Scatter(
                    x=X_pca[self.best_training_indices, 0], y=X_pca[self.best_training_indices, 1],
                    mode='markers',
                    marker=dict(size=6, color='red', symbol='star'),
                    name='Final Training Samples',
                    text=[f"Final Sample {i}" for i in self.best_training_indices]
                ))

            fig.update_layout(
                title="Sample Selection Pattern Analysis (PCA)",
                xaxis_title="Principal Component 1",
                yaxis_title="Principal Component 2",
                showlegend=True
            )

            filepath = os.path.join(self.adaptive_viz_config.get('output_dir', 'adaptive_visualizations'),
                                   'sample_selection_analysis.html')
            import plotly.io as pio
            pio.write_html(fig, filepath)
            print(f"✓ Sample selection analysis saved: {filepath}")

        except Exception as e:
            print(f"⚠️ Error creating sample selection analysis: {e}")

    def _create_round_comparison_visualization(self, final_weights: np.ndarray):
        """Create visualization comparing different rounds"""
        if not self.adaptive_visualizer or len(self.round_stats) < 2:
            return

        try:
            # This would be an extension that compares weight distributions across rounds
            # For now, we'll create a simple round comparison
            print("✓ Round comparison visualization placeholder created")

        except Exception as e:
            print(f"⚠️ Error creating round comparison: {e}")

    def _plot_confusion_matrix_comparison(self, y_true, y_pred, round_num, previous_cm=None):
        """Plot confusion matrix with progress indicators"""
        if not self.stats_config.get('enable_confusion_matrix', True):
            return

        current_cm = confusion_matrix(y_true, y_pred)
        n_classes = len(np.unique(y_true))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Current confusion matrix
        sns.heatmap(current_cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_title(f'Round {round_num} - Confusion Matrix\nDevice: {self.device_type}')
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
        round_durations = [stat['round_duration_seconds'] for stat in self.round_stats]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

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

        # Round duration
        ax3.bar(rounds, round_durations, color='lightblue', alpha=0.7)
        ax3.set_xlabel('Round')
        ax3.set_ylabel('Duration (seconds)')
        ax3.set_title('Round Duration')
        ax3.grid(True, alpha=0.3)

        # Cumulative time
        cumulative_times = [stat['total_elapsed_time'] for stat in self.round_stats]
        ax4.plot(rounds, cumulative_times, '^-', color='green', linewidth=2, markersize=8)
        ax4.set_xlabel('Round')
        ax4.set_ylabel('Cumulative Time (seconds)')
        ax4.set_title('Cumulative Training Time')
        ax4.grid(True, alpha=0.3)

        # Add device info to the plot
        plt.suptitle(f'Adaptive Learning Progress - Device: {self.device_type}', fontsize=14, y=1.02)

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

    def adaptive_learning_cycle(self, X: np.ndarray, y: np.ndarray):
        """Main adaptive learning cycle with enhanced statistics"""
        max_rounds = self.adaptive_config.get('max_adaptive_rounds', 10)

        # Store the full dataset for later use
        self.X_full = X
        self.y_full = y

        # Track adaptive learning start time
        self.adaptive_start_time = datetime.now()
        print(f"\n🚀 Starting Adaptive Learning at: {self.adaptive_start_time}")
        print(f"💻 Device: {self.device_type}")

        # Prepare initial data
        X_train, y_train, X_test, y_test = self.prepare_adaptive_data(X, y)

        # Initialize best weights tracking
        self.best_weights = None
        self.best_accuracy = 0.0
        self.best_training_indices = self.training_indices.copy()
        self.best_test_indices = self.test_indices.copy()

        for round_num in range(1, max_rounds + 1):
            self.adaptive_round = round_num
            self.round_start_time = datetime.now()  # Track round start time

            print(f"\n{'='*60}")
            print(f"=== Adaptive Learning Round {round_num}/{max_rounds} ===")
            print(f"{'='*60}")

            # Train model on current training set and get best accuracy from DBNN
            print("Training model...")
            current_accuracy, current_weights = self._train_with_custom_data(X_train, y_train, X_test, y_test)

            print(f"Current round accuracy: {current_accuracy:.4f}")
            print(f"Best accuracy so far: {self.best_accuracy:.4f}")

            # Check if this is the best accuracy so far
            if current_accuracy > self.best_accuracy:
                self.best_accuracy = current_accuracy
                self.best_weights = current_weights.copy() if current_weights is not None else None
                self.best_training_indices = self.training_indices.copy()
                self.best_test_indices = self.test_indices.copy()
                self.best_accuracy_round = round_num
                self.patience_counter = 0  # Reset patience counter
                print(f"🎯 New best accuracy: {current_accuracy:.4f}")

                # Save best indices
                self._save_best_indices(round_num)
            else:
                # No improvement - increment patience counter
                self.patience_counter += 1

            # Check for early stopping based on patience
            if self._check_early_stopping(current_accuracy, round_num):
                break

            # Record statistics
            self._record_round_statistics(round_num, X_train, y_train, X_test, y_test,
                                        current_accuracy, [])

            # Evaluate on test set with current model
            test_posteriors = self._compute_batch_posterior_for_indices(X, self.test_indices)
            test_predictions = np.argmax(test_posteriors, axis=1)

            # Ensure y_test is integer type (not float)
            y_test = y_test.astype(int)

            # Print detailed classification report
            self._print_detailed_classification_report(y_test, test_predictions, round_num)

            # Plot confusion matrix comparison
            current_confusion = self._plot_confusion_matrix_comparison(
                y_test, test_predictions, round_num, self.previous_confusion
            )
            self.previous_confusion = current_confusion

            # Check stopping conditions
            if current_accuracy >= 1.0 or len(self.test_indices) == 0:
                print("🎯 Stopping condition met!")
                break

            # Find samples to add based on margin criteria
            samples_to_add = self._find_margin_based_samples(
                X, y, test_predictions, test_posteriors
            )

            if not samples_to_add:
                print("⏹️  No more informative samples found. Stopping.")
                break

            # Add samples to training set
            print(f"📥 Adding {len(samples_to_add)} informative samples to training set")
            self.training_indices.extend(samples_to_add)
            self.test_indices = [idx for idx in self.test_indices if idx not in samples_to_add]

            # Update datasets for next round
            X_train = X[self.training_indices]
            y_train = y[self.training_indices]
            X_test = X[self.test_indices]
            y_test = y[self.test_indices]

            # Update statistics with samples added
            self.round_stats[-1]['samples_added'] = len(samples_to_add)

            print(f"📊 New training set size: {len(X_train)}")
            print(f"📊 New test set size: {len(X_test)}")

        # Final evaluation with best model
        adaptive_end_time = datetime.now()
        total_adaptive_time = (adaptive_end_time - self.adaptive_start_time).total_seconds()

        print(f"\n{'='*60}")
        print("=== Final Results ===")
        print(f"{'='*60}")
        print(f"🏆 Best accuracy achieved: {self.best_accuracy:.4f}")
        print(f"📦 Final training set size: {len(self.best_training_indices)}")
        print(f"🕒 Adaptive learning started: {self.adaptive_start_time}")
        print(f"🕒 Adaptive learning ended: {adaptive_end_time}")
        print(f"⏱️  Total adaptive learning time: {total_adaptive_time:.2f}s")
        print(f"💻 Device: {self.device_type}")

        # Load best weights for final model
        if self.best_weights is not None:
            self.model.current_W = self.best_weights
            print("✅ Loaded best weights from round", self.best_accuracy_round)

        # Plot overall progress
        self._plot_training_progress()

        # Save final statistics
        self._save_final_statistics()

        # Generate comprehensive adaptive learning report
        print("\n📋 Generating comprehensive adaptive learning report...")
        self._generate_adaptive_learning_report(self.best_weights)

        # Create final visualizations using best data
        X_train_best = X[self.best_training_indices]
        y_train_best = y[self.best_training_indices]
        X_test_best = X[self.best_test_indices]
        y_test_best = y[self.best_test_indices]

        self._create_final_visualizations(X_train_best, y_train_best, X_test_best, y_test_best)

        return self.best_accuracy, self.best_training_indices

    def _train_with_custom_data(self, X_train, y_train, X_test, y_test):
        """Train the model with custom data and return best accuracy and weights"""
        # Store original data and state
        original_data = self.model.data.copy()
        original_weights = getattr(self.model, 'current_W', None)
        original_best_weights = getattr(self.model, 'best_W', None)
        original_best_accuracy = getattr(self.model, 'best_accuracy', 0.0)

        # Create a temporary DataFrame with our custom training data
        column_names = list(self.model.data.columns)

        # For training data
        train_df = pd.DataFrame(X_train, columns=column_names[:-1])
        train_df[column_names[-1]] = y_train

        # Temporarily replace the model's data
        self.model.data = train_df

        # Disable pruning during adaptive learning
        original_pruning = getattr(self.model, 'pruning_enabled', True)
        self.model.set_pruning_enabled(False)

        # Reset the model's internal state for fresh training
        self.model._initialize_weights()
        self.model.best_accuracy = 0.0
        self.model.best_W = None

        # Train the model using its original method
        training_start_time = time.time()
        self.model.train()
        training_duration = time.time() - training_start_time

        # Get the best accuracy and weights achieved during this training session
        current_best_accuracy = getattr(self.model, 'best_accuracy', 0.0)
        current_best_weights = getattr(self.model, 'best_W', None)

        # If no best weights were saved during training, use current weights
        if current_best_weights is None:
            current_best_weights = getattr(self.model, 'current_W', None)

        print(f"Training completed in {training_duration:.2f}s. Best accuracy: {current_best_accuracy:.4f}")

        # Restore original data and state
        self.model.data = original_data
        self.model.set_pruning_enabled(original_pruning)

        # Store the trained weights for this round
        self._current_training_weights = current_best_weights

        # Return the best accuracy and weights from this training session
        return current_best_accuracy, current_best_weights

    def _compute_batch_posterior_for_indices(self, X: np.ndarray, indices: List[int]) -> np.ndarray:
        """Compute posteriors for specific indices using current training weights"""
        X_subset = X[indices]

        # Store original weights temporarily
        original_weights = getattr(self.model, 'current_W', None)

        try:
            # Use the weights from the current training session
            if hasattr(self, '_current_training_weights') and self._current_training_weights is not None:
                self.model.current_W = self._current_training_weights

            # Now compute the posterior with the correct weights
            return self.model._compute_batch_posterior(X_subset)
        finally:
            # Restore original weights
            self.model.current_W = original_weights

    def _save_final_statistics(self):
        """Save final statistics to file with comprehensive timing information"""
        stats_file = f"{self.dataset_name}_adaptive_stats.json"

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

        # Calculate final timing statistics
        end_time = datetime.now()
        total_time = (end_time - self.start_time).total_seconds()
        adaptive_time = (end_time - self.adaptive_start_time).total_seconds() if self.adaptive_start_time else total_time

        final_stats = {
            'dataset': self.dataset_name,
            'best_accuracy': float(self.best_accuracy),
            'final_training_size': int(len(self.best_training_indices)),
            'final_test_size': int(len(self.best_test_indices)),
            'best_accuracy_round': int(self.best_accuracy_round),
            'total_rounds': int(len(self.round_stats)),
            'device_type': self.device_type,
            'timing': {
                'start_time': self.start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'adaptive_start_time': self.adaptive_start_time.isoformat() if self.adaptive_start_time else None,
                'total_time_seconds': float(total_time),
                'adaptive_learning_time_seconds': float(adaptive_time),
                'average_round_time_seconds': float(np.mean([stat['round_duration_seconds'] for stat in self.round_stats])) if self.round_stats else 0.0
            },
            'round_statistics': convert_numpy_types(self.round_stats),
            'config': self.adaptive_config,
            'adaptive_visualization_config': self.adaptive_viz_config
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

        # FIX: Only transform, don't re-fit scaler on existing model
        X_scaled = self.model.scaler.transform(X)

        return X_scaled, y_encoded

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the best model from adaptive learning"""
        if self.best_weights is None:
            raise ValueError("No trained model found. Run adaptive_learning_cycle first.")

        # Store original weights
        original_weights = getattr(self.model, 'current_W', None)

        try:
            # Load best weights from adaptive learning
            self.model.current_W = self.best_weights

            # Scale the input data using the same scaler
            X_scaled = self.model.scaler.transform(X)

            # Make predictions
            posteriors = self.model._compute_batch_posterior(X_scaled)
            predictions = np.argmax(posteriors, axis=1)

            # Convert back to original labels
            predictions_original = self.model.label_encoder.inverse_transform(predictions)

            return predictions_original

        finally:
            # Restore original weights
            self.model.current_W = original_weights

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities using the best model from adaptive learning"""
        if self.best_weights is None:
            raise ValueError("No trained model found. Run adaptive_learning_cycle first.")

        # Store original weights
        original_weights = getattr(self.model, 'current_W', None)

        try:
            # Load best weights from adaptive learning
            self.model.current_W = self.best_weights

            # Scale the input data using the same scaler
            X_scaled = self.model.scaler.transform(X)

            # Get probabilities
            posteriors = self.model._compute_batch_posterior(X_scaled)

            return posteriors

        finally:
            # Restore original weights
            self.model.current_W = original_weights

    def save_adaptive_model(self, overwrite: bool = False):
        """Save the adaptive learning model state with proper naming"""
        if self.best_weights is None:
            print("❌ No trained model found to save")
            return

        # Check if model already exists
        model_filename = f"Model/{self.adaptive_model_name}_weights.json"
        if os.path.exists(model_filename) and not overwrite:
            response = input(f"⚠️ Adaptive model '{self.adaptive_model_name}' already exists. Overwrite? (y/N): ").strip().lower()
            if response != 'y':
                print("❌ Save cancelled")
                return

        # Prepare model state for saving
        model_state = {
            'version': 6,  # Mark as adaptive model
            'weights': self.best_weights.tolist(),
            'shape': self.best_weights.shape,
            'dataset_fingerprint': self.model._get_dataset_fingerprint(),
            'model_type': 'adaptive_histogram',
            'histogram_bins': getattr(self.model, 'histogram_bins', 64),
            'histogram_method': getattr(self.model, 'histogram_method', 'vectorized'),
            'adaptive_learning_info': {
                'best_accuracy': float(self.best_accuracy),
                'best_round': int(self.best_accuracy_round),
                'final_training_size': int(len(self.best_training_indices)),
                'adaptive_config': self.adaptive_config,
                'device_type': self.device_type
            },
            'scaler_mean': self.model.scaler.mean_.tolist() if hasattr(self.model.scaler, 'mean_') else [],
            'scaler_scale': self.model.scaler.scale_.tolist() if hasattr(self.model.scaler, 'scale_') else [],
            'label_encoder_classes': self.model.label_encoder.classes_.tolist() if hasattr(self.model, 'label_encoder') else [],
        }

        # Ensure Model directory exists
        os.makedirs('Model', exist_ok=True)

        # Save weights
        with open(model_filename, 'w') as f:
            json.dump(model_state, f, indent=2)

        # Save encoders
        encoder_filename = f"Model/{self.adaptive_model_name}_encoders.pkl"
        encoder_state = {
            'label_encoder': self.model.label_encoder,
            'categorical_encoders': getattr(self.model, 'categorical_encoders', {}),
            'feature_names': list(self.model.data.columns[:-1]) if hasattr(self.model, 'data') else []
        }

        import pickle
        with open(encoder_filename, 'wb') as f:
            pickle.dump(encoder_state, f)

        print(f"✅ Adaptive model saved to: {model_filename}")
        print(f"✅ Encoders saved to: {encoder_filename}")

    def load_adaptive_model(self) -> bool:
        """Load a previously saved adaptive learning model"""
        model_filename = f"Model/{self.adaptive_model_name}_weights.json"
        encoder_filename = f"Model/{self.adaptive_model_name}_encoders.pkl"

        if not os.path.exists(model_filename):
            print(f"❌ Adaptive model not found: {model_filename}")
            return False

        try:
            # Load weights
            with open(model_filename, 'r') as f:
                model_state = json.load(f)

            # Verify it's an adaptive model
            if model_state.get('model_type') != 'adaptive_histogram':
                print("❌ Loaded model is not an adaptive learning model")
                return False

            # Load weights
            self.best_weights = np.array(model_state['weights'])
            self.model.current_W = self.best_weights

            # Load adaptive learning info
            adaptive_info = model_state.get('adaptive_learning_info', {})
            self.best_accuracy = adaptive_info.get('best_accuracy', 0.0)
            self.best_accuracy_round = adaptive_info.get('best_round', 0)

            # Load encoders
            if os.path.exists(encoder_filename):
                import pickle
                with open(encoder_filename, 'rb') as f:
                    encoder_state = pickle.load(f)

                self.model.label_encoder = encoder_state['label_encoder']
                self.model.categorical_encoders = encoder_state.get('categorical_encoders', {})

            print(f"✅ Adaptive model loaded: {model_filename}")
            print(f"   Best accuracy: {self.best_accuracy:.4f}")
            print(f"   Best round: {self.best_accuracy_round}")
            print(f"   Training size: {adaptive_info.get('final_training_size', 'Unknown')}")

            return True

        except Exception as e:
            print(f"❌ Error loading adaptive model: {e}")
            return False

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
    """Main function for adaptive learning with proper model management"""
    dataset_name = select_dataset()

    # Use original dataset name for config and data, but adaptive name for model files
    adaptive_model_name = f"{dataset_name}_adaptive"
    adaptive_weights_file = f"Model/{adaptive_model_name}_weights.json"

    # Check for existing adaptive model
    if os.path.exists(adaptive_weights_file):
        use_existing = input(f"🔄 Found existing adaptive model for {dataset_name}. Use it? (y/N): ").strip().lower()
        if use_existing == 'y':
            adaptive_model = AdaptiveDBNN(dataset_name)
            if adaptive_model.load_adaptive_model():
                print("✅ Adaptive model loaded and ready for prediction")

                # Test prediction
                print("\n🧪 Testing prediction with loaded adaptive model...")
                X_full, y_full = adaptive_model.prepare_full_data()
                sample_size = min(10, len(X_full))
                predictions = adaptive_model.predict(X_full[:sample_size])
                print(f"Sample predictions: {predictions}")
                print(f"True labels: {y_full[:sample_size]}")

                return adaptive_model

    # Initialize new adaptive learning
    print(f"🚀 Initializing adaptive learning for {dataset_name}...")
    print(f"   Using existing configuration from: {dataset_name}.conf")

    adaptive_model = AdaptiveDBNN(dataset_name)

    # Verify configuration was loaded properly
    if hasattr(adaptive_model.model, 'data') and adaptive_model.model.data is not None:
        print(f"✅ Successfully loaded data: {adaptive_model.model.data.shape}")
    else:
        print("❌ Failed to load data from configuration")
        return None

    # Check action mode from config
    action = getattr(adaptive_model.model, 'action', 'train')
    print(f"📋 Action mode from config: {action}")

    # Check if standard model exists and ask if we should start from it
    standard_weights_file = f"Model/Best_{dataset_name}_weights.json"
    if os.path.exists(standard_weights_file):
        start_from_standard = input(f"📚 Found standard trained model. Start adaptive learning from it? (y/N): ").strip().lower()
        if start_from_standard == 'y':
            print("🔄 Loading standard model as starting point...")
            # The adaptive model will train from scratch but can benefit from existing config

    # Check if adaptive learning is enabled
    if not adaptive_model.adaptive_config.get('enable_adaptive', False):
        print("Adaptive learning not enabled in config. Using standard training.")
        # Fall back to standard DBNN training
        model = GPUDBNN(dataset_name)
        model.train()
        X_test, y_test = model._prepare_data()[1:3]
        model.predict(X_test)
        return model

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
        'best_accuracy': float(best_accuracy),
        'training_indices': [int(idx) for idx in best_indices],
        'training_set_size': int(len(best_indices)),
        'total_samples': int(len(X_full)),
        'device_type': adaptive_model.device_type,
        'total_time_seconds': float((datetime.now() - adaptive_model.start_time).total_seconds()),
        'adaptive_config': adaptive_model.adaptive_config,
        'adaptive_visualization_config': adaptive_model.adaptive_viz_config
    }

    results_file = f"{dataset_name}_adaptive_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"✅ Results saved to {results_file}")

    # Demonstrate prediction on entire dataset
    print("\n🎯 Making predictions on entire dataset with best adaptive model...")
    all_predictions = adaptive_model.predict(X_full)
    all_probabilities = adaptive_model.predict_proba(X_full)

    print(f"Predictions shape: {all_predictions.shape}")
    print(f"Probabilities shape: {all_probabilities.shape}")
    print(f"First 10 predictions: {all_predictions[:10]}")
    print(f"First 10 true labels: {y_full[:10]}")

    # Calculate final accuracy
    final_accuracy = accuracy_score(y_full, all_predictions)
    print(f"🏆 Final accuracy on entire dataset: {final_accuracy:.4f}")

    # Ask if user wants to save the final adaptive model
    save_model = input("\n💾 Save the final adaptive model? (y/N): ").strip().lower()
    if save_model == 'y':
        adaptive_model.save_adaptive_model()
        print("✅ Adaptive model saved for future predictions")

    # Final config update
    adaptive_model._update_config_file()

    return adaptive_model

if __name__ == "__main__":
    main()
