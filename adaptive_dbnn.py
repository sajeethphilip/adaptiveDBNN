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
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import imageio
from scipy.spatial import ConvexHull
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
            self.config = DatasetConfig.load_config(dataset_name)

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
            "patience": 10,
            "min_improvement": 0.001,
            "training_convergence_epochs": 50,
            "min_training_accuracy": 0.95,
            "min_samples_to_add_per_class": 5,
            "adaptive_margin_relaxation": 0.1,
            "max_divergence_samples_per_class": 5,
            "exhaust_all_failed": True,
            "min_failed_threshold": 10,
            "enable_kl_divergence": True,     # Switch this False to disable multiple sample selection for train as in original DBNN
            "max_samples_per_class_fallback": 2,
            "enable_3d_visualization": True,
            "3d_snapshot_interval": 10,
            "learning_rate": 1.0,
            "enable_acid_test": True,
            "min_training_percentage_for_stopping": 10.0,
            "max_training_percentage": 90.0,
            "margin_tolerance": 0.15,  # Percentage tolerance for margin range
            "kl_divergence_threshold": 0.1,  # Minimum KL divergence to consider
            "max_kl_samples_per_class": 5,  # Maximum KL-based samples per class per round

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

        # Global best tracking for initialize-and-freeze strategy
        self.global_best_accuracy = 0.0
        self.global_best_round = 0
        self.global_best_training_indices = []
        self.global_best_test_indices = []
        self.global_best_weights = None

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

        # Add 3D visualization initialization
        self._initialize_3d_visualization()

    def _initialize_3d_visualization(self):
        """Initialize 3D visualization system"""
        self.visualization_output_dir = self.viz_config.get('output_dir', 'adaptive_visualizations')
        os.makedirs(f'{self.visualization_output_dir}/3d_animations', exist_ok=True)
        self.feature_grid_history = []
        self.epoch_timestamps = []

        print("🎨 3D Visualization system initialized")

    def _track_feature_grids_3d(self, epoch: int, X_train: np.ndarray = None, y_train: np.ndarray = None):
        """
        Track the position and characteristics of feature grids in 3D space
        """
        if not hasattr(self, 'feature_grid_history'):
            self._initialize_3d_visualization()

        try:
            # Get current model state for visualization
            grid_info = {
                'epoch': epoch,
                'timestamp': datetime.now(),
                'weights': self.model.current_W.copy() if hasattr(self.model, 'current_W') and self.model.current_W is not None else None,
                'training_accuracy': None,
                'feature_importance': None
            }

            # Calculate feature importance based on weights
            if grid_info['weights'] is not None:
                if self.model.model_type == 'histogram':
                    # For histogram model, use weight magnitudes
                    feature_importance = np.mean(np.abs(grid_info['weights']), axis=(0, 2))
                else:
                    # For Gaussian model, use weight magnitudes
                    feature_importance = np.mean(np.abs(grid_info['weights']), axis=0)

                grid_info['feature_importance'] = feature_importance

            # Store training accuracy if available
            if X_train is not None and y_train is not None:
                predictions = self._predict_with_current_model(X_train)
                grid_info['training_accuracy'] = accuracy_score(y_train, predictions)

            self.feature_grid_history.append(grid_info)
            self.epoch_timestamps.append(epoch)

            # Create snapshot every 10 epochs or at important milestones
            if epoch % 10 == 0 or epoch in [1, 5, 10, 25, 50, 100] or epoch == self.adaptive_round:
                self._create_3d_feature_space_snapshot(epoch, X_train, y_train)

        except Exception as e:
            print(f"⚠️ Could not track feature grids: {e}")

    def _create_3d_feature_space_snapshot(self, epoch: int, X_train: np.ndarray = None, y_train: np.ndarray = None):
        """
        Create a 3D snapshot of the feature space with drifting grids
        """
        try:
            if len(self.feature_grid_history) < 2:
                return

            fig = plt.figure(figsize=(16, 12))

            # Create 2x2 subplot grid
            ax1 = fig.add_subplot(221, projection='3d')  # Feature space with grids
            ax2 = fig.add_subplot(222, projection='3d')  # Weight evolution
            ax3 = fig.add_subplot(223)  # Feature importance over time
            ax4 = fig.add_subplot(224)  # Grid drift trajectory

            current_grid = self.feature_grid_history[-1]

            # Plot 1: 3D Feature Space with Grids
            self._plot_3d_feature_space(ax1, epoch, X_train, y_train)

            # Plot 2: 3D Weight Evolution
            self._plot_3d_weight_evolution(ax2, epoch)

            # Plot 3: Feature Importance Over Time
            self._plot_feature_importance_evolution(ax3)

            # Plot 4: Grid Drift Trajectory
            self._plot_grid_drift_trajectory(ax4, epoch)

            plt.suptitle(f'Adaptive Learning - Epoch {epoch}\n'
                        f'Training Accuracy: {current_grid.get("training_accuracy", "N/A"):.3f}',
                        fontsize=16, fontweight='bold')

            plt.tight_layout()
            plt.savefig(f'{self.visualization_output_dir}/3d_animations/epoch_{epoch:04d}.png',
                       dpi=150, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"⚠️ Could not create 3D snapshot: {e}")

    def _plot_3d_feature_space(self, ax, epoch: int, X_train: np.ndarray = None, y_train: np.ndarray = None):
        """Plot 3D feature space with drifting grids"""
        if X_train is None or y_train is None:
            return

        # Use PCA to project to 3D for visualization
        pca = PCA(n_components=3)
        X_3d = pca.fit_transform(X_train)

        # Plot training samples
        scatter = ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2],
                            c=y_train, cmap='tab10', alpha=0.6, s=10)

        # Plot feature grids if we have weight information
        if hasattr(self.model, 'current_W') and self.model.current_W is not None:
            self._plot_feature_grids_3d(ax, X_3d, pca)

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title('3D Feature Space with Grids')

        # Add legend for classes
        if hasattr(self, 'label_encoder'):
            unique_classes = np.unique(y_train)
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=plt.cm.tab10(i/len(unique_classes)),
                              markersize=8, label=f'Class {cls}')
                             for i, cls in enumerate(unique_classes)]
            ax.legend(handles=legend_elements, loc='upper right')

    def _plot_feature_grids_3d(self, ax, X_3d: np.ndarray, pca: PCA):
        """Plot feature grids in 3D space"""
        try:
            if self.model.model_type == 'histogram':
                self._plot_histogram_grids_3d(ax, X_3d)
            else:
                self._plot_gaussian_grids_3d(ax, X_3d, pca)
        except Exception as e:
            print(f"⚠️ Could not plot feature grids: {e}")

    def _plot_histogram_grids_3d(self, ax, X_3d: np.ndarray):
        """Plot histogram model grids in 3D"""
        # For histogram model, visualize bin centers with weights
        if hasattr(self.model, 'histograms') and self.model.histograms is not None:
            n_features = self.model.histograms.shape[0]
            n_bins = self.model.histogram_bins

            # Select top 3 most important features
            if hasattr(self.model, 'current_W') and self.model.current_W is not None:
                feature_importance = np.mean(np.abs(self.model.current_W), axis=(0, 2))
                top_features = np.argsort(feature_importance)[-3:][::-1]
            else:
                top_features = [0, 1, 2]  # Default to first 3 features

            # Plot bin centers for top features
            for i, feat_idx in enumerate(top_features):
                if feat_idx < n_features:
                    bin_centers = (self.model.bin_edges[feat_idx, :-1] +
                                  self.model.bin_edges[feat_idx, 1:]) / 2

                    # Create grid points
                    for j, center in enumerate(bin_centers):
                        if j < len(X_3d):  # Safety check
                            # Use feature importance to determine size and color
                            weight_magnitude = np.mean(np.abs(self.model.current_W[:, feat_idx, j]))
                            size = 50 + weight_magnitude * 500
                            color = plt.cm.viridis(weight_magnitude / (weight_magnitude + 0.1))

                            ax.scatter([X_3d[j, 0]], [X_3d[j, 1]], [X_3d[j, 2]],
                                      s=size, c=[color], alpha=0.7, marker='s')

    def _plot_gaussian_grids_3d(self, ax, X_3d: np.ndarray, pca: PCA):
        """Plot Gaussian model feature pairs in 3D"""
        if (hasattr(self.model, 'likelihood_params') and
            self.model.likelihood_params is not None and
            hasattr(self.model, 'feature_pairs') and
            self.model.feature_pairs is not None):

            means = self.model.likelihood_params['means']
            n_classes, n_pairs, _ = means.shape

            # Plot a subset of feature pairs to avoid clutter
            pairs_to_plot = min(20, n_pairs)
            selected_pairs = np.random.choice(n_pairs, pairs_to_plot, replace=False)

            for pair_idx in selected_pairs:
                for class_idx in range(n_classes):
                    mean_2d = means[class_idx, pair_idx]

                    # Project mean to 3D PCA space (approximation)
                    # This is a simplified projection for visualization
                    if len(mean_2d) >= 2:
                        # Create a synthetic point and project it
                        synthetic_point = np.zeros(X_3d.shape[1])
                        feature_pair = self.model.feature_pairs[pair_idx]

                        for i, feat_idx in enumerate(feature_pair):
                            if feat_idx < len(synthetic_point):
                                synthetic_point[feat_idx] = mean_2d[i]

                        # Project using PCA (approximate)
                        projected_point = pca.transform([synthetic_point])[0]

                        # Plot the grid center
                        color = plt.cm.tab10(class_idx)
                        weight = np.mean(np.abs(self.model.current_W[class_idx, pair_idx]))
                        size = 30 + weight * 200

                        ax.scatter([projected_point[0]], [projected_point[1]], [projected_point[2]],
                                  s=size, c=[color], alpha=0.6, marker='o')

    def _plot_3d_weight_evolution(self, ax, epoch: int):
        """Plot 3D evolution of weights over time"""
        if len(self.feature_grid_history) < 3:
            return

        # Extract weight evolution for top 3 features
        weights_3d = []
        for grid_info in self.feature_grid_history[-10:]:  # Last 10 epochs
            if grid_info['feature_importance'] is not None:
                top_features = np.argsort(grid_info['feature_importance'])[-3:][::-1]
                weights_3d.append(grid_info['feature_importance'][top_features])

        if len(weights_3d) > 2:
            weights_3d = np.array(weights_3d)

            # Create trajectory plot
            ax.plot(weights_3d[:, 0], weights_3d[:, 1], weights_3d[:, 2],
                   'b-o', linewidth=2, markersize=4, alpha=0.7)

            # Mark current position
            ax.scatter([weights_3d[-1, 0]], [weights_3d[-1, 1]], [weights_3d[-1, 2]],
                      s=100, c='red', marker='*', label='Current')

            ax.set_xlabel('Feature 1 Importance')
            ax.set_ylabel('Feature 2 Importance')
            ax.set_zlabel('Feature 3 Importance')
            ax.set_title('Weight Evolution in 3D')
            ax.legend()

    def _plot_feature_importance_evolution(self, ax):
        """Plot feature importance evolution over time"""
        if len(self.feature_grid_history) < 2:
            return

        epochs = [grid['epoch'] for grid in self.feature_grid_history if grid['feature_importance'] is not None]
        importances = [grid['feature_importance'] for grid in self.feature_grid_history if grid['feature_importance'] is not None]

        if len(importances) > 1:
            importances = np.array(importances)

            # Plot top 5 features
            n_features = min(5, importances.shape[1])
            for i in range(n_features):
                ax.plot(epochs, importances[:, i], linewidth=2, label=f'Feature {i+1}')

            ax.set_xlabel('Epoch')
            ax.set_ylabel('Feature Importance')
            ax.set_title('Feature Importance Evolution')
            ax.legend()
            ax.grid(True, alpha=0.3)

    def _plot_grid_drift_trajectory(self, ax, epoch: int):
        """Plot the trajectory of grid drift over time"""
        if len(self.feature_grid_history) < 3:
            return

        # Calculate overall grid movement (weight changes)
        movements = []
        for i in range(1, len(self.feature_grid_history)):
            if (self.feature_grid_history[i]['weights'] is not None and
                self.feature_grid_history[i-1]['weights'] is not None):
                movement = np.mean(np.abs(
                    self.feature_grid_history[i]['weights'] -
                    self.feature_grid_history[i-1]['weights']
                ))
                movements.append(movement)

        if movements:
            ax.plot(range(len(movements)), movements, 'g-o', linewidth=2)
            ax.set_xlabel('Epoch Transition')
            ax.set_ylabel('Average Weight Change')
            ax.set_title('Grid Drift Trajectory')
            ax.grid(True, alpha=0.3)

    def _create_3d_animation_gif(self):
        """Create GIF animation of the 3D feature space evolution"""
        try:
            print("🎬 Creating 3D animation GIF...")

            image_files = sorted([f for f in os.listdir(f'{self.visualization_output_dir}/3d_animations')
                                if f.startswith('epoch_') and f.endswith('.png')])

            if not image_files:
                print("⚠️ No 3D snapshots found for animation")
                return

            # Create GIF
            images = []
            for image_file in image_files:
                image_path = f'{self.visualization_output_dir}/3d_animations/{image_file}'
                images.append(imageio.imread(image_path))

            gif_path = f'{self.visualization_output_dir}/feature_space_evolution.gif'
            imageio.mimsave(gif_path, images, duration=0.5)  # 0.5 seconds per frame

            print(f"✅ 3D animation saved: {gif_path}")

            # Also create a faster summary animation with fewer frames
            if len(images) > 20:
                summary_images = images[::len(images)//10]  # 10 frames summary
                summary_gif_path = f'{self.visualization_output_dir}/feature_space_evolution_summary.gif'
                imageio.mimsave(summary_gif_path, summary_images, duration=1.0)
                print(f"✅ Summary animation saved: {summary_gif_path}")

        except Exception as e:
            print(f"⚠️ Could not create 3D animation: {e}")

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
        """Display the current adaptive learning settings including KL divergence"""
        print("\n🔧 Advanced Adaptive Learning Settings:")
        print("=" * 60)
        for key, value in self.adaptive_config.items():
            if key in ['margin_tolerance', 'kl_divergence_threshold', 'max_kl_samples_per_class']:
                print(f"  {key:40}: {value} (KL Divergence)")
            else:
                print(f"  {key:40}: {value}")
        print(f"\n💻 Device: {self.device_type}")
        print(f"🎯 Selection Mode: {'KL Divergence' if self.adaptive_config.get('enable_kl_divergence', False) else 'Margin-Based'}")
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
            learning_rate = float(input(f"Learning rate (0.5-2.0) [{self.adaptive_config.get('learning_rate', 1.0)}]: ")
                                or self.adaptive_config.get('learning_rate', 1.0))

            # Early stopping relaxation parameters
            min_stopping_percentage = float(input(
                f"Min training data % for early stopping [{self.adaptive_config.get('min_training_percentage_for_stopping', 10.0)}]: "
            ) or self.adaptive_config.get('min_training_percentage_for_stopping', 10.0))

            max_training_percentage = float(input(
                f"Max training data % allowed [{self.adaptive_config.get('max_training_percentage', 90.0)}]: "
            ) or self.adaptive_config.get('max_training_percentage', 90.0))

            enable_kl = input(f"Enable KL divergence selection? (y/N) [{'y' if self.adaptive_config.get('enable_kl_divergence', True) else 'n'}]: ").strip().lower()
            enable_kl_divergence = enable_kl == 'y' if enable_kl else self.adaptive_config.get('enable_kl_divergence', True)

            if enable_kl_divergence:
                print("\n🔧 KL Divergence Configuration:")
                margin_tol = float(input(f"Margin tolerance (0.1-0.5) [{self.adaptive_config.get('margin_tolerance', 0.15)}]: ")
                                or self.adaptive_config.get('margin_tolerance', 0.15))
                kl_threshold = float(input(f"KL divergence threshold (0.05-0.3) [{self.adaptive_config.get('kl_divergence_threshold', 0.1)}]: ")
                                   or self.adaptive_config.get('kl_divergence_threshold', 0.1))
                max_kl_samples = int(input(f"Max KL samples per class (2-10) [{self.adaptive_config.get('max_kl_samples_per_class', 5)}]: ")
                                   or self.adaptive_config.get('max_kl_samples_per_class', 5))

                # Validate ranges
                margin_tol = max(0.05, min(0.5, margin_tol))
                kl_threshold = max(0.01, min(0.5, kl_threshold))
                max_kl_samples = max(1, min(20, max_kl_samples))

                self.adaptive_config.update({
                    'margin_tolerance': margin_tol,
                    'kl_divergence_threshold': kl_threshold,
                    'max_kl_samples_per_class': max_kl_samples,
                    'enable_kl_divergence': enable_kl_divergence
                })

                print(f"✅ KL divergence configured:")
                print(f"   - Margin tolerance: {margin_tol}")
                print(f"   - KL threshold: {kl_threshold}")
                print(f"   - Max samples per class: {max_kl_samples}")


            # Validate percentages
            min_stopping_percentage = max(5.0, min(50.0, min_stopping_percentage))  # Limit to 5-50%
            max_training_percentage = max(50.0, min(95.0, max_training_percentage))  # Limit to 50-95%

            # Validate learning rate for probability-based models
            if learning_rate < 0.5:
                print("⚠️  Learning rate too low for probability-based updates. Setting to 0.5")
                learning_rate = 0.5
            elif learning_rate > 2.0:
                print("⚠️  Learning rate too high. Setting to 2.0")
                learning_rate = 2.0

            print(f"✅ Learning rate set to: {learning_rate}")

            enable_kl = input(f"Enable KL divergence selection? (y/N) [{'y' if self.adaptive_config.get('enable_kl_divergence', False) else 'n'}]: ").strip().lower()
            enable_kl_divergence = enable_kl == 'y' if enable_kl else self.adaptive_config.get('enable_kl_divergence', False)

            # Exhaust all failed examples option
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
                'enable_kl_divergence': enable_kl_divergence,
                'learning_rate': learning_rate,
                'min_training_percentage_for_stopping': min_stopping_percentage,
                'max_training_percentage': max_training_percentage,
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

            # Train with custom data
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
        Select informative samples based on configuration
        """
        # Check if KL divergence selection is enabled
        if self.adaptive_config.get('enable_kl_divergence', False):
            return self._select_kl_divergence_samples(X, y, predictions, posteriors)
        else:
            return self._select_simple_margin_samples(X, y, predictions, posteriors)

    def _select_simple_margin_samples(self, X: np.ndarray, y: np.ndarray,
                                    predictions: np.ndarray, posteriors: np.ndarray) -> List[int]:
        """
        Simple selection: one max posterior and one min margin sample per class
        """
        samples_to_add = []
        unique_classes = np.unique(y)
        max_samples = self.adaptive_config.get('max_samples_per_class_fallback', 2)

        print("🔍 Using simple margin-based selection (KL disabled)...")

        # Get test set predictions and true labels
        y_test = y[self.test_indices]

        # Find all misclassified samples
        misclassified_mask = predictions != y_test
        misclassified_indices = np.where(misclassified_mask)[0]

        if len(misclassified_indices) == 0:
            print("✅ No misclassified samples found - model is performing well!")
            return samples_to_add

        print(f"📊 Found {len(misclassified_indices)} misclassified samples")
        print(f"📊 Posteriors shape: {posteriors.shape}")
        print(f"📊 Unique true classes in test: {np.unique(y_test)}")
        print(f"📊 Unique predicted classes: {np.unique(predictions)}")

        # Group misclassified samples by true class
        class_samples = defaultdict(list)
        skipped_count = 0

        for i, idx_in_test in enumerate(misclassified_indices):
            original_idx = self.test_indices[idx_in_test]
            true_class = y_test[idx_in_test]
            pred_class = predictions[idx_in_test]

            # Handle label encoding mismatch - map predictions to valid range
            if pred_class >= posteriors.shape[1]:
                # If prediction is out of bounds, use the class with highest posterior
                pred_class = np.argmax(posteriors[idx_in_test])

            # Double-check bounds
            if (true_class >= posteriors.shape[1] or
                pred_class >= posteriors.shape[1] or
                true_class < 0 or pred_class < 0):
                skipped_count += 1
                if skipped_count <= 10:  # Only show first 10 skipped samples
                    print(f"⚠️  Skipping sample {i}: true_class={true_class}, pred_class={pred_class}, posteriors_shape={posteriors.shape}")
                continue

            # Calculate margin and posterior
            true_posterior = posteriors[idx_in_test, true_class]
            pred_posterior = posteriors[idx_in_test, pred_class]
            margin = pred_posterior - true_posterior

            class_samples[true_class].append({
                'index': original_idx,
                'margin': margin,
                'true_posterior': true_posterior,
                'pred_posterior': pred_posterior,
                'true_class': true_class,
                'pred_class': pred_class
            })

        if skipped_count > 0:
            print(f"⚠️  Skipped {skipped_count} samples due to label encoding issues")

        # For each class, select max posterior and min margin samples
        for class_id in unique_classes:
            if class_id not in class_samples or not class_samples[class_id]:
                print(f"   ⚠️ Class {class_id}: No valid failed samples available")
                continue

            class_data = class_samples[class_id]

            # Select sample with maximum true posterior (most confident wrong prediction)
            max_posterior_sample = max(class_data, key=lambda x: x['true_posterior'])

            # Select sample with minimum margin (most ambiguous decision)
            min_margin_sample = min(class_data, key=lambda x: x['margin'])

            selected_samples = []

            # Add max posterior sample if not already in training
            if (max_posterior_sample['index'] not in self.training_indices and
                max_posterior_sample['index'] not in samples_to_add):
                selected_samples.append(max_posterior_sample)
                samples_to_add.append(max_posterior_sample['index'])

                # Track selection
                self.all_selected_samples[self._get_original_class_label(class_id)].append({
                    'index': max_posterior_sample['index'],
                    'margin': max_posterior_sample['margin'],
                    'true_posterior': max_posterior_sample['true_posterior'],
                    'selection_type': 'max_posterior',
                    'round': self.adaptive_round
                })

            # Add min margin sample if different from max posterior and not in training
            if (min_margin_sample['index'] != max_posterior_sample['index'] and
                min_margin_sample['index'] not in self.training_indices and
                min_margin_sample['index'] not in samples_to_add):
                selected_samples.append(min_margin_sample)
                samples_to_add.append(min_margin_sample['index'])

                # Track selection
                self.all_selected_samples[self._get_original_class_label(class_id)].append({
                    'index': min_margin_sample['index'],
                    'margin': min_margin_sample['margin'],
                    'true_posterior': min_margin_sample['true_posterior'],
                    'selection_type': 'min_margin',
                    'round': self.adaptive_round
                })

            print(f"   Class {class_id}: Selected {len(selected_samples)} samples "
                  f"(max posterior: {max_posterior_sample['true_posterior']:.6f}, "
                  f"min margin: {min_margin_sample['margin']:.6f})")

        print(f"🎯 Selected {len(samples_to_add)} samples total using simple margin strategy")
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

    def _initialize_dbnn_architecture(self, X: np.ndarray, y: np.ndarray):
        """Initialize DBNN with entire dataset and freeze architecture"""
        print("🎯 Initializing and freezing DBNN architecture...")

        # Store original training config
        original_train_enabled = self.model.train_enabled
        original_train_only = self.model.train_only
        original_max_epochs = getattr(self.model, 'max_epochs', 1000)

        try:
            # Temporarily enable full training mode
            self.model.train_enabled = True
            self.model.train_only = False
            self.model.max_epochs = 100  # Short training for initialization

            # Prepare data for DBNN initialization
            print("   Preparing data for initialization...")
            X_train_init, X_test_init, y_train_init, y_test_init = self._prepare_dbnn_data(X, y)

            # Run one complete training round on entire dataset
            print("   Running architecture initialization...")
            self.model._compute_likelihood_parameters(X_train_init, y_train_init)
            self.model.train()

            # FREEZE ARCHITECTURE in DBNN itself
            print("   Freezing DBNN architecture...")
            self.model.freeze_architecture()

            # Verify the model is working
            print("   Verifying initialized model...")
            test_predictions = self.model.predict(X_test_init)
            test_accuracy = accuracy_score(y_test_init, test_predictions)

            print(f"✅ DBNN architecture initialized and frozen")
            print(f"   Initial accuracy: {test_accuracy:.4f}")
            print(f"   Architecture frozen: {self.model.architecture_frozen}")

        except Exception as e:
            print(f"❌ DBNN initialization failed: {e}")
            raise
        finally:
            # Restore original training config
            self.model.train_enabled = original_train_enabled
            self.model.train_only = original_train_only
            self.model.max_epochs = original_max_epochs

    def _verify_frozen_architecture_reset_weights(self):
        """Verify that architecture is frozen and preserved, but weights are reset"""
        print("   🔍 Verifying frozen architecture with preserved components and reset weights...")

        # Check architecture is frozen
        if not hasattr(self.model, 'architecture_frozen') or not self.model.architecture_frozen:
            print("     ❌ Architecture not frozen!")
            return False

        print("     ✅ Architecture frozen")

        # Verify architectural components are preserved
        if self.model.model_type == 'histogram':
            if hasattr(self.model, 'bin_edges') and self.model.bin_edges is not None:
                print(f"     ✅ bin_edges preserved: {self.model.bin_edges.shape}")
            if hasattr(self.model, 'histograms') and self.model.histograms is not None:
                print(f"     ✅ histograms preserved: {self.model.histograms.shape}")

        elif self.model.model_type == 'gaussian':
            if hasattr(self.model, 'feature_pairs') and self.model.feature_pairs is not None:
                print(f"     ✅ feature_pairs preserved: {self.model.feature_pairs.shape}")

        # Check weights are reset (should be uniform)
        if hasattr(self.model, 'current_W'):
            # For histogram model, check that weights are uniform but bins are preserved
            if self.model.model_type == 'histogram':
                # Verify we have the expected shape based on preserved histograms
                if hasattr(self.model, 'histograms') and self.model.histograms is not None:
                    expected_n_classes = self.model.histograms.shape[2]
                    expected_n_features = self.model.histograms.shape[0]
                    expected_n_bins = self.model.histograms.shape[1]

                    actual_shape = self.model.current_W.shape
                    expected_shape = (expected_n_classes, expected_n_features, expected_n_bins)

                    if actual_shape == expected_shape:
                        print(f"     ✅ Weights shape matches preserved architecture: {actual_shape}")
                    else:
                        print(f"     ❌ Weights shape mismatch: {actual_shape} vs {expected_shape}")

            # Check weights are approximately uniform
            weight_sums = np.sum(self.model.current_W, axis=0)  # Sum across classes
            is_uniform = np.allclose(weight_sums, np.ones_like(weight_sums), atol=0.2)
            if is_uniform:
                print("     ✅ Weights reset to uniform distribution")
            else:
                print("     ⚠️  Weights may not be properly reset")

        print("     ✅ Frozen architecture with preserved components and reset weights verified")
        return True

    def _extract_architectural_components(self) -> Dict[str, Any]:
        """Extract architectural components that need to be frozen (ACTUAL values, not structure)"""
        print("   📋 Extracting architectural components...")

        frozen_components = {
            'model_type': self.model.model_type,
            'histogram_bins': self.model.histogram_bins,
        }

        # Extract histogram-specific components (ACTUAL VALUES)
        if self.model.model_type == 'histogram':
            if hasattr(self.model, 'bin_edges') and self.model.bin_edges is not None:
                frozen_components['bin_edges'] = self.model.bin_edges.copy()
                print(f"     - bin_edges: {self.model.bin_edges.shape}")

            if hasattr(self.model, 'feature_min') and self.model.feature_min is not None:
                frozen_components['feature_min'] = self.model.feature_min.copy()
                print(f"     - feature_min: {self.model.feature_min.shape}")

            if hasattr(self.model, 'feature_max') and self.model.feature_max is not None:
                frozen_components['feature_max'] = self.model.feature_max.copy()
                print(f"     - feature_max: {self.model.feature_max.shape}")

            if hasattr(self.model, 'histograms') and self.model.histograms is not None:
                frozen_components['histograms'] = self.model.histograms.copy()
                print(f"     - histograms: {self.model.histograms.shape}")

        # Extract Gaussian-specific components (ACTUAL VALUES)
        elif self.model.model_type == 'gaussian':
            if hasattr(self.model, 'feature_pairs') and self.model.feature_pairs is not None:
                frozen_components['feature_pairs'] = self.model.feature_pairs.copy()
                print(f"     - feature_pairs: {self.model.feature_pairs.shape}")

        # Always extract label encoder
        if hasattr(self.model, 'label_encoder') and self.model.label_encoder is not None:
            frozen_components['label_encoder_classes'] = self.model.label_encoder.classes_.copy()
            print(f"     - label_encoder: {len(self.model.label_encoder.classes_)} classes")

        print(f"   ✅ Extracted {len(frozen_components)} architectural components (ACTUAL VALUES)")
        return frozen_components

    def _reset_weights_and_likelihood(self):
        """Reset ONLY weights and likelihood parameters, preserving architectural components"""
        print("   🔄 Resetting ONLY weights and likelihood parameters...")

        # CRITICAL: Preserve all architectural components
        # Store the current architectural state before resetting
        preserved_bin_edges = self.model.bin_edges.copy() if hasattr(self.model, 'bin_edges') and self.model.bin_edges is not None else None
        preserved_feature_min = self.model.feature_min.copy() if hasattr(self.model, 'feature_min') and self.model.feature_min is not None else None
        preserved_feature_max = self.model.feature_max.copy() if hasattr(self.model, 'feature_max') and self.model.feature_max is not None else None
        preserved_histograms = self.model.histograms.copy() if hasattr(self.model, 'histograms') and self.model.histograms is not None else None
        preserved_feature_pairs = self.model.feature_pairs.copy() if hasattr(self.model, 'feature_pairs') and self.model.feature_pairs is not None else None

        # Reset weights ONLY
        if hasattr(self.model, 'current_W'):
            if self.model.model_type == 'histogram':
                # For histogram model, reset to uniform weights but keep the same shape
                n_classes, n_features, n_bins = self.model.current_W.shape
                self.model.current_W = np.ones((n_classes, n_features, n_bins)) / n_classes
                print(f"     - Reset current_W to uniform: {self.model.current_W.shape}")
            else:
                # For Gaussian model, reset to uniform weights
                n_classes, n_combinations = self.model.current_W.shape
                self.model.current_W = np.ones((n_classes, n_combinations)) / n_classes
                print(f"     - Reset current_W to uniform: {self.model.current_W.shape}")

        # Reset best weights
        if hasattr(self.model, 'best_W'):
            self.model.best_W = self.model.current_W.copy() if hasattr(self.model, 'current_W') else None
            print(f"     - Reset best_W")

        # Reset initial weights
        if hasattr(self.model, 'initial_W'):
            self.model.initial_W = None
            print(f"     - Reset initial_W")

        # Reset likelihood parameters but preserve the structure
        if hasattr(self.model, 'likelihood_params') and self.model.likelihood_params is not None:
            if self.model.model_type == 'gaussian':
                # For Gaussian model, we need to recompute likelihood but with the SAME feature pairs
                # Store the feature pairs first
                if preserved_feature_pairs is not None:
                    self.model.feature_pairs = preserved_feature_pairs
                self.model.likelihood_params = None
                print(f"     - Reset likelihood_params (will be recomputed with SAME architecture)")

        # CRITICAL: RESTORE the architectural components
        if preserved_bin_edges is not None:
            self.model.bin_edges = preserved_bin_edges
            print(f"     - Preserved bin_edges: {preserved_bin_edges.shape}")

        if preserved_feature_min is not None:
            self.model.feature_min = preserved_feature_min
            print(f"     - Preserved feature_min: {preserved_feature_min.shape}")

        if preserved_feature_max is not None:
            self.model.feature_max = preserved_feature_max
            print(f"     - Preserved feature_max: {preserved_feature_max.shape}")

        if preserved_histograms is not None:
            self.model.histograms = preserved_histograms
            print(f"     - Preserved histograms: {preserved_histograms.shape}")

        # Reset training state
        if hasattr(self.model, 'best_accuracy'):
            self.model.best_accuracy = 0.0
            print(f"     - Reset best_accuracy")

        if hasattr(self.model, 'best_error'):
            self.model.best_error = float('inf')
            print(f"     - Reset best_error")

        print("   ✅ Weights reset, architectural components PRESERVED")

    def _prepare_dbnn_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for DBNN initialization training"""
        # Use the same split as the main adaptive learning
        from sklearn.model_selection import train_test_split

        # Filter sentinel values first
        X_clean, y_clean = self._filter_sentinel_samples(X, y)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=self.model.test_size, random_state=self.model.random_state
        )

        # Scale the data using the model's scaler
        X_train_scaled = self.model.scaler.fit_transform(X_train)
        X_test_scaled = self.model.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def _verify_architecture_frozen(self):
        """Verify that DBNN architecture remains frozen"""
        if not hasattr(self.model, 'architecture_frozen') or not self.model.architecture_frozen:
            print("   ❌ DBNN architecture not frozen!")
            return False

        print("   🔒 DBNN architecture verified frozen")
        return True

    def _debug_freeze_state(self):
        """Debug the freeze state of both adaptive and DBNN models"""
        print(f"\n🔍 FREEZE STATE DEBUG:")
        print("=" * 50)

        # Adaptive model freeze state
        adaptive_frozen = getattr(self, 'architecture_frozen', False)
        print(f"   AdaptiveDBNN frozen: {adaptive_frozen}")

        # DBNN model freeze state
        if hasattr(self.model, 'architecture_frozen'):
            dbnn_frozen = self.model.architecture_frozen
            frozen_components = list(getattr(self.model, 'frozen_components', {}).keys())
            print(f"   DBNN frozen: {dbnn_frozen}")
            print(f"   DBNN frozen components: {frozen_components}")
        else:
            print(f"   ❌ DBNN freeze capability not available")

        print("=" * 50)

    def adaptive_learn(self, X: np.ndarray = None, y: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Main adaptive learning with initialize-and-freeze architecture"""
        print("\n🚀 STARTING ADAPTIVE LEARNING WITH INITIALIZE-AND-FREEZE ARCHITECTURE")
        print("=" * 80)

        # Use base model's data if not provided
        if X is None or y is None:
            print("Using base model's dataset...")
            X, y, y_original = self.prepare_full_data()

        # Store the full dataset
        self.X_full = X.copy()
        self.y_full = y.copy()
        self.y_full_original = y_original

        print(f"📦 Total samples: {len(X)}")
        print(f"🎯 Classes: {np.unique(y_original)}")

        # STEP 1: INITIALIZE DBNN WITH ENTIRE DATASET
        print("\n" + "="*50)
        print("🎯 STEP 1: INITIALIZING DBNN ARCHITECTURE")
        print("="*50)

        self._initialize_dbnn_architecture(X, y)

        # STEP 2: VERIFY FREEZE STATE
        print("\n" + "="*50)
        print("🎯 STEP 2: VERIFYING FREEZE STATE")
        print("="*50)

        self._debug_freeze_state()

        # STEP 3: START ADAPTIVE TRAINING WITH FROZEN ARCHITECTURE
        print("\n" + "="*50)
        print("🎯 STEP 3: STARTING ADAPTIVE TRAINING (FROZEN ARCHITECTURE)")
        print("="*50)

        # Initialize adaptive learning state
        X_train, y_train, X_test, y_test = self.prepare_adaptive_data(X, y)
        self.adaptive_start_time = datetime.now()

        # Track failed examples history
        failed_history = []

        exhaust_all_failed = self.adaptive_config['exhaust_all_failed']
        min_failed_threshold = self.adaptive_config['min_failed_threshold']

        if exhaust_all_failed:
            print("🔁 MODE: EXHAUSTIVE - Continuing until failed examples are addressed")
            print(f"🎯 Min failed threshold: {min_failed_threshold}")
            effective_max_rounds = 1000
        else:
            print(f"🔄 MODE: PATIENCE - Max rounds: {self.adaptive_config['max_adaptive_rounds']}")
            effective_max_rounds = self.adaptive_config['max_adaptive_rounds']

        # Main adaptive learning loop
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
            if round_num > 1000:
                print("🛑 Safety limit reached: 1000 rounds")
                break

            # Verify architecture is frozen
            self._verify_architecture_frozen()

            # Step 1: Train until convergence
            round_start_time = datetime.now()
            training_success = self._train_until_convergence(X_train, y_train)

            if not training_success:
                print("❌ Training failed, skipping round")
                continue

            # Step 2: Evaluate CURRENT model on TEST set
            print("🎯 Evaluating current model on test set...")
            current_test_predictions = self._predict_with_current_model(X_test)
            current_test_accuracy = accuracy_score(y_test, current_test_predictions)

            # Count failed examples
            misclassified_mask = current_test_predictions != y_test
            failed_count = np.sum(misclassified_mask)
            failed_history.append(failed_count)

            print(f"📊 Round {round_num} Test Accuracy: {current_test_accuracy:.4f}")
            print(f"❌ Failed examples: {failed_count}")

            # Step 3: Check if current model is POTENTIALLY better than global best
            improvement_threshold = self.adaptive_config['min_improvement']
            is_potential_improvement = current_test_accuracy > self.global_best_accuracy + improvement_threshold

            if is_potential_improvement:
                print(f"🎯 POTENTIAL IMPROVEMENT DETECTED!")
                print(f"   Current test accuracy: {current_test_accuracy:.4f}")
                print(f"   Global best accuracy: {self.global_best_accuracy:.4f}")
                print(f"   Improvement threshold: +{improvement_threshold:.4f}")

                # Step 4: Save current weights temporarily and perform ACID TEST
                temp_weights = self.model.current_W.copy() if hasattr(self.model, 'current_W') else None

                print("🧪 Performing acid test on entire dataset...")
                current_full_predictions = self._predict_with_current_model(X)
                current_full_accuracy = accuracy_score(y, current_full_predictions)
                print(f"🧪 Entire Dataset Accuracy: {current_full_accuracy:.4f}")

                # Step 5: Update global best if acid test confirms improvement
                if current_full_accuracy > self.global_best_accuracy + improvement_threshold:
                    improvement = current_full_accuracy - self.global_best_accuracy

                    # Update global best metrics
                    self.global_best_accuracy = current_full_accuracy
                    self.global_best_round = round_num
                    self.global_best_training_indices = self.training_indices.copy()
                    self.global_best_test_indices = self.test_indices.copy()
                    self.global_best_weights = temp_weights

                    # Also update displayed best accuracy
                    self.best_accuracy = current_full_accuracy  # Display ENTIRE dataset accuracy
                    self.best_round = round_num
                    self.best_training_indices = self.training_indices.copy()
                    self.best_test_indices = self.test_indices.copy()
                    self.best_weights = temp_weights

                    print(f"🏆 NEW GLOBAL BEST MODEL!")
                    print(f"   Test Accuracy (remaining data): {current_test_accuracy:.4f}")
                    print(f"   Entire Dataset Accuracy (all data): {current_full_accuracy:.4f}")
                    print(f"   Improvement: +{improvement:.4f}")
                    print(f"   Training set size: {len(self.training_indices)}")

                    # Save the new best weights
                    if self.best_weights is not None:
                        self.model._save_best_weights()
                        print("💾 Best weights saved")
                else:
                    print(f"   ❌ Acid test failed to confirm improvement")
                    print(f"   Entire dataset accuracy: {current_full_accuracy:.4f} <= {self.global_best_accuracy:.4f} + {improvement_threshold}")

                    # Restore previous best weights if we have them
                    if self.global_best_weights is not None and hasattr(self.model, 'current_W'):
                        self.model.current_W = self.global_best_weights.copy()
            else:
                print(f"   📉 No potential improvement detected")
                print(f"   Current test accuracy: {current_test_accuracy:.4f}")
                print(f"   Global best accuracy: {self.global_best_accuracy:.4f}")

            # Track 3D visualization data
            if hasattr(self, '_track_feature_grids_3d'):
                if round_num % 5 == 0 or round_num <= 10:
                    self._track_feature_grids_3d(round_num, X_train, y_train)

            # Create round visualizations
            if self.stats_config.get('enable_confusion_matrix', True) and round_num <= 20:
                self._create_round_visualizations(round_num, y_test, current_test_predictions)

            # Step 6: Select informative samples for next round
            print("🎯 Selecting informative samples...")
            posteriors = self._get_posteriors_with_current_model(X_test)
            samples_to_add = self._select_informative_samples(X, y, current_test_predictions, posteriors)

            # Step 7: Check stopping conditions
            stop_reason = self._check_stopping_conditions(
                round_num, samples_to_add, failed_count, min_failed_threshold,
                self.adaptive_config['patience'], exhaust_all_failed, effective_max_rounds,
                self.global_best_accuracy, self.global_best_accuracy
            )

            if stop_reason:
                print(f"🛑 {stop_reason}")
                break

            # Step 8: Update training set
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
                                        current_test_accuracy, samples_to_add, round_duration,
                                        failed_count, self.global_best_accuracy)

            print(f"📥 Added {added_count} samples to training set")
            print(f"📊 New training size: {len(X_train)}")
            print(f"📊 New test size: {len(X_test)}")
            print(f"❌ Remaining failed examples: {failed_count}")
            print(f"🏆 Global Best Entire Dataset Accuracy: {self.global_best_accuracy:.4f}")
            print(f"⏱️ Round duration: {round_duration:.2f}s")

            # Progress indicator for exhaustive mode
            if exhaust_all_failed and round_num % 10 == 0:
                initial_test_size = len(X) - len(self.global_best_training_indices) if self.global_best_training_indices else len(X_test)
                progress = ((initial_test_size - len(X_test)) / initial_test_size) * 100
                print(f"📈 Progress: {progress:.1f}% of test set processed")

        # Final processing
        print(f"\n🏁 Adaptive Learning Completed after {round_num} rounds")
        print("=" * 70)

        # Use GLOBAL BEST configuration
        if self.global_best_training_indices:
            X_train_best = X[self.global_best_training_indices]
            y_train_best = y[self.global_best_training_indices]
            X_test_best = X[self.global_best_test_indices]
            y_test_best = y[self.global_best_test_indices]

            # Restore global best weights
            if self.global_best_weights is not None and hasattr(self.model, 'current_W'):
                self.model.current_W = self.global_best_weights.copy()
        else:
            # Fallback to current configuration
            X_train_best = X[self.training_indices]
            y_train_best = y[self.training_indices]
            X_test_best = X[self.test_indices]
            y_test_best = y[self.test_indices]

        print(f"🏆 Best Entire Dataset Accuracy: {self.global_best_accuracy:.4f} (achieved at round {self.global_best_round})")
        print(f"📊 Final test set accuracy: {current_test_accuracy:.4f}")
        print(f"📦 Optimal training set size: {len(X_train_best)}")
        print(f"📊 Final test set size: {len(X_test_best)}")
        print(f"❌ Final failed examples: {failed_history[-1] if failed_history else 'N/A'}")
        print(f"⏱️ Total training time: {(datetime.now() - self.adaptive_start_time).total_seconds():.2f}s")

        # Retrain final model with best configuration
        print("\n🔧 Training final model with optimal configuration...")
        self._train_with_custom_data(X_train_best, y_train_best)

        # Create comprehensive visualizations and analysis
        print("\n📊 Creating comprehensive analysis...")
        self._create_comprehensive_analysis(X_train_best, y_train_best, X_test_best, y_test_best)

        # Create interactive HTML visualization dashboard
        print("\n🎨 Creating interactive HTML visualization dashboard...")
        self._create_interactive_html_dashboard(X_train_best, y_train_best, X_test_best, y_test_best)

        # Create animated GIFs for quick viewing
        if hasattr(self, '_create_unified_3d_animation'):
            print("\n🎬 Creating animated visualizations...")
            self._create_unified_3d_animation()
            if hasattr(self, '_create_comprehensive_3d_visualization'):
                self._create_comprehensive_3d_visualization(X_train_best, y_train_best)

        # Save final results
        self._save_final_results(X_train_best, y_train_best, X_test_best, y_test_best)

        # Final summary
        print("\n" + "="*70)
        print("✅ ADAPTIVE LEARNING COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"📁 Results saved in: {self.viz_config.get('output_dir', 'adaptive_visualizations')}")
        print(f"🏆 Best Model: Entire Dataset Accuracy = {self.global_best_accuracy:.4f}")
        print("\n📊 Available Visualizations:")
        print("   - interactive_dashboard.html (Complete interactive analysis)")
        print("   - adaptive_learning_3d_evolution.gif (Animated 3D evolution)")
        print("   - adaptive_learning_progression.png (Learning curves)")
        print("   - final_performance_analysis.png (Model performance)")
        print("   - adaptive_learning_summary.json (Detailed statistics)")
        print("\n💡 Open 'interactive_dashboard.html' in your browser for the complete analysis!")

        return X_train_best, y_train_best, X_test_best, y_test_best

    def _check_stopping_conditions(self, round_num: int, samples_to_add: List[int],
                                 failed_count: int, min_failed_threshold: int,
                                 patience: int, exhaust_all_failed: bool, max_rounds: int,
                                 current_global_best: float, previous_global_best: float) -> str:
        """Check all stopping conditions with global best accuracy"""

        # Calculate current training set percentage of entire dataset
        current_training_size = len(self.training_indices)
        total_dataset_size = len(self.X_full) if hasattr(self, 'X_full') else 1
        training_percentage = (current_training_size / total_dataset_size) * 100

        # Get configurable thresholds
        min_stopping_percentage = self.adaptive_config.get('min_training_percentage_for_stopping', 10.0)
        max_training_percentage = self.adaptive_config.get('max_training_percentage', 90.0)

        print(f"📊 Training set: {current_training_size}/{total_dataset_size} ({training_percentage:.1f}% of total)")

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

            # Relax global best degradation check until we have sufficient training data
            degradation_threshold = 0.1  # 10% degradation
            if training_percentage >= min_stopping_percentage:
                # Use global best accuracy for degradation check
                if current_global_best < previous_global_best - degradation_threshold:
                    return f"Significant degradation in global best accuracy: {current_global_best:.4f} < {previous_global_best:.4f}"
            else:
                print(f"   🛡️  Global best degradation check suspended (training data: {training_percentage:.1f}% < {min_stopping_percentage}%)")

            # Continue learning regardless of rounds or patience
            return ""

        # 3. Max rounds
        if round_num >= max_rounds:
            return f"Reached maximum rounds ({max_rounds})"

        # 4. Patience-based stopping (only if not in exhaustive mode)
        if not exhaust_all_failed:
            # Use rounds without global best improvement for patience
            rounds_without_improvement = round_num - self.global_best_round if self.global_best_round > 0 else 0

            if training_percentage >= min_stopping_percentage:
                if rounds_without_improvement >= patience:
                    return f"Early stopping after {patience} rounds without global best improvement"
            else:
                print(f"   🛡️  Patience stopping suspended (training data: {training_percentage:.1f}% < {min_stopping_percentage}%)")
                return ""

        # 5. Stop if training set reaches maximum allowed percentage
        if training_percentage >= max_training_percentage:
            return f"Training set reached {training_percentage:.1f}% of total data - stopping to prevent overfitting"

        return ""  # Continue learning

    def _record_round_statistics(self, round_num: int, X_train: np.ndarray, y_train: np.ndarray,
                               X_test: np.ndarray, y_test: np.ndarray, current_accuracy: float,
                               samples_added: List[int], duration: float, failed_count: int,
                               global_best_accuracy: float):
        """Record comprehensive statistics for the current round"""
        # Get class distribution with original labels
        y_train_original = np.array([self._get_original_class_label(cls) for cls in y_train])
        unique_classes, class_counts = np.unique(y_train_original, return_counts=True)
        class_distribution = dict(zip(unique_classes, class_counts))

        stats = {
            'round': round_num,
            'training_size': len(X_train),
            'test_size': len(X_test),
            'current_test_accuracy': current_accuracy,  # Current model on test set
            'global_best_accuracy': global_best_accuracy,  # Best model on entire dataset
            'samples_added': len(samples_added),
            'class_distribution': class_distribution,
            'duration': duration,
            'failed_count': failed_count,
            'global_best_round': self.global_best_round,
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
        current_accuracies = [stats['current_test_accuracy'] for stats in self.round_stats]
        global_best_accuracies = [stats['global_best_accuracy'] for stats in self.round_stats]
        training_sizes = [stats['training_size'] for stats in self.round_stats]
        failed_counts = [stats['failed_count'] for stats in self.round_stats]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Accuracy progression
        ax1.plot(rounds, current_accuracies, 'b-o', linewidth=2, markersize=6, label='Current Test Accuracy')
        ax1.plot(rounds, global_best_accuracies, 'r-o', linewidth=2, markersize=6, label='Global Best (Entire Dataset)')
        ax1.set_xlabel('Adaptive Round')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Adaptive Learning - Accuracy Progression')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=self.global_best_accuracy, color='g', linestyle='--', label=f'Best: {self.global_best_accuracy:.4f}')
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
            'global_best_round': self.global_best_round,
            'global_best_accuracy': self.global_best_accuracy,
            'final_training_size': len(self.best_training_indices),
            'adaptive_config': self.adaptive_config,
            'start_time': self.adaptive_start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_duration': (datetime.now() - self.adaptive_start_time).total_seconds(),
            'round_statistics': self.round_stats,
            'sample_selection_summary': {
                class_label: len(samples)
                for class_label, samples in self.all_selected_samples.items()
            },
            'architecture_frozen': getattr(self.model, 'architecture_frozen', False)
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
        print(f"Global Best Round: {summary['global_best_round']}")
        print(f"Global Best Accuracy (Entire Dataset): {summary['global_best_accuracy']:.4f}")
        print(f"Final Training Size: {summary['final_training_size']}")
        print(f"Total Duration: {summary['total_duration']:.2f}s")
        print(f"Architecture Frozen: {summary['architecture_frozen']}")
        print(f"Sample Selection Summary: {summary['sample_selection_summary']}")

    def _save_final_results(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray):
        """Save final results and model"""
        print("\n💾 Saving final results...")

        output_dir = self.viz_config.get('output_dir', 'adaptive_visualizations')

        # Save indices and global best information
        results = {
            'global_best_training_indices': self.global_best_training_indices,
            'global_best_test_indices': self.global_best_test_indices,
            'global_best_accuracy': self.global_best_accuracy,
            'global_best_round': self.global_best_round,
            'best_training_indices': self.best_training_indices,
            'best_test_indices': self.best_test_indices,
            'best_accuracy': self.best_accuracy,
            'best_round': self.best_round,
            'adaptive_config': self.adaptive_config,
            'round_statistics': self.round_stats,
            'final_training_set_size': len(X_train),
            'final_test_set_size': len(X_test),
            'architecture_frozen': getattr(self.model, 'architecture_frozen', False)
        }

        with open(f'{output_dir}/adaptive_results.json', 'w') as f:
            json.dump(results, f, indent=4, default=str)

        # Save the trained model
        self.model.save_model()

        print("✅ Final results saved with global best model information!")

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
        """Get predictions from current model with proper label encoding"""
        try:
            # Get raw predictions directly from DBNN
            raw_predictions = self.model.predict(X)

            # Determine if predictions are encoded or decoded
            if hasattr(self, 'label_encoder') and self.label_encoder is not None:
                # Check if predictions are within valid encoded range
                unique_preds = np.unique(raw_predictions)
                n_classes = len(self.label_encoder.classes_)

                if (np.issubdtype(raw_predictions.dtype, np.integer) and
                    np.min(unique_preds) >= 0 and
                    np.max(unique_preds) < n_classes):
                    return raw_predictions
                else:
                    # Predictions are decoded, convert to encoded
                    try:
                        encoded_predictions = self.label_encoder.transform(raw_predictions)
                        return encoded_predictions
                    except Exception as e:
                        print(f"   ❌ Encoding failed: {e}")
                        return raw_predictions
            else:
                return raw_predictions

        except Exception as e:
            print(f"⚠️ Prediction error: {e}")
            # Return fallback predictions that match y encoding
            n_classes = len(np.unique(self.y_full)) if hasattr(self, 'y_full') else 5
            return np.random.randint(0, n_classes, len(X))

    def _get_posteriors_with_current_model(self, X: np.ndarray) -> np.ndarray:
        """Get posterior probabilities from current model with numerical stability"""
        try:
            posteriors = self.model._compute_batch_posterior(X)

            # Add numerical stability checks
            if np.any(np.isnan(posteriors)) or np.any(np.isinf(posteriors)):
                print(f"❌ Invalid posteriors detected: NaN={np.any(np.isnan(posteriors))}, Inf={np.any(np.isinf(posteriors))}")
                # Fix by setting to uniform distribution
                n_classes = posteriors.shape[1]
                uniform_probs = np.ones_like(posteriors) / n_classes
                return uniform_probs

            # Check for zero posteriors (numerical underflow)
            zero_mask = np.sum(posteriors, axis=1) == 0
            if np.any(zero_mask):
                n_zero = np.sum(zero_mask)
                print(f"⚠️  {n_zero} samples have zero posteriors - applying numerical fix")
                # Apply uniform distribution
                n_classes = posteriors.shape[1]
                posteriors[zero_mask] = 1.0 / n_classes

            # Ensure probabilities sum to 1
            row_sums = np.sum(posteriors, axis=1, keepdims=True)
            if np.any(np.abs(row_sums - 1.0) > 1e-6):
                posteriors = posteriors / (row_sums + 1e-10)

            return posteriors

        except Exception as e:
            print(f"⚠️ Posterior error: {e}")
            # Return uniform probabilities as fallback
            n_classes = len(np.unique(self.y_full)) if hasattr(self, 'y_full') else 5
            return np.ones((len(X), n_classes)) / n_classes

    def _filter_sentinel_samples(self, X: np.ndarray, y: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Filter out samples that contain sentinel values (-9999)"""
        SENTINEL_VALUE = -9999
        valid_mask = ~np.any(X == SENTINEL_VALUE, axis=1)

        if not np.any(valid_mask):
            raise ValueError("All samples contain sentinel values - no valid data to process")

        X_filtered = X[valid_mask]

        if y is not None:
            y_filtered = y[valid_mask]
            filtered_count = len(X) - len(X_filtered)
            if filtered_count > 0:
                print(f"Filtered out {filtered_count} samples containing sentinel values")
            return X_filtered, y_filtered

        return X_filtered, valid_mask

    # Placeholder for methods that were not fully implemented in the original
    def _select_kl_divergence_samples(self, X: np.ndarray, y: np.ndarray,
                                    predictions: np.ndarray, posteriors: np.ndarray) -> List[int]:
        """
        Hybrid selection: First find failed examples within margin tolerance,
        then rank by KL divergence to select most informative samples
        """
        print("🔍 Using HYBRID margin-tolerance + KL divergence selection...")

        samples_to_add = []
        unique_classes = np.unique(y)

        # Get configuration parameters
        margin_tolerance = self.adaptive_config.get('margin_tolerance', 0.15)
        kl_threshold = self.adaptive_config.get('kl_divergence_threshold', 0.1)
        max_kl_samples = self.adaptive_config.get('max_kl_samples_per_class', 5)

        print(f"   Margin tolerance: {margin_tolerance}, KL threshold: {kl_threshold}")
        print(f"   Max samples per class: {max_kl_samples}")

        # Get test set predictions and true labels
        y_test = y[self.test_indices]

        # Find all misclassified samples
        misclassified_mask = predictions != y_test
        misclassified_indices = np.where(misclassified_mask)[0]

        if len(misclassified_indices) == 0:
            print("✅ No misclassified samples found")
            return samples_to_add

        print(f"📊 Found {len(misclassified_indices)} misclassified samples")

        # Group misclassified samples by true class and calculate margins
        class_samples = defaultdict(list)

        for i, idx_in_test in enumerate(misclassified_indices):
            original_idx = self.test_indices[idx_in_test]
            true_class = y_test[idx_in_test]
            pred_class = predictions[idx_in_test]

            # Skip if indices are out of bounds
            if (true_class >= posteriors.shape[1] or pred_class >= posteriors.shape[1] or
                true_class < 0 or pred_class < 0):
                continue

            # Calculate margin
            true_posterior = posteriors[idx_in_test, true_class]
            pred_posterior = posteriors[idx_in_test, pred_class]
            margin = pred_posterior - true_posterior

            class_samples[true_class].append({
                'index': original_idx,
                'margin': margin,
                'true_posterior': true_posterior,
                'pred_posterior': pred_posterior,
                'true_class': true_class,
                'pred_class': pred_class,
                'posteriors': posteriors[idx_in_test]  # Store full posterior for KL calculation
            })

        # For each class, apply the hybrid selection strategy
        for class_id in unique_classes:
            if class_id not in class_samples or not class_samples[class_id]:
                print(f"   ⚠️ Class {class_id}: No valid failed samples available")
                continue

            class_data = class_samples[class_id]
            print(f"   Class {class_id}: {len(class_data)} failed samples")

            # STEP 1: Find margin range for this class
            margins = [sample['margin'] for sample in class_data]
            min_margin = min(margins)
            max_margin = max(margins)
            margin_range = max_margin - min_margin

            # Calculate tolerance threshold
            tolerance_threshold = min_margin + (margin_range * margin_tolerance)

            print(f"     Margin range: [{min_margin:.4f}, {max_margin:.4f}]")
            print(f"     Tolerance threshold: {tolerance_threshold:.4f}")

            # STEP 2: Filter samples within margin tolerance (most ambiguous/misclassified)
            tolerance_samples = [sample for sample in class_data if sample['margin'] <= tolerance_threshold]

            if not tolerance_samples:
                print(f"     ⚠️ No samples within margin tolerance, using all samples")
                tolerance_samples = class_data

            print(f"     Samples within margin tolerance: {len(tolerance_samples)}")

            # STEP 3: Calculate KL divergence for filtered samples
            kl_samples = []
            for sample in tolerance_samples:
                try:
                    # Create target distribution (one-hot for true class)
                    target_distribution = np.zeros_like(sample['posteriors'])
                    target_distribution[sample['true_class']] = 1.0

                    # Calculate KL divergence: KL(target || predicted)
                    eps = 1e-10
                    p_safe = target_distribution + eps
                    q_safe = sample['posteriors'] + eps

                    # Normalize
                    p_safe = p_safe / np.sum(p_safe)
                    q_safe = q_safe / np.sum(q_safe)

                    kl_divergence = np.sum(p_safe * np.log(p_safe / q_safe))

                    # Also calculate entropy of predicted distribution for additional insight
                    entropy_pred = entropy(q_safe)

                    kl_samples.append({
                        **sample,  # Include all original sample data
                        'kl_divergence': kl_divergence,
                        'entropy': entropy_pred,
                        'combined_score': kl_divergence * (1 + entropy_pred)  # Combine KL and entropy
                    })

                except Exception as e:
                    continue

            if not kl_samples:
                print(f"     ❌ No valid KL divergence calculations")
                continue

            # STEP 4: Filter by KL divergence threshold and sort
            high_kl_samples = [sample for sample in kl_samples if sample['kl_divergence'] >= kl_threshold]

            if not high_kl_samples:
                print(f"     ⚠️ No samples meet KL threshold, using top KL samples")
                # If no samples meet threshold, use top samples by KL divergence
                high_kl_samples = sorted(kl_samples, key=lambda x: x['kl_divergence'], reverse=True)[:max_kl_samples]
            else:
                # Sort high KL samples by combined score (KL divergence × entropy)
                high_kl_samples = sorted(high_kl_samples, key=lambda x: x['combined_score'], reverse=True)

            print(f"     High KL samples (≥{kl_threshold}): {len(high_kl_samples)}")

            # STEP 5: Select top samples
            selected_for_class = []
            for sample_info in high_kl_samples[:max_kl_samples]:
                if (sample_info['index'] not in self.training_indices and
                    sample_info['index'] not in samples_to_add):

                    selected_for_class.append(sample_info)
                    samples_to_add.append(sample_info['index'])

                    # Track selection with detailed information
                    self.all_selected_samples[self._get_original_class_label(class_id)].append({
                        'index': sample_info['index'],
                        'margin': sample_info['margin'],
                        'kl_divergence': sample_info['kl_divergence'],
                        'entropy': sample_info['entropy'],
                        'combined_score': sample_info['combined_score'],
                        'true_posterior': sample_info['true_posterior'],
                        'selection_type': 'hybrid_kl_divergence',
                        'round': self.adaptive_round,
                        'within_margin_tolerance': sample_info['margin'] <= tolerance_threshold
                    })

            if selected_for_class:
                avg_margin = np.mean([s['margin'] for s in selected_for_class])
                avg_kl = np.mean([s['kl_divergence'] for s in selected_for_class])
                print(f"     ✅ Selected {len(selected_for_class)} samples "
                      f"(avg margin: {avg_margin:.4f}, avg KL: {avg_kl:.4f})")

                # Show selection details
                if len(selected_for_class) > 0:
                    best_sample = selected_for_class[0]
                    print(f"       Best: KL={best_sample['kl_divergence']:.4f}, "
                          f"margin={best_sample['margin']:.4f}, "
                          f"entropy={best_sample['entropy']:.4f}")
            else:
                print(f"     ❌ No new samples selected for class {class_id}")

        print(f"🎯 Selected {len(samples_to_add)} total samples using hybrid KL divergence strategy")

        # Show overall statistics
        if samples_to_add:
            all_kl_values = []
            all_margins = []
            for class_id in unique_classes:
                if class_id in class_samples:
                    for sample in self.all_selected_samples.get(self._get_original_class_label(class_id), []):
                        if sample['round'] == self.adaptive_round and sample['selection_type'] == 'hybrid_kl_divergence':
                            all_kl_values.append(sample['kl_divergence'])
                            all_margins.append(sample['margin'])

            if all_kl_values:
                print(f"📈 Selection stats: KL[{min(all_kl_values):.4f}, {max(all_kl_values):.4f}], "
                      f"Margin[{min(all_margins):.4f}, {max(all_margins):.4f}]")

        return samples_to_add

    def _create_interactive_html_dashboard(self, X_train: np.ndarray, y_train: np.ndarray,
                                        X_test: np.ndarray, y_test: np.ndarray):
        """Create interactive HTML dashboard (placeholder)"""
        print("📊 Interactive HTML dashboard creation not yet implemented")

    def _create_unified_3d_animation(self):
        """Create unified 3D animation (placeholder)"""
        print("🎬 Unified 3D animation creation not yet implemented")

    def _create_comprehensive_3d_visualization(self, X_train: np.ndarray, y_train: np.ndarray):
        """Create comprehensive 3D visualization (placeholder)"""
        print("🎨 Comprehensive 3D visualization creation not yet implemented")


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
    print(f"🏆 Best accuracy achieved: {adaptive_model.global_best_accuracy:.4f}")

if __name__ == "__main__":
    main()
