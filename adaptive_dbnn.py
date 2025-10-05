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
            "enable_kl_divergence": False,  # NEW: Enable KL divergence based selection
            "max_samples_per_class_fallback": 2,  # NEW: When KL disabled, add max 2 samples per class
            "enable_3d_visualization": True,
            "3d_snapshot_interval": 10,
            "learning_rate": 1.0,  # NEW: Higher learning rate for probability-based updates
            "enable_acid_test": True,  # NEW: Enable entire dataset validation
            "min_training_percentage_for_stopping": 10.0,  # NEW: Minimum training data % before applying early stopping
            "max_training_percentage": 90.0,  # NEW: Maximum training data % to prevent overfitting
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

    def _create_interactive_3d_plotly(self):
        """Create interactive 3D visualization using Plotly"""
        try:
            if len(self.feature_grid_history) < 2:
                return

            print("📊 Creating interactive 3D visualization...")

            # Create interactive plot with Plotly
            fig = make_subplots(
                rows=2, cols=2,
                specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}],
                       [{"type": "scatter"}, {"type": "scatter"}]],
                subplot_titles=('3D Feature Space Evolution', 'Weight Trajectory',
                              'Feature Importance', 'Grid Drift')
            )

            # Add your Plotly 3D visualization code here
            # This would be similar to the matplotlib versions but interactive

            fig.update_layout(height=800, title_text="Interactive Adaptive Learning Visualization")
            fig.write_html(f"{self.visualization_output_dir}/interactive_3d_visualization.html")

            print("✅ Interactive 3D visualization saved")

        except Exception as e:
            print(f"⚠️ Could not create interactive 3D visualization: {e}")

    def _create_unified_3d_animation(self):
        """Create a unified rotating 3D animation showing the entire evolution"""
        try:
            print("🎬 Creating unified rotating 3D animation...")

            # Get all snapshot files
            image_files = sorted([f for f in os.listdir(f'{self.visualization_output_dir}/3d_animations')
                                if f.startswith('epoch_') and f.endswith('.png')])

            if not image_files:
                print("⚠️ No 3D snapshots found for animation")
                return

            # Create a unified figure with rotating views
            self._create_rotating_3d_snapshots(image_files)

            # Create the final animated GIF
            self._compile_rotating_gif()

            print("✅ Unified 3D animation completed!")

        except Exception as e:
            print(f"⚠️ Could not create unified 3D animation: {e}")

    def _create_rotating_3d_snapshots(self, image_files):
        """Create rotating 3D snapshots for animation"""
        try:
            # Load the first image to get dimensions
            first_image_path = f'{self.visualization_output_dir}/3d_animations/{image_files[0]}'
            first_img = plt.imread(first_image_path)

            # Create rotation frames for each epoch
            rotation_frames = []

            for i, image_file in enumerate(image_files):
                epoch = int(image_file.split('_')[1].split('.')[0])
                image_path = f'{self.visualization_output_dir}/3d_animations/{image_file}'

                print(f"🔄 Creating rotation frames for epoch {epoch} ({i+1}/{len(image_files)})")

                # Create multiple rotated views for this epoch
                epoch_frames = self._create_single_epoch_rotation(image_path, epoch, i, len(image_files))
                rotation_frames.extend(epoch_frames)

            # Save rotation frames
            for j, frame in enumerate(rotation_frames):
                frame_path = f'{self.visualization_output_dir}/3d_animations/rotation_frame_{j:04d}.png'
                frame.savefig(frame_path, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close()

            print(f"✅ Created {len(rotation_frames)} rotation frames")

        except Exception as e:
            print(f"⚠️ Error creating rotating snapshots: {e}")

    def _create_single_epoch_rotation(self, image_path, epoch, epoch_index, total_epochs):
        """Create multiple rotated views for a single epoch"""
        frames = []

        try:
            # Load the original image
            img = plt.imread(image_path)

            # Create figure for composition
            fig = plt.figure(figsize(16, 12))

            # Number of rotation frames per epoch
            frames_per_epoch = 8  # Reduced for smoother animation

            for rotation_idx in range(frames_per_epoch):
                # Calculate rotation angle (0 to 360 degrees)
                rotation_angle = (rotation_idx / frames_per_epoch) * 360

                # Clear figure
                plt.clf()

                # Create 2x2 grid for comprehensive view
                gs = plt.GridSpec(2, 2, figure=fig)

                # Main 3D view (top-left)
                ax1 = fig.add_subplot(gs[0, 0], projection='3d')
                self._plot_rotating_3d_view(ax1, img, rotation_angle, "Feature Space Evolution")

                # Top view (top-right)
                ax2 = fig.add_subplot(gs[0, 1], projection='3d')
                self._plot_top_down_view(ax2, img, "Top View")

                # Side view (bottom-left)
                ax3 = fig.add_subplot(gs[1, 0], projection='3d')
                self._plot_side_view(ax3, img, "Side View")

                # Progress and metrics (bottom-right)
                ax4 = fig.add_subplot(gs[1, 1])
                self._plot_progress_panel(ax4, epoch, epoch_index, total_epochs, rotation_angle)

                # Main title
                plt.suptitle(f'Adaptive Learning Evolution\nEpoch {epoch} | Rotation: {rotation_angle:.0f}°',
                            fontsize=16, fontweight='bold', y=0.95)

                plt.tight_layout()
                frames.append(fig)

            return frames

        except Exception as e:
            print(f"⚠️ Error creating rotation for epoch {epoch}: {e}")
            return []

    def _plot_rotating_3d_view(self, ax, img, rotation_angle, title):
        """Plot main rotating 3D view"""
        try:
            # This would be your actual 3D data plotting
            # For now, we'll create a placeholder with the image
            ax.imshow(img, aspect='auto', extent=[-1, 1, -1, 1, -1, 1])
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            ax.set_title(title)

            # Set rotation
            ax.view_init(elev=30, azim=rotation_angle)

        except Exception as e:
            ax.text(0.5, 0.5, f"3D View\nError: {e}", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)

    def _plot_top_down_view(self, ax, img, title):
        """Plot top-down view"""
        try:
            ax.imshow(img, aspect='auto', extent=[-1, 1, -1, 1, 0, 0])
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_title(title)
            ax.view_init(elev=90, azim=0)  # Top-down view
        except Exception as e:
            ax.text(0.5, 0.5, f"Top View\nError: {e}", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)

    def _plot_side_view(self, ax, img, title):
        """Plot side view"""
        try:
            ax.imshow(img, aspect='auto', extent=[-1, 1, 0, 0, -1, 1])
            ax.set_xlabel('PC1')
            ax.set_zlabel('PC3')
            ax.set_title(title)
            ax.view_init(elev=0, azim=0)  # Side view
        except Exception as e:
            ax.text(0.5, 0.5, f"Side View\nError: {e}", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)

    def _plot_progress_panel(self, ax, epoch, epoch_index, total_epochs, rotation_angle):
        """Plot progress and metrics panel"""
        try:
            # Clear the axis
            ax.clear()
            ax.axis('off')

            # Progress information
            progress_text = f"""
    ADAPTIVE LEARNING PROGRESS
    ──────────────────────────
    Epoch: {epoch}/{total_epochs}
    Progress: {(epoch_index + 1) / total_epochs * 100:.1f}%
    Rotation: {rotation_angle:.0f}°

    CURRENT STATS
    ─────────────
    Training Size: {len(self.training_indices) if hasattr(self, 'training_indices') else 'N/A'}
    Test Accuracy: {self.best_accuracy:.4f}
    Failed Examples: {getattr(self, 'failed_count', 'N/A')}

    FEATURE EVOLUTION
    ────────────────
    Grids Active: {getattr(self, 'active_grids', 'N/A')}
    Learning Rate: {self.adaptive_config.get('learning_rate', 1.0):.2f}
            """

            ax.text(0.05, 0.95, progress_text, transform=ax.transAxes, fontfamily='monospace',
                   verticalalignment='top', fontsize=10, bbox=dict(boxstyle="round,pad=0.3",
                   facecolor="lightblue", alpha=0.7))

            # Add mini progress bar
            progress = (epoch_index + 1) / total_epochs
            progress_bar = plt.Rectangle((0.05, 0.02), progress * 0.9, 0.05,
                                       facecolor='green', alpha=0.7)
            ax.add_patch(progress_bar)
            ax.text(0.5, 0.045, f'{progress * 100:.1f}%', transform=ax.transAxes,
                   ha='center', fontweight='bold')

        except Exception as e:
            ax.text(0.5, 0.5, f"Progress Panel\nError: {e}", ha='center', va='center',
                   transform=ax.transAxes)

    def _compile_rotating_gif(self):
        """Compile all rotation frames into a single animated GIF"""
        try:
            # Get all rotation frames
            rotation_files = sorted([f for f in os.listdir(f'{self.visualization_output_dir}/3d_animations')
                                   if f.startswith('rotation_frame_') and f.endswith('.png')])

            if not rotation_files:
                print("⚠️ No rotation frames found")
                return

            print(f"🎞️ Compiling {len(rotation_files)} frames into animated GIF...")

            # Create GIF with optimal settings
            images = []
            for rotation_file in rotation_files:
                image_path = f'{self.visualization_output_dir}/3d_animations/{rotation_file}'
                images.append(imageio.imread(image_path))

            # Save main animation
            gif_path = f'{self.visualization_output_dir}/adaptive_learning_3d_evolution.gif'
            imageio.mimsave(gif_path, images, duration=0.3, loop=0)  # 0.3s per frame, infinite loop

            # Create a faster preview version
            preview_gif_path = f'{self.visualization_output_dir}/adaptive_learning_preview.gif'
            preview_images = images[::2]  # Use every second frame for faster preview
            imageio.mimsave(preview_gif_path, preview_images, duration=0.2, loop=0)

            # Clean up temporary rotation frames
            self._cleanup_temp_files(rotation_files)

            print(f"✅ Main animation saved: {gif_path}")
            print(f"✅ Preview animation saved: {preview_gif_path}")
            print(f"📊 Animation stats: {len(images)} frames, {len(images)*0.3:.1f}s total duration")

        except Exception as e:
            print(f"⚠️ Error compiling GIF: {e}")

    def _cleanup_temp_files(self, rotation_files):
        """Clean up temporary rotation frame files"""
        try:
            for rotation_file in rotation_files:
                file_path = f'{self.visualization_output_dir}/3d_animations/{rotation_file}'
                if os.path.exists(file_path):
                    os.remove(file_path)
            print("✅ Cleaned up temporary rotation frames")
        except Exception as e:
            print(f"⚠️ Could not clean up temp files: {e}")

    def _create_comprehensive_3d_visualization(self, X_train: np.ndarray, y_train: np.ndarray):
        """Create a comprehensive 3D visualization with actual data"""
        try:
            if X_train is None or y_train is None:
                return

            print("🎨 Creating comprehensive 3D data visualization...")

            # Use PCA for 3D projection
            pca = PCA(n_components=3)
            X_3d = pca.fit_transform(X_train)

            # Create rotating animation with actual data
            fig = plt.figure(figsize(15, 10))

            def animate(frame):
                plt.clf()

                # Create subplots
                ax1 = fig.add_subplot(221, projection='3d')
                ax2 = fig.add_subplot(222, projection='3d')
                ax3 = fig.add_subplot(223)
                ax4 = fig.add_subplot(224)

                # Rotation angle
                angle = frame * 2  # 2 degrees per frame

                # Plot 1: Rotating 3D scatter
                scatter = ax1.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2],
                                     c=y_train, cmap='tab10', alpha=0.7, s=20)
                ax1.view_init(elev=20, azim=angle)
                ax1.set_title(f'3D Feature Space\nRotation: {angle}°')
                ax1.set_xlabel('PC1')
                ax1.set_ylabel('PC2')
                ax1.set_zlabel('PC3')

                # Plot 2: Top view
                ax2.scatter(X_3d[:, 0], X_3d[:, 1], c=y_train, cmap='tab10', alpha=0.7, s=20)
                ax2.set_title('Top View (PC1 vs PC2)')
                ax2.set_xlabel('PC1')
                ax2.set_ylabel('PC2')

                # Plot 3: Feature distribution
                for i in range(min(3, X_train.shape[1])):
                    ax3.hist(X_train[:, i], alpha=0.7, bins=30, label=f'Feature {i+1}')
                ax3.set_title('Feature Distributions')
                ax3.legend()

                # Plot 4: Class distribution
                unique, counts = np.unique(y_train, return_counts=True)
                ax4.bar(unique, counts, alpha=0.7, color=plt.cm.tab10(unique/len(unique)))
                ax4.set_title('Class Distribution')
                ax4.set_xlabel('Class')
                ax4.set_ylabel('Count')

                plt.tight_layout()

            # Create animation
            anim = animation.FuncAnimation(fig, animate, frames=180, interval=50, blit=False)

            # Save as GIF
            anim_path = f'{self.visualization_output_dir}/data_3d_evolution.gif'
            anim.save(anim_path, writer='pillow', fps=20, dpi=100)

            plt.close()
            print(f"✅ Data 3D animation saved: {anim_path}")

        except Exception as e:
            print(f"⚠️ Error creating comprehensive 3D visualization: {e}")

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
            print(f"  {key:40}: {value}")
        print(f"\n💻 Device: {self.device_type}")

        # Show protection status
        min_percent = self.adaptive_config.get('min_training_percentage_for_stopping', 10.0)
        max_percent = self.adaptive_config.get('max_training_percentage', 90.0)
        print(f"🛡️  Early stopping protection: <{min_percent}% training data")
        print(f"📊 Max training data allowed: {max_percent}%")
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

            # NEW: Early stopping relaxation parameters
            min_stopping_percentage = float(input(
                f"Min training data % for early stopping [{self.adaptive_config.get('min_training_percentage_for_stopping', 10.0)}]: "
            ) or self.adaptive_config.get('min_training_percentage_for_stopping', 10.0))

            max_training_percentage = float(input(
                f"Max training data % allowed [{self.adaptive_config.get('max_training_percentage', 90.0)}]: "
            ) or self.adaptive_config.get('max_training_percentage', 90.0))

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

            # FIX: Handle label encoding mismatch - map predictions to valid range
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
                  f"(max posterior: {max_posterior_sample['true_posterior']:.3f}, "
                  f"min margin: {min_margin_sample['margin']:.3f})")

        print(f"🎯 Selected {len(samples_to_add)} samples total using simple margin strategy")
        return samples_to_add

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

            # FIX: Handle label encoding mismatch - map predictions to valid range
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
                  f"(max posterior: {max_posterior_sample['true_posterior']:.3f}, "
                  f"min margin: {min_margin_sample['margin']:.3f})")

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

    def _create_interactive_html_dashboard(self, X_train: np.ndarray, y_train: np.ndarray,
                                        X_test: np.ndarray, y_test: np.ndarray):
        """Create an interactive HTML dashboard with comprehensive visualization"""
        try:
            print("🖥️  Creating interactive HTML dashboard...")

            # Create the dashboard
            fig = make_subplots(
                rows=3, cols=3,
                specs=[
                    [{"type": "scatter3d", "rowspan": 2}, {"type": "scatter", "rowspan": 2}, {"type": "scatter"}],
                    [None, None, {"type": "scatter"}],
                    [{"type": "heatmap", "colspan": 2}, None, {"type": "bar"}]
                ],
                subplot_titles=(
                    '3D Feature Space Evolution', 'Learning Progress', 'Feature Importance',
                    'Class Distribution', 'Confusion Matrix', 'Sample Selection Analysis'
                ),
                vertical_spacing=0.08,
                horizontal_spacing=0.08
            )

            # 1. 3D Feature Space (placeholder - would use actual data)
            if len(self.feature_grid_history) > 1:
                self._add_3d_evolution_to_dashboard(fig)

            # 2. Learning Progress
            self._add_learning_progress_to_dashboard(fig)

            # 3. Feature Importance
            self._add_feature_importance_to_dashboard(fig)

            # 4. Class Distribution
            self._add_class_distribution_to_dashboard(fig, y_train)

            # 5. Confusion Matrix
            self._add_confusion_matrix_to_dashboard(fig, X_test, y_test)

            # 6. Sample Selection Analysis
            self._add_sample_selection_to_dashboard(fig)

            # Update layout
            fig.update_layout(
                title_text=f"Adaptive DBNN Learning Dashboard - {self.dataset_name}",
                height=1200,
                showlegend=True,
                template="plotly_white",
                font=dict(size=10)
            )

            # Add summary annotations
            fig.add_annotation(
                text=f"<b>Final Results:</b><br>Best Accuracy: {self.best_accuracy:.4f}<br>"
                     f"Rounds: {len(self.round_stats)}<br>Training Size: {len(X_train)}<br>"
                     f"Device: {self.device_type}",
                xref="paper", yref="paper",
                x=0.98, y=0.02,
                showarrow=False,
                bgcolor="lightblue",
                bordercolor="black",
                borderwidth=1
            )

            # Save interactive dashboard
            dashboard_path = f"{self.viz_config.get('output_dir', 'adaptive_visualizations')}/interactive_dashboard.html"
            fig.write_html(dashboard_path)

            print(f"✅ Interactive dashboard saved: {dashboard_path}")

        except Exception as e:
            print(f"⚠️ Could not create interactive dashboard: {e}")

    def _add_learning_progress_to_dashboard(self, fig):
        """Add learning progress plots to dashboard"""
        if len(self.round_stats) < 2:
            return

        rounds = [s['round'] for s in self.round_stats]
        accuracies = [s['accuracy'] for s in self.round_stats]
        full_accuracies = [s.get('full_dataset_accuracy', 0) for s in self.round_stats]
        training_sizes = [s['training_size'] for s in self.round_stats]

        # Test accuracy vs Full dataset accuracy
        fig.add_trace(
            go.Scatter(x=rounds, y=accuracies, mode='lines+markers',
                      name='Test Accuracy', line=dict(color='blue')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=rounds, y=full_accuracies, mode='lines+markers',
                      name='Full Dataset Accuracy', line=dict(color='red')),
            row=1, col=2
        )

        # Training size progression
        fig.add_trace(
            go.Scatter(x=rounds, y=training_sizes, mode='lines+markers',
                      name='Training Size', line=dict(color='green')),
            row=2, col=3
        )

        fig.update_xaxes(title_text="Round", row=1, col=2)
        fig.update_yaxes(title_text="Accuracy", row=1, col=2)
        fig.update_xaxes(title_text="Round", row=2, col=3)
        fig.update_yaxes(title_text="Samples", row=2, col=3)

    def _add_feature_importance_to_dashboard(self, fig):
        """Add feature importance visualization to dashboard"""
        if len(self.round_stats) < 2:
            return

        # Use the last round's feature importance if available
        last_stats = self.round_stats[-1]
        if 'feature_importance' in last_stats and last_stats['feature_importance'] is not None:
            importance = last_stats['feature_importance']
            features = [f'Feature {i+1}' for i in range(len(importance))]

            fig.add_trace(
                go.Bar(x=features, y=importance, name='Feature Importance'),
                row=1, col=3
            )
            fig.update_xaxes(title_text="Features", row=1, col=3)
            fig.update_yaxes(title_text="Importance", row=1, col=3)

    def _add_class_distribution_to_dashboard(self, fig, y_train):
        """Add class distribution to dashboard"""
        unique, counts = np.unique(y_train, return_counts=True)
        class_names = [f'Class {cls}' for cls in unique]

        fig.add_trace(
            go.Pie(labels=class_names, values=counts, name='Class Distribution'),
            row=2, col=3
        )

    def _add_confusion_matrix_to_dashboard(self, fig, X_test, y_test):
        """Add confusion matrix to dashboard"""
        try:
            y_pred = self._predict_with_current_model(X_test)
            cm = confusion_matrix(y_test, y_pred)

            fig.add_trace(
                go.Heatmap(z=cm, colorscale='Blues', showscale=True),
                row=3, col=1
            )
            fig.update_xaxes(title_text="Predicted", row=3, col=1)
            fig.update_yaxes(title_text="Actual", row=3, col=1)
        except Exception as e:
            print(f"⚠️ Could not add confusion matrix: {e}")

    def _add_sample_selection_to_dashboard(self, fig):
        """Add sample selection analysis to dashboard"""
        if not self.all_selected_samples:
            return

        # Analyze selection patterns
        selection_counts = {}
        for class_samples in self.all_selected_samples.values():
            for sample in class_samples:
                sel_type = sample.get('selection_type', 'unknown')
                selection_counts[sel_type] = selection_counts.get(sel_type, 0) + 1

        if selection_counts:
            fig.add_trace(
                go.Bar(x=list(selection_counts.keys()), y=list(selection_counts.values()),
                      name='Selection Types'),
                row=3, col=3
            )
            fig.update_xaxes(title_text="Selection Type", row=3, col=3)
            fig.update_yaxes(title_text="Count", row=3, col=3)

    def _add_3d_evolution_to_dashboard(self, fig):
        """Add 3D evolution visualization (placeholder)"""
        # This would be populated with actual 3D data from feature_grid_history
        fig.add_trace(
            go.Scatter3d(x=[0, 1, 2], y=[0, 1, 2], z=[0, 1, 2],
                        mode='markers', name='Feature Evolution'),
            row=1, col=1
        )

    def adaptive_learn(self, X: np.ndarray = None, y: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Main adaptive learning loop with acid test validation and interactive visualization"""
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
            effective_max_rounds = 1000
        else:
            print(f"🔄 MODE: PATIENCE - Max rounds: {self.adaptive_config['max_adaptive_rounds']}")
            print(f"⏳ Patience: {self.adaptive_config['patience']}")
            effective_max_rounds = self.adaptive_config['max_adaptive_rounds']

        # Initialize adaptive learning
        X_train, y_train, X_test, y_test = self.prepare_adaptive_data(X, y)
        self.adaptive_start_time = datetime.now()

        # Track failed examples history
        failed_history = []

        # Track performance on entire dataset
        full_dataset_accuracies = []
        best_full_dataset_accuracy = 0.0

        # Track 3D visualization data
        if hasattr(self, '_initialize_3d_visualization'):
            self._initialize_3d_visualization()

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
            if round_num > 1000:
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

            # ACID TEST - Evaluate on entire dataset
            print("🧪 Performing acid test on entire dataset...")
            full_predictions = self._predict_with_current_model(X)
            full_accuracy = accuracy_score(y, full_predictions)
            full_dataset_accuracies.append(full_accuracy)
            print(f"🧪 Entire Dataset Accuracy: {full_accuracy:.4f}")

            # Track 3D visualization data
            if hasattr(self, '_track_feature_grids_3d'):
                if round_num % 5 == 0 or round_num <= 10:  # Track more frequently in early rounds
                    self._track_feature_grids_3d(round_num, X_train, y_train)

            # Create round visualizations
            if self.stats_config.get('enable_confusion_matrix', True) and round_num <= 20:
                self._create_round_visualizations(round_num, y_test, predictions)

            # Step 3: Save best model based on ACID TEST (entire dataset accuracy)
            improvement_threshold = self.adaptive_config['min_improvement']

            # Use entire dataset accuracy for validation
            acid_test_improvement = full_accuracy > best_full_dataset_accuracy + improvement_threshold

            if acid_test_improvement:
                improvement = full_accuracy - best_full_dataset_accuracy
                best_full_dataset_accuracy = full_accuracy
                self.best_accuracy = accuracy  # Still track test accuracy for display
                self.best_round = round_num
                self.best_training_indices = self.training_indices.copy()
                self.best_test_indices = self.test_indices.copy()
                self.best_weights = self.model.current_W.copy() if hasattr(self.model, 'current_W') else None
                self.patience_counter = 0  # Reset patience when accuracy improves

                print(f"🎯 NEW BEST MODEL (ACID TEST): {full_accuracy:.4f} (improvement: {improvement:.4f})")
                print(f"📦 Best training set size: {len(self.best_training_indices)}")

                # Save best weights
                if self.best_weights is not None:
                    self.model._save_best_weights()
            else:
                self.patience_counter += 1
                if not exhaust_all_failed:
                    print(f"⏳ No acid test improvement - Patience: {self.patience_counter}/{patience}")

                # Show why we're not improving
                if full_accuracy <= best_full_dataset_accuracy:
                    print(f"   📉 Entire dataset accuracy didn't improve: {full_accuracy:.4f} <= {best_full_dataset_accuracy:.4f}")
                else:
                    print(f"   📈 Improvement too small: {full_accuracy - best_full_dataset_accuracy:.4f} < {improvement_threshold}")

            # Step 4: Select informative samples for next round
            print("🎯 Selecting informative samples...")
            samples_to_add = self._select_informative_samples(X, y, predictions, posteriors)

            # Step 5: Check stopping conditions
            stop_reason = self._check_stopping_conditions(
                round_num, samples_to_add, failed_count, min_failed_threshold,
                patience, exhaust_all_failed, effective_max_rounds,
                full_accuracy, best_full_dataset_accuracy  # Pass acid test results
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
                                        accuracy, samples_to_add, round_duration,
                                        failed_count, full_accuracy)  # Add full accuracy

            print(f"📥 Added {added_count} samples to training set")
            print(f"📊 New training size: {len(X_train)}")
            print(f"📊 New test size: {len(X_test)}")
            print(f"❌ Remaining failed examples: {failed_count}")
            print(f"🧪 Entire dataset accuracy: {full_accuracy:.4f}")
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

        print(f"🎯 Best test accuracy: {self.best_accuracy:.4f} (achieved at round {self.best_round})")
        print(f"🧪 Best entire dataset accuracy: {best_full_dataset_accuracy:.4f}")
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

        # NEW: Create interactive HTML visualization dashboard
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
                                 current_full_accuracy: float, best_full_accuracy: float) -> str:
        """Check all stopping conditions with relaxation for early training"""

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

            # Relax acid test degradation check until we have sufficient training data
            degradation_threshold = 0.1  # 10% degradation
            if training_percentage >= min_stopping_percentage:
                if current_full_accuracy < best_full_accuracy - degradation_threshold:
                    return f"Significant degradation in entire dataset accuracy: {current_full_accuracy:.4f} < {best_full_accuracy:.4f}"
            else:
                print(f"   🛡️  Acid test degradation check suspended (training data: {training_percentage:.1f}% < {min_stopping_percentage}%)")

            # Continue learning regardless of rounds or patience
            return ""

        # 3. Max rounds
        if round_num >= max_rounds:
            return f"Reached maximum rounds ({max_rounds})"

        # 4. Patience-based stopping (only if not in exhaustive mode)
        if not exhaust_all_failed and self.patience_counter >= patience:
            # Relax patience stopping until sufficient training data
            if training_percentage >= min_stopping_percentage:
                return f"Early stopping after {patience} rounds without acid test improvement"
            else:
                print(f"   🛡️  Patience stopping suspended (training data: {training_percentage:.1f}% < {min_stopping_percentage}%)")
                # Reset patience counter to give more time
                self.patience_counter = max(0, self.patience_counter - 1)
                return ""

        # 5. Exhaustive mode: stop when failed examples are below threshold
        if exhaust_all_failed and failed_count <= min_failed_threshold:
            return f"Failed examples ({failed_count}) below threshold ({min_failed_threshold})"

        # 6. Exhaustive mode: stop when test set is too small
        if exhaust_all_failed and len(self.test_indices) <= min_failed_threshold * 2:
            return f"Test set too small ({len(self.test_indices)} samples)"

        # 7. Stop if training set reaches maximum allowed percentage
        if training_percentage >= max_training_percentage:
            return f"Training set reached {training_percentage:.1f}% of total data - stopping to prevent overfitting"

        return ""  # Continue learning

    def _record_round_statistics(self, round_num: int, X_train: np.ndarray, y_train: np.ndarray,
                               X_test: np.ndarray, y_test: np.ndarray, accuracy: float,
                               samples_added: List[int], duration: float, failed_count: int,
                               full_accuracy: float):  # NEW: Add full_accuracy parameter
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
            'full_dataset_accuracy': full_accuracy,  # NEW: Track entire dataset accuracy
            'samples_added': len(samples_added),
            'class_distribution': class_distribution,
            'duration': duration,
            'failed_count': failed_count,
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
