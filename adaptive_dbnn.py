import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
import json
from collections import defaultdict
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import dbnn

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import networkx as nx

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
import scipy.stats as stats

import multiprocessing
import queue
import threading

class AdvancedInteractiveVisualizer:
    """Advanced interactive 3D visualization with dynamic controls"""

    def __init__(self, dataset_name, output_base_dir='Visualizer/adaptiveDBNN'):
        self.dataset_name = dataset_name
        self.output_dir = Path(output_base_dir) / dataset_name / 'interactive_3d'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.colors = px.colors.qualitative.Set1 + px.colors.qualitative.Pastel

    def create_advanced_3d_dashboard(self, X_full, y_full, training_history, feature_names, round_num=None):
        """Create advanced interactive 3D dashboard with multiple visualization options"""
        print("🌐 Creating advanced interactive 3D dashboard...")

        # Create multiple visualization methods
        self._create_pca_3d_plot(X_full, y_full, training_history, feature_names, round_num)
        self._create_feature_space_3d(X_full, y_full, training_history, feature_names, round_num)
        self._create_network_graph_3d(X_full, y_full, training_history, feature_names, round_num)
        self._create_density_controlled_3d(X_full, y_full, training_history, feature_names, round_num)

        # Create main dashboard that links all visualizations
        self._create_main_dashboard(X_full, y_full, training_history, feature_names, round_num)

    def _create_pca_3d_plot(self, X_full, y_full, training_history, feature_names, round_num):
        """Create PCA-based 3D plot with interactive controls"""
        from sklearn.decomposition import PCA

        # Reduce dimensions
        pca = PCA(n_components=3, random_state=42)
        X_3d = pca.fit_transform(X_full)
        explained_var = pca.explained_variance_ratio_

        # Create interactive plot
        unique_classes = np.unique(y_full)
        fig = go.Figure()

        for i, cls in enumerate(unique_classes):
            class_mask = y_full == cls
            scatter = go.Scatter3d(
                x=X_3d[class_mask, 0],
                y=X_3d[class_mask, 1],
                z=X_3d[class_mask, 2],
                mode='markers',
                marker=dict(
                    size=4,
                    color=self.colors[i % len(self.colors)],
                    opacity=0.7,
                    line=dict(width=0.5, color='black')
                ),
                name=f'Class {cls}',
                text=[f'Class: {cls}<br>PC1: {x:.3f}<br>PC2: {y:.3f}<br>PC3: {z:.3f}'
                      for x, y, z in zip(X_3d[class_mask, 0], X_3d[class_mask, 1], X_3d[class_mask, 2])],
                hoverinfo='text'
            )
            fig.add_trace(scatter)

        # Add network connections for training samples
        if training_history and len(training_history) > 0:
            training_indices = training_history[-1] if round_num is None else training_history[round_num]
            self._add_network_connections_3d(fig, X_3d, y_full, training_indices)

        fig.update_layout(
            title=f'3D PCA Visualization - {self.dataset_name}<br>'
                  f'Explained Variance: PC1: {explained_var[0]:.3f}, PC2: {explained_var[1]:.3f}, PC3: {explained_var[2]:.3f}',
            scene=dict(
                xaxis_title=f'PC1 ({explained_var[0]:.2%} variance)',
                yaxis_title=f'PC2 ({explained_var[1]:.2%} variance)',
                zaxis_title=f'PC3 ({explained_var[2]:.2%} variance)',
            ),
            width=1000,
            height=800
        )

        filename = f'pca_3d_round_{round_num}.html' if round_num else 'pca_3d_final.html'
        fig.write_html(self.output_dir / filename)

    def _create_feature_space_3d(self, X_full, y_full, training_history, feature_names, round_num):
        """Create feature space 3D plot with selectable features"""
        # Allow selection of any 3 features for visualization
        if len(feature_names) >= 3:
            # Use first 3 features by default, but create interface for selection
            feature_indices = [0, 1, 2]
            selected_features = [feature_names[i] for i in feature_indices]

            fig = go.Figure()
            unique_classes = np.unique(y_full)

            for i, cls in enumerate(unique_classes):
                class_mask = y_full == cls
                scatter = go.Scatter3d(
                    x=X_full[class_mask, feature_indices[0]],
                    y=X_full[class_mask, feature_indices[1]],
                    z=X_full[class_mask, feature_indices[2]],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=self.colors[i % len(self.colors)],
                        opacity=0.6,
                        symbol='circle'
                    ),
                    name=f'Class {cls}',
                    text=[f'Class: {cls}<br>{selected_features[0]}: {x:.3f}<br>{selected_features[1]}: {y:.3f}<br>{selected_features[2]}: {z:.3f}'
                          for x, y, z in zip(X_full[class_mask, feature_indices[0]],
                                           X_full[class_mask, feature_indices[1]],
                                           X_full[class_mask, feature_indices[2]])],
                    hoverinfo='text'
                )
                fig.add_trace(scatter)

            fig.update_layout(
                title=f'3D Feature Space - {self.dataset_name}<br>Features: {selected_features}',
                scene=dict(
                    xaxis_title=selected_features[0],
                    yaxis_title=selected_features[1],
                    zaxis_title=selected_features[2],
                ),
                width=1000,
                height=800
            )

            filename = f'feature_3d_round_{round_num}.html' if round_num else 'feature_3d_final.html'
            fig.write_html(self.output_dir / filename)

    def _add_network_connections_3d(self, fig, X_3d, y_full, training_indices):
        """Add network connections between training samples"""
        from scipy.spatial import distance_matrix
        import networkx as nx

        training_mask = np.isin(range(len(X_3d)), training_indices)
        X_train = X_3d[training_mask]
        y_train = y_full[training_mask]

        unique_classes = np.unique(y_train)

        for i, cls in enumerate(unique_classes):
            class_mask = y_train == cls
            class_points = X_train[class_mask]

            if len(class_points) < 2:
                continue

            try:
                # Create minimum spanning tree
                dist_matrix = distance_matrix(class_points, class_points)
                G = nx.Graph()

                for j in range(len(class_points)):
                    for k in range(j+1, len(class_points)):
                        if dist_matrix[j, k] < np.percentile(dist_matrix, 25):  # Connect only close points
                            G.add_edge(j, k, weight=dist_matrix[j, k])

                if G.number_of_edges() > 0:
                    mst = nx.minimum_spanning_tree(G)

                    # Add edges to plot
                    for edge in mst.edges():
                        x_edges = [class_points[edge[0], 0], class_points[edge[1], 0], None]
                        y_edges = [class_points[edge[0], 1], class_points[edge[1], 1], None]
                        z_edges = [class_points[edge[0], 2], class_points[edge[1], 2], None]

                        fig.add_trace(go.Scatter3d(
                            x=x_edges, y=y_edges, z=z_edges,
                            mode='lines',
                            line=dict(color=self.colors[i % len(self.colors)], width=2, opacity=0.6),
                            showlegend=False,
                            hoverinfo='none'
                        ))
            except Exception:
                continue

    def _create_density_controlled_3d(self, X_full, y_full, training_history, feature_names, round_num):
        """Create density-controlled 3D visualization with point skipping"""
        from sklearn.decomposition import PCA

        pca = PCA(n_components=3, random_state=42)
        X_3d = pca.fit_transform(X_full)

        # Apply density-based sampling
        X_sampled, y_sampled = self._density_based_sampling(X_3d, y_full, max_points_per_class=100)

        fig = go.Figure()
        unique_classes = np.unique(y_sampled)

        for i, cls in enumerate(unique_classes):
            class_mask = y_sampled == cls
            scatter = go.Scatter3d(
                x=X_sampled[class_mask, 0],
                y=X_sampled[class_mask, 1],
                z=X_sampled[class_mask, 2],
                mode='markers',
                marker=dict(
                    size=6,
                    color=self.colors[i % len(self.colors)],
                    opacity=0.8,
                    line=dict(width=1, color='black')
                ),
                name=f'Class {cls} (density-controlled)',
                text=[f'Class: {cls}' for _ in range(np.sum(class_mask))],
                hoverinfo='text'
            )
            fig.add_trace(scatter)

        fig.update_layout(
            title=f'Density-Controlled 3D Visualization - {self.dataset_name}<br>'
                  f'Points sampled to reduce overcrowding',
            scene=dict(
                xaxis_title='PC1',
                yaxis_title='PC2',
                zaxis_title='PC3',
            ),
            width=1000,
            height=800
        )

        filename = f'density_3d_round_{round_num}.html' if round_num else 'density_3d_final.html'
        fig.write_html(self.output_dir / filename)

    def _density_based_sampling(self, X, y, max_points_per_class=100, min_distance_ratio=0.1):
        """Sample points based on density to reduce overcrowding"""
        from sklearn.neighbors import NearestNeighbors

        unique_classes = np.unique(y)
        X_sampled_list = []
        y_sampled_list = []

        for cls in unique_classes:
            class_mask = y == cls
            X_class = X[class_mask]

            if len(X_class) <= max_points_per_class:
                # No sampling needed
                X_sampled_list.append(X_class)
                y_sampled_list.append(np.full(len(X_class), cls))
            else:
                # Use k-nearest neighbors to sample diverse points
                nbrs = NearestNeighbors(n_neighbors=min(10, len(X_class)), algorithm='auto').fit(X_class)
                distances, indices = nbrs.kneighbors(X_class)

                # Use average distance to neighbors as density measure
                avg_distances = np.mean(distances, axis=1)

                # Select points with higher average distances (less crowded)
                density_scores = 1 / (avg_distances + 1e-8)  # Avoid division by zero

                # Sample points inversely proportional to density
                probabilities = 1 / (density_scores + 1e-8)
                probabilities = probabilities / np.sum(probabilities)

                selected_indices = np.random.choice(
                    len(X_class),
                    size=max_points_per_class,
                    replace=False,
                    p=probabilities
                )

                X_sampled_list.append(X_class[selected_indices])
                y_sampled_list.append(np.full(max_points_per_class, cls))

        return np.vstack(X_sampled_list), np.hstack(y_sampled_list)

    def _create_main_dashboard(self, X_full, y_full, training_history, feature_names, round_num):
        """Create main dashboard linking all visualizations"""
        dashboard_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Advanced 3D Visualization Dashboard - {dataset_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                         color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
                .nav {{ display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }}
                .nav-button {{ padding: 10px 20px; background: #4CAF50; color: white;
                            border: none; border-radius: 5px; cursor: pointer; text-decoration: none; }}
                .nav-button:hover {{ background: #45a049; }}
                .iframe-container {{ border: 1px solid #ddd; border-radius: 5px; margin-bottom: 20px; }}
                iframe {{ width: 100%; height: 800px; border: none; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🌐 Advanced 3D Visualization Dashboard</h1>
                <h2>Dataset: {dataset_name}</h2>
                <p>Round: {round_info} | Features: {feature_count} | Samples: {sample_count}</p>
            </div>

            <div class="nav">
                <a class="nav-button" href="#pca">PCA 3D</a>
                <a class="nav-button" href="#feature">Feature Space 3D</a>
                <a class="nav-button" href="#density">Density-Controlled 3D</a>
                <a class="nav-button" href="#network">Network Graph</a>
            </div>

            <div id="pca" class="iframe-container">
                <h3>📊 PCA 3D Visualization</h3>
                <iframe src="pca_3d_{round_suffix}.html"></iframe>
            </div>

            <div id="feature" class="iframe-container">
                <h3>🔧 Feature Space 3D</h3>
                <iframe src="feature_3d_{round_suffix}.html"></iframe>
            </div>

            <div id="density" class="iframe-container">
                <h3>📈 Density-Controlled 3D</h3>
                <iframe src="density_3d_{round_suffix}.html"></iframe>
            </div>

            <script>
                // Smooth scrolling for navigation
                document.querySelectorAll('.nav-button').forEach(button => {{
                    button.addEventListener('click', function(e) {{
                        e.preventDefault();
                        const targetId = this.getAttribute('href').substring(1);
                        document.getElementById(targetId).scrollIntoView({{
                            behavior: 'smooth'
                        }});
                    }});
                }});
            </script>
        </body>
        </html>
        """.format(
            dataset_name=self.dataset_name,
            round_info=f"Round {round_num}" if round_num else "Final",
            feature_count=len(feature_names),
            sample_count=len(X_full),
            round_suffix=f"round_{round_num}" if round_num else "final"
        )

        with open(self.output_dir / f"dashboard_{'round_' + str(round_num) if round_num else 'final'}.html", "w") as f:
            f.write(dashboard_html)

class ComprehensiveAdaptiveVisualizer:
    """Comprehensive visualization system for Adaptive DBNN with intuitive plots"""

    def __init__(self, dataset_name, output_base_dir='Visualizer/adaptiveDBNN'):
        self.dataset_name = dataset_name
        self.output_dir = Path(output_base_dir) / dataset_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for different plot types
        self.subdirs = {
            'performance': self.output_dir / 'performance',
            'samples': self.output_dir / 'sample_evolution',
            'distributions': self.output_dir / 'distributions',
            'networks': self.output_dir / 'networks',
            'comparisons': self.output_dir / 'comparisons',
            'interactive': self.output_dir / 'interactive'
        }

        for subdir in self.subdirs.values():
            subdir.mkdir(exist_ok=True)

        # Color schemes
        self.colors = px.colors.qualitative.Set1
        self.set_plot_style()

        print(f"🎨 Comprehensive visualizer initialized for: {dataset_name}")
        print(f"📁 Output directory: {self.output_dir}")

    def set_plot_style(self):
        """Set consistent plot style with safe colors"""
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12

        # Use safe colors that work with both matplotlib and plotly
        self.colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]

    def create_comprehensive_visualizations(self, adaptive_model, X_full, y_full,
                                         training_history, round_stats, feature_names):
        """Create all comprehensive visualizations"""
        print("\n" + "="*60)
        print("🎨 CREATING COMPREHENSIVE ADAPTIVE DBNN VISUALIZATIONS")
        print("="*60)

        # 1. Performance Evolution
        self.plot_performance_evolution(round_stats)

        # 2. Sample Selection Analysis
        self.plot_sample_selection_analysis(training_history, y_full)

        # 3. Training Sample Distributions
        self.plot_training_sample_distributions(X_full, y_full, training_history, feature_names)

        # 4. 3D Network Visualizations
        self.plot_3d_networks(X_full, y_full, training_history, feature_names)

        # 5. Feature Importance Analysis
        self.plot_feature_importance_analysis(adaptive_model, X_full, y_full, feature_names)

        # 6. Class Separation Analysis
        self.plot_class_separation_analysis(X_full, y_full, training_history)

        # 7. Confidence Evolution
        self.plot_confidence_evolution(adaptive_model, X_full, y_full, training_history)

        # 8. Interactive Dashboard
        self.create_interactive_dashboard(round_stats, training_history, X_full, y_full, feature_names)

        # 9. Final Model Analysis
        self.plot_final_model_analysis(adaptive_model, X_full, y_full, feature_names)

        print(f"✅ All visualizations saved to: {self.output_dir}")

    def plot_performance_evolution(self, round_stats):
        """Plot comprehensive performance evolution across rounds - OPTIMIZED"""
        print("📈 Creating performance evolution plots...")

        if not round_stats:
            return

        rounds = [stat['round'] for stat in round_stats]
        train_acc = [stat['train_accuracy'] * 100 for stat in round_stats]
        test_acc = [stat['test_accuracy'] * 100 for stat in round_stats]
        training_sizes = [stat['training_size'] for stat in round_stats]
        improvements = [stat['improvement'] * 100 for stat in round_stats]

        # Create subplots with optimized layout
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Accuracy Evolution - OPTIMIZED LEGEND
        line1, = ax1.plot(rounds, train_acc, 'o-', linewidth=2, markersize=6,
                         label='Training Accuracy', color=self.colors[0])
        line2, = ax1.plot(rounds, test_acc, 's-', linewidth=2, markersize=6,
                         label='Test Accuracy', color=self.colors[1])

        # Highlight best round without legend
        best_round_idx = np.argmax(test_acc)
        ax1.axvline(x=rounds[best_round_idx], color='red', linestyle='--', alpha=0.7)

        ax1.set_xlabel('Adaptive Round')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Accuracy Evolution Across Rounds', fontweight='bold', fontsize=14)

        # Use manual legend positioning instead of loc="best"
        ax1.legend([line1, line2], ['Training Accuracy', 'Test Accuracy'],
                   loc='upper left', frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Training Size Growth
        ax2.plot(rounds, training_sizes, '^-', linewidth=2, markersize=6, color=self.colors[2])
        ax2.set_xlabel('Adaptive Round')
        ax2.set_ylabel('Training Set Size')
        ax2.set_title('Training Set Growth', fontweight='bold', fontsize=14)
        ax2.grid(True, alpha=0.3)

        # Add percentage growth annotation
        if len(training_sizes) > 1:
            growth_pct = ((training_sizes[-1] - training_sizes[0]) / training_sizes[0]) * 100
            ax2.annotate(f'+{growth_pct:.1f}% growth',
                        xy=(rounds[-1], training_sizes[-1]),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        fontsize=9)

        # Plot 3: Improvement per Round
        bars = ax3.bar(rounds, improvements,
                       color=np.where(np.array(improvements) >= 0, 'green', 'red'),
                       alpha=0.7, width=0.6)
        ax3.set_xlabel('Adaptive Round')
        ax3.set_ylabel('Accuracy Improvement (%)')
        ax3.set_title('Accuracy Improvement per Round', fontweight='bold', fontsize=14)
        ax3.grid(True, alpha=0.3)

        # Add value labels on bars - optimized for performance
        for bar, improvement in zip(bars, improvements):
            height = bar.get_height()
            if abs(height) > 0.1:  # Only label significant improvements
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{improvement:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=8)

        # Plot 4: Cumulative Performance
        cumulative_improvement = np.cumsum(improvements)
        ax4.plot(rounds, cumulative_improvement, 'o-', linewidth=2, markersize=6, color=self.colors[3])
        ax4.set_xlabel('Adaptive Round')
        ax4.set_ylabel('Cumulative Improvement (%)')
        ax4.set_title('Cumulative Performance Improvement', fontweight='bold', fontsize=14)
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.subdirs['performance'] / 'performance_evolution.png', dpi=200, bbox_inches='tight')
        plt.savefig(self.subdirs['performance'] / 'performance_evolution.pdf', bbox_inches='tight')
        plt.close()

        # Create interactive version
        self._create_interactive_performance_plot(rounds, train_acc, test_acc, training_sizes, improvements, cumulative_improvement)

    def _create_interactive_performance_plot(self, rounds, train_acc, test_acc, training_sizes, improvements, cumulative_improvement):
        """Create optimized interactive performance plot"""
        fig_int = make_subplots(rows=2, cols=2,
                               subplot_titles=('Accuracy Evolution', 'Training Set Growth',
                                             'Improvement per Round', 'Cumulative Improvement'))

        fig_int.add_trace(go.Scatter(x=rounds, y=train_acc, name='Training Accuracy',
                                   line=dict(color=self.colors[0])), row=1, col=1)
        fig_int.add_trace(go.Scatter(x=rounds, y=test_acc, name='Test Accuracy',
                                   line=dict(color=self.colors[1])), row=1, col=1)

        fig_int.add_trace(go.Scatter(x=rounds, y=training_sizes, name='Training Size',
                                   line=dict(color=self.colors[2])), row=1, col=2)

        fig_int.add_trace(go.Bar(x=rounds, y=improvements, name='Improvement',
                               marker_color=np.where(np.array(improvements) >= 0, 'green', 'red')),
                         row=2, col=1)

        fig_int.add_trace(go.Scatter(x=rounds, y=cumulative_improvement, name='Cumulative Improvement',
                                   line=dict(color=self.colors[3])), row=2, col=2)

        fig_int.update_layout(height=800, title_text="Adaptive Learning Performance Evolution",
                             showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02))
        fig_int.write_html(self.subdirs['interactive'] / 'performance_evolution.html')

    def plot_sample_selection_analysis(self, training_history, y_full):
        """Analyze how samples are selected across rounds - OPTIMIZED"""
        print("🔍 Creating sample selection analysis...")

        if not training_history:
            return

        unique_classes = np.unique(y_full)
        rounds = list(range(1, len(training_history) + 1))

        # Calculate class distribution per round - optimized calculation
        class_distributions = []
        for round_indices in training_history:
            round_labels = y_full[round_indices]
            class_counts = [np.sum(round_labels == cls) for cls in unique_classes]
            class_distributions.append(class_counts)

        class_distributions = np.array(class_distributions)

        # Create optimized plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Stacked area plot for class distribution - optimized
        if len(unique_classes) <= 10:  # Limit for reasonable visualization
            ax1.stackplot(rounds, class_distributions.T,
                         labels=[f'Class {cls}' for cls in unique_classes],
                         colors=self.colors[:len(unique_classes)], alpha=0.8)
            ax1.set_xlabel('Adaptive Round')
            ax1.set_ylabel('Number of Samples')
            ax1.set_title('Class Distribution Evolution', fontweight='bold', fontsize=14)
            # Use fixed legend position
            ax1.legend(loc='upper left', frameon=True, fancybox=True)
        else:
            # For many classes, use line plot instead
            for i, cls in enumerate(unique_classes[:10]):  # Limit to first 10 classes
                ax1.plot(rounds, class_distributions[:, i], 'o-', linewidth=1, markersize=3,
                        label=f'Class {cls}', color=self.colors[i % len(self.colors)])
            ax1.set_xlabel('Adaptive Round')
            ax1.set_ylabel('Number of Samples')
            ax1.set_title('Class Distribution Evolution (Top 10 Classes)', fontweight='bold', fontsize=14)
            ax1.legend(loc='upper left', frameon=True, fancybox=True)

        ax1.grid(True, alpha=0.3)

        # Plot 2: Class Proportion Evolution - optimized
        class_proportions = class_distributions / class_distributions.sum(axis=1, keepdims=True)

        # Limit number of classes shown for clarity
        classes_to_show = min(8, len(unique_classes))
        for i, cls in enumerate(unique_classes[:classes_to_show]):
            ax2.plot(rounds, class_proportions[:, i] * 100, 'o-', linewidth=1.5, markersize=4,
                    label=f'Class {cls}', color=self.colors[i])

        ax2.set_xlabel('Adaptive Round')
        ax2.set_ylabel('Percentage (%)')
        ax2.set_title('Class Proportion Evolution', fontweight='bold', fontsize=14)
        ax2.legend(loc='upper right', frameon=True, fancybox=True)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)

        plt.tight_layout()
        plt.savefig(self.subdirs['samples'] / 'class_distribution_evolution.png', dpi=200, bbox_inches='tight')
        plt.close()

        # Plot sample selection efficiency separately
        self._plot_sample_efficiency(rounds, training_history)

    def _plot_sample_efficiency(self, rounds, training_history):
        """Plot sample selection efficiency - OPTIMIZED"""
        fig, ax = plt.subplots(figsize=(12, 6))

        total_samples = [len(indices) for indices in training_history]
        new_samples_per_round = [len(training_history[0])] + \
                               [len(training_history[i]) - len(training_history[i-1])
                                for i in range(1, len(training_history))]

        width = 0.35
        x = np.arange(len(rounds))

        bars1 = ax.bar(x - width/2, total_samples, width, label='Cumulative Samples', alpha=0.7)
        bars2 = ax.bar(x + width/2, new_samples_per_round, width, label='New Samples per Round', alpha=0.7)

        ax.set_xlabel('Adaptive Round')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Sample Selection Efficiency', fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(rounds)
        ax.legend(loc='upper left', frameon=True, fancybox=True)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.subdirs['samples'] / 'sample_selection_efficiency.png', dpi=200, bbox_inches='tight')
        plt.close()

    def plot_training_sample_distributions(self, X_full, y_full, training_history, feature_names):
        """Plot feature distributions of selected training samples"""
        print("📊 Creating training sample distribution analysis...")

        if not training_history or len(training_history) < 3:
            return

        # Select key rounds to visualize
        key_rounds = [0, len(training_history)//2, -1]  # Start, middle, end
        round_names = ['Initial', 'Middle', 'Final']

        fig, axes = plt.subplots(3, min(5, X_full.shape[1]), figsize=(20, 12))
        if X_full.shape[1] == 1:
            axes = axes.reshape(-1, 1)

        for round_idx, (round_num, round_name) in enumerate(zip(key_rounds, round_names)):
            training_indices = training_history[round_num]
            X_train = X_full[training_indices]
            y_train = y_full[training_indices]

            # Plot distributions for first 5 features (or all if less than 5)
            n_features = min(5, X_full.shape[1])
            for feature_idx in range(n_features):
                ax = axes[round_idx, feature_idx]

                # Plot distribution for each class
                unique_classes = np.unique(y_train)
                for cls in unique_classes:
                    class_mask = y_train == cls
                    if np.any(class_mask):
                        feature_values = X_train[class_mask, feature_idx]
                        ax.hist(feature_values, bins=20, alpha=0.6,
                               label=f'Class {cls}', density=True)

                ax.set_xlabel(f'{feature_names[feature_idx]}')
                if feature_idx == 0:
                    ax.set_ylabel(f'{round_name}\nRound\nDensity')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

        plt.suptitle('Feature Distribution Evolution in Training Set', fontweight='bold', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.subdirs['distributions'] / 'feature_distribution_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_3d_networks(self, X_full, y_full, training_history, feature_names):
        """Create optimized 3D network visualizations of training samples"""
        print("🌐 Creating optimized 3D network visualizations...")

        if not training_history:
            return

        # Reduce dimensionality for visualization - use PCA for better performance
        if X_full.shape[1] > 3:
            pca = PCA(n_components=3, random_state=42)
            X_3d = pca.fit_transform(X_full)
            explained_var = pca.explained_variance_ratio_.sum()
        else:
            X_3d = X_full
            explained_var = 1.0

        # Limit to key rounds for performance
        total_rounds = len(training_history)
        if total_rounds > 5:
            # Show first, middle, and last rounds only
            key_rounds = [0, total_rounds//2, -1]
        else:
            key_rounds = list(range(total_rounds))

        for round_num in key_rounds:
            training_indices = training_history[round_num]
            self._create_optimized_3d_network(X_3d, y_full, training_indices,
                                            round_num, explained_var, feature_names)

    def _create_optimized_3d_network(self, X_3d, y_full, training_indices, round_num, explained_var, feature_names):
        """Create optimized single 3D network visualization"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Limit data for better performance
        max_points = 1000  # Maximum points to display
        if len(X_3d) > max_points:
            # Sample points for better performance
            sample_indices = np.random.choice(len(X_3d), max_points, replace=False)
            X_display = X_3d[sample_indices]
            y_display = y_full[sample_indices]
            training_mask_display = np.isin(sample_indices, training_indices)
        else:
            X_display = X_3d
            y_display = y_full
            training_mask_display = np.isin(range(len(X_3d)), training_indices)

        unique_classes = np.unique(y_display)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_classes)))

        # Plot non-training samples (background) with reduced alpha and size
        background_mask = ~training_mask_display
        for i, cls in enumerate(unique_classes):
            class_mask = (y_display == cls) & background_mask
            if np.any(class_mask):
                ax.scatter(X_display[class_mask, 0], X_display[class_mask, 1], X_display[class_mask, 2],
                          c=[colors[i]], alpha=0.05, s=5, marker='.')  # Reduced alpha and size

        # Plot training samples (foreground) - limit legend entries
        legend_handles = []
        legend_labels = []

        for i, cls in enumerate(unique_classes):
            class_mask = (y_display == cls) & training_mask_display
            if np.any(class_mask):
                scatter = ax.scatter(X_display[class_mask, 0], X_display[class_mask, 1], X_display[class_mask, 2],
                                   c=[colors[i]], alpha=0.8, s=30, label=f'Class {cls}',
                                   edgecolors='black', linewidth=0.5)
                if len(legend_handles) < 8:  # Limit legend entries
                    legend_handles.append(scatter)
                    legend_labels.append(f'Class {cls}')

        # Add network connections only for training samples (limited)
        if len(training_indices) <= 200:  # Only add connections for reasonable dataset sizes
            self._add_optimized_network_connections(ax, X_3d, y_full, training_indices, colors)

        ax.set_xlabel(f'PC1 ({explained_var*100:.1f}% variance)')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title(f'3D Training Network - Round {round_num + 1}\n'
                    f'Training Samples: {len(training_indices)}', fontweight='bold', fontsize=12)

        # Use limited legend
        if legend_handles:
            ax.legend(legend_handles, legend_labels, loc='upper left', bbox_to_anchor=(0, 1))

        plt.tight_layout()
        filename = f'3d_network_round_{round_num + 1}.png'
        plt.savefig(self.subdirs['networks'] / filename, dpi=150, bbox_inches='tight')  # Reduced DPI
        plt.close()

    def _add_optimized_network_connections(self, ax, X_3d, y_full, training_indices, colors):
        """Add optimized network connections between training samples"""
        training_mask = np.isin(range(len(X_3d)), training_indices)
        X_train = X_3d[training_mask]
        y_train = y_full[training_mask]

        unique_classes = np.unique(y_train)

        for i, cls in enumerate(unique_classes):
            class_mask = y_train == cls
            class_points = X_train[class_mask]

            # Only create connections for reasonable class sizes
            if len(class_points) < 2 or len(class_points) > 50:
                continue

            try:
                # Create minimum spanning tree with distance threshold
                from scipy.spatial import distance_matrix
                dist_matrix = distance_matrix(class_points, class_points)

                # Apply distance threshold to reduce connections
                max_distance = np.percentile(dist_matrix[dist_matrix > 0], 50)  # Median distance

                G = nx.Graph()
                for j in range(len(class_points)):
                    for k in range(j+1, len(class_points)):
                        if dist_matrix[j, k] <= max_distance:
                            G.add_edge(j, k, weight=dist_matrix[j, k])

                if G.number_of_edges() > 0:
                    mst = nx.minimum_spanning_tree(G)

                    # Plot MST edges
                    for edge in list(mst.edges())[:50]:  # Limit number of edges
                        point1 = class_points[edge[0]]
                        point2 = class_points[edge[1]]
                        ax.plot([point1[0], point2[0]],
                               [point1[1], point2[1]],
                               [point1[2], point2[2]],
                               color=colors[i], alpha=0.4, linewidth=0.8)  # Reduced alpha and linewidth

            except Exception as e:
                # Silently continue if MST fails
                continue

    def _create_single_3d_network(self, X_3d, y_full, training_indices, round_num, explained_var, feature_names):
        """Create a single 3D network visualization"""
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot all samples (background)
        unique_classes = np.unique(y_full)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_classes)))

        # Background points (non-training)
        background_mask = ~np.isin(range(len(X_3d)), training_indices)
        for i, cls in enumerate(unique_classes):
            class_mask = (y_full == cls) & background_mask
            if np.any(class_mask):
                ax.scatter(X_3d[class_mask, 0], X_3d[class_mask, 1], X_3d[class_mask, 2],
                          c=[colors[i]], alpha=0.1, s=10, label=f'_nolegend_')

        # Training samples (foreground)
        for i, cls in enumerate(unique_classes):
            class_mask = (y_full == cls) & np.isin(range(len(X_3d)), training_indices)
            if np.any(class_mask):
                ax.scatter(X_3d[class_mask, 0], X_3d[class_mask, 1], X_3d[class_mask, 2],
                          c=[colors[i]], alpha=0.8, s=50, label=f'Class {cls}',
                          edgecolors='black', linewidth=0.5)

        # Create network connections
        self._add_network_connections(ax, X_3d, y_full, training_indices, colors)

        ax.set_xlabel(f'PC1 ({explained_var*100:.1f}% variance)')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title(f'3D Training Network - Round {round_num + 1}\n'
                    f'Training Samples: {len(training_indices)}', fontweight='bold', fontsize=14)
        ax.legend()

        plt.tight_layout()
        filename = f'3d_network_round_{round_num + 1}.png'
        plt.savefig(self.subdirs['networks'] / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def _add_network_connections(self, ax, X_3d, y_full, training_indices, colors):
        """Add network connections between training samples"""
        training_mask = np.isin(range(len(X_3d)), training_indices)
        X_train = X_3d[training_mask]
        y_train = y_full[training_mask]

        unique_classes = np.unique(y_train)

        for i, cls in enumerate(unique_classes):
            class_mask = y_train == cls
            class_points = X_train[class_mask]

            if len(class_points) < 2:
                continue

            try:
                # Create minimum spanning tree
                from scipy.spatial import distance_matrix
                dist_matrix = distance_matrix(class_points, class_points)

                G = nx.Graph()
                for j in range(len(class_points)):
                    for k in range(j+1, len(class_points)):
                        G.add_edge(j, k, weight=dist_matrix[j, k])

                mst = nx.minimum_spanning_tree(G)

                # Plot MST edges
                for edge in mst.edges():
                    point1 = class_points[edge[0]]
                    point2 = class_points[edge[1]]
                    ax.plot([point1[0], point2[0]],
                           [point1[1], point2[1]],
                           [point1[2], point2[2]],
                           color=colors[i], alpha=0.6, linewidth=1.5)

            except Exception as e:
                print(f"⚠️ Could not create MST for class {cls}: {e}")

    def plot_feature_importance_analysis(self, adaptive_model, X_full, y_full, feature_names):
        """Analyze and plot feature importance"""
        print("🔧 Creating feature importance analysis...")

        try:
            # Use model's feature importance if available, otherwise use variance
            if hasattr(adaptive_model.model, 'feature_importances_'):
                importances = adaptive_model.model.feature_importances_
            else:
                # Use variance as proxy for importance
                importances = np.var(X_full, axis=0)

            # Sort features by importance
            sorted_idx = np.argsort(importances)[::-1]
            sorted_importances = importances[sorted_idx]
            sorted_names = [feature_names[i] for i in sorted_idx]

            # Plot feature importance
            fig, ax = plt.subplots(figsize=(12, 8))
            y_pos = np.arange(len(sorted_names))

            bars = ax.barh(y_pos, sorted_importances, color=self.colors[0], alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sorted_names)
            ax.set_xlabel('Importance Score')
            ax.set_title('Feature Importance Analysis', fontweight='bold', fontsize=14)
            ax.grid(True, alpha=0.3, axis='x')

            # Add value labels
            for bar, importance in zip(bars, sorted_importances):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2.,
                       f'{importance:.4f}', ha='left', va='center')

            plt.tight_layout()
            plt.savefig(self.subdirs['distributions'] / 'feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"⚠️ Feature importance analysis failed: {e}")

    def plot_class_separation_analysis(self, X_full, y_full, training_history):
        """Analyze class separation evolution"""
        print("🎯 Creating class separation analysis...")

        if not training_history:
            return

        # Calculate class separation metrics for each round
        separation_scores = []

        for training_indices in training_history:
            X_train = X_full[training_indices]
            y_train = y_full[training_indices]

            # Simple separation score: ratio of between-class to within-class variance
            unique_classes = np.unique(y_train)
            if len(unique_classes) < 2:
                separation_scores.append(0)
                continue

            overall_mean = np.mean(X_train, axis=0)
            between_var = 0
            within_var = 0

            for cls in unique_classes:
                class_mask = y_train == cls
                class_mean = np.mean(X_train[class_mask], axis=0)
                between_var += np.sum(class_mask) * np.sum((class_mean - overall_mean) ** 2)
                within_var += np.sum((X_train[class_mask] - class_mean) ** 2)

            if within_var > 0:
                separation_score = between_var / within_var
            else:
                separation_score = 0

            separation_scores.append(separation_score)

        # Plot separation evolution
        fig, ax = plt.subplots(figsize=(12, 6))
        rounds = list(range(1, len(separation_scores) + 1))

        ax.plot(rounds, separation_scores, 'o-', linewidth=2, markersize=8, color=self.colors[0])
        ax.set_xlabel('Adaptive Round')
        ax.set_ylabel('Separation Score')
        ax.set_title('Class Separation Evolution in Training Set', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3)

        # Add trend line
        if len(separation_scores) > 1:
            z = np.polyfit(rounds, separation_scores, 1)
            p = np.poly1d(z)
            ax.plot(rounds, p(rounds), "--", color='red', alpha=0.7,
                   label=f'Trend: {z[0]:.3f}x + {z[1]:.3f}')
            ax.legend()

        plt.tight_layout()
        plt.savefig(self.subdirs['comparisons'] / 'class_separation_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_confidence_evolution(self, adaptive_model, X_full, y_full, training_history):
        """Plot confidence evolution across rounds"""
        print("🎲 Creating confidence evolution analysis...")

        if not training_history or not hasattr(adaptive_model.model, 'predict_proba'):
            return

        confidence_evolution = []

        for training_indices in training_history:
            # Train temporary model (simplified - in practice you'd use the actual trained model)
            X_train = X_full[training_indices]
            y_train = y_full[training_indices]

            try:
                # Get prediction probabilities
                probas = adaptive_model.model.predict_proba(X_full)
                max_probas = np.max(probas, axis=1)
                avg_confidence = np.mean(max_probas)
                confidence_evolution.append(avg_confidence)
            except:
                confidence_evolution.append(0.5)  # Default value

        # Plot confidence evolution
        fig, ax = plt.subplots(figsize=(12, 6))
        rounds = list(range(1, len(confidence_evolution) + 1))

        ax.plot(rounds, confidence_evolution, 'o-', linewidth=2, markersize=8, color=self.colors[1])
        ax.set_xlabel('Adaptive Round')
        ax.set_ylabel('Average Prediction Confidence')
        ax.set_title('Prediction Confidence Evolution', fontweight='bold', fontsize=14)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.subdirs['performance'] / 'confidence_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_interactive_dashboard(self, round_stats, training_history, X_full, y_full, feature_names):
        """Create interactive dashboard with all visualizations"""
        print("📊 Creating interactive dashboard...")

        # Create comprehensive dashboard HTML
        dashboard_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Adaptive DBNN Dashboard - {self.dataset_name}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                         color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
                .plot-container {{ margin: 20px 0; border: 1px solid #ddd; border-radius: 5px; padding: 15px; }}
                .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; margin: 20px 0; }}
                .stat-card {{ background: #f8f9fa; padding: 15px; border-radius: 5px; text-align: center; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎯 Adaptive DBNN Analysis Dashboard</h1>
                <h2>Dataset: {self.dataset_name}</h2>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>

            <div class="stats">
                <div class="stat-card">
                    <h3>Total Rounds</h3>
                    <p style="font-size: 24px; font-weight: bold; color: #667eea;">{len(round_stats) if round_stats else 0}</p>
                </div>
                <div class="stat-card">
                    <h3>Final Training Size</h3>
                    <p style="font-size: 24px; font-weight: bold; color: #28a745;">{training_history[-1] if training_history else 0}</p>
                </div>
                <div class="stat-card">
                    <h3>Best Accuracy</h3>
                    <p style="font-size: 24px; font-weight: bold; color: #dc3545;">{max([s['test_accuracy'] for s in round_stats])*100:.1f}%</p>
                </div>
                <div class="stat-card">
                    <h3>Features</h3>
                    <p style="font-size: 24px; font-weight: bold; color: #ffc107;">{len(feature_names)}</p>
                </div>
            </div>

            <div class="plot-container">
                <h3>📈 Performance Evolution</h3>
                <div id="performance-plot"></div>
            </div>

            <div class="plot-container">
                <h3>🔍 Sample Selection Analysis</h3>
                <div id="sample-plot"></div>
            </div>

            <script>
                // Performance data
                const rounds = {[s['round'] for s in round_stats] if round_stats else []};
                const trainAcc = {[s['train_accuracy']*100 for s in round_stats] if round_stats else []};
                const testAcc = {[s['test_accuracy']*100 for s in round_stats] if round_stats else []};

                // Create performance plot
                Plotly.newPlot('performance-plot', [
                    {{x: rounds, y: trainAcc, type: 'scatter', name: 'Training Accuracy', line: {{color: '#1f77b4'}}}},
                    {{x: rounds, y: testAcc, type: 'scatter', name: 'Test Accuracy', line: {{color: '#ff7f0e'}}}}
                ], {{title: 'Accuracy Evolution Across Rounds'}});

                // Sample selection data
                const trainingSizes = {[len(indices) for indices in training_history] if training_history else []};

                Plotly.newPlot('sample-plot', [
                    {{x: rounds, y: trainingSizes, type: 'scatter', name: 'Training Size', line: {{color: '#2ca02c'}}}}
                ], {{title: 'Training Set Growth'}});
            </script>
        </body>
        </html>
        """

        with open(self.subdirs['interactive'] / 'dashboard.html', 'w') as f:
            f.write(dashboard_html)

    def plot_final_model_analysis(self, adaptive_model, X_full, y_full, feature_names):
        """Create final model analysis plots"""
        print("🏆 Creating final model analysis...")

        try:
            # Get predictions
            y_pred = adaptive_model.model.predict(X_full)

            # Confusion Matrix
            fig, ax = plt.subplots(figsize=(10, 8))
            cm = confusion_matrix(y_full, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Final Model Confusion Matrix', fontweight='bold', fontsize=14)
            plt.tight_layout()
            plt.savefig(self.subdirs['performance'] / 'final_confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()

            # Classification Report
            report = classification_report(y_full, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv(self.subdirs['performance'] / 'classification_report.csv')

        except Exception as e:
            print(f"⚠️ Final model analysis failed: {e}")


class AdaptiveVisualizer3D:
    """3D Visualization system for adaptive learning training samples"""

    def __init__(self, output_dir='adaptive_3d_visualizations'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def create_3d_training_network(self, X_full, y_full, training_indices, feature_names=None,
                                 round_num=None, method='pca'):
        """Create 3D visualization of training samples forming class networks"""

        print("🎨 Creating 3D training sample network visualization...")

        # Reduce to 3D for visualization
        if method == 'pca':
            reducer = PCA(n_components=3, random_state=42)
            X_3d = reducer.fit_transform(X_full)
            explained_var = sum(reducer.explained_variance_ratio_)
            print(f"📊 PCA explained variance: {explained_var:.3f}")
        else:  # tsne
            reducer = TSNE(n_components=3, random_state=42, perplexity=30)
            X_3d = reducer.fit_transform(X_full)
            explained_var = 1.0

        # Separate training and non-training samples
        train_mask = np.zeros(len(X_full), dtype=bool)
        train_mask[training_indices] = True

        X_train_3d = X_3d[train_mask]
        y_train = y_full[train_mask]
        X_other_3d = X_3d[~train_mask]
        y_other = y_full[~train_mask]

        # Create the plot
        fig = plt.figure(figsize=(15, 10))

        # 3D scatter plot
        ax = fig.add_subplot(111, projection='3d')

        # Plot all samples (transparent)
        unique_classes = np.unique(y_full)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_classes)))

        # Plot non-training samples (faint)
        for i, cls in enumerate(unique_classes):
            mask = y_other == cls
            if np.any(mask):
                ax.scatter(X_other_3d[mask, 0], X_other_3d[mask, 1], X_other_3d[mask, 2],
                          c=[colors[i]], alpha=0.1, s=10, label=f'Class {cls} (other)')

        # Plot training samples (bright)
        for i, cls in enumerate(unique_classes):
            mask = y_train == cls
            if np.any(mask):
                ax.scatter(X_train_3d[mask, 0], X_train_3d[mask, 1], X_train_3d[mask, 2],
                          c=[colors[i]], alpha=0.8, s=50, label=f'Class {cls} (training)',
                          edgecolors='black', linewidth=0.5)

        # Create network connections within each class
        self._add_class_networks(ax, X_train_3d, y_train, colors)

        # Customize the plot
        ax.set_xlabel(f'Component 1 ({explained_var*100:.1f}% variance)')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')

        title = '3D Training Sample Network'
        if round_num is not None:
            title += f' - Round {round_num}'
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Save the plot
        filename = f'training_network_round_{round_num}.png' if round_num else 'training_network_final.png'
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{filename}', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ 3D network visualization saved: {filename}")

        # Also create interactive Plotly version
        self._create_interactive_3d_plot(X_3d, y_full, train_mask, training_indices, round_num)

    def _add_class_networks(self, ax, X_3d, y_train, colors):
        """Add network connections between training samples of the same class"""
        unique_classes = np.unique(y_train)

        for i, cls in enumerate(unique_classes):
            class_mask = y_train == cls
            class_points = X_3d[class_mask]

            if len(class_points) < 2:
                continue

            # Create a minimum spanning tree for the class
            try:
                # Calculate distance matrix
                from scipy.spatial import distance_matrix
                dist_matrix = distance_matrix(class_points, class_points)

                # Create graph and minimum spanning tree
                G = nx.Graph()
                for j in range(len(class_points)):
                    for k in range(j+1, len(class_points)):
                        G.add_edge(j, k, weight=dist_matrix[j, k])

                mst = nx.minimum_spanning_tree(G)

                # Plot MST edges
                for edge in mst.edges():
                    point1 = class_points[edge[0]]
                    point2 = class_points[edge[1]]
                    ax.plot([point1[0], point2[0]],
                           [point1[1], point2[1]],
                           [point1[2], point2[2]],
                           color=colors[i], alpha=0.6, linewidth=1.5)

            except Exception as e:
                print(f"⚠️ Could not create MST for class {cls}: {e}")

    def _create_interactive_3d_plot(self, X_3d, y_full, train_mask, training_indices, round_num):
        """Create interactive 3D plot using Plotly"""

        # Create DataFrame for Plotly
        import pandas as pd
        df = pd.DataFrame({
            'x': X_3d[:, 0],
            'y': X_3d[:, 1],
            'z': X_3d[:, 2],
            'class': y_full,
            'type': ['Training' if i in training_indices else 'Other' for i in range(len(X_3d))],
            'index': range(len(X_3d))
        })

        # Create interactive scatter plot
        fig = px.scatter_3d(df, x='x', y='y', z='z',
                           color='class',
                           symbol='type',
                           hover_data=['index'],
                           title=f'Interactive 3D Training Network - Round {round_num}' if round_num else 'Interactive 3D Training Network - Final',
                           opacity=0.7)

        # Update marker sizes
        fig.update_traces(marker=dict(size=5 if df['type'] == 'Other' else 8),
                         selector=dict(mode='markers'))

        # Save interactive plot
        filename = f'interactive_network_round_{round_num}.html' if round_num else 'interactive_network_final.html'
        fig.write_html(f'{self.output_dir}/{filename}')

        print(f"✅ Interactive 3D visualization saved: {filename}")

    def create_adaptive_learning_animation(self, X_full, y_full, training_history):
        """Create animation showing evolution of training samples"""
        print("🎬 Creating adaptive learning animation...")

        # Reduce to 3D once for consistency
        reducer = PCA(n_components=3, random_state=42)
        X_3d = reducer.fit_transform(X_full)

        frames = []

        for round_num, training_indices in enumerate(training_history):
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Plot all samples
            unique_classes = np.unique(y_full)
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_classes)))

            # Plot non-training samples
            other_mask = ~np.isin(range(len(X_full)), training_indices)
            for i, cls in enumerate(unique_classes):
                class_mask = y_full == cls
                mask = class_mask & other_mask
                if np.any(mask):
                    ax.scatter(X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2],
                              c=[colors[i]], alpha=0.1, s=5)

            # Plot training samples
            for i, cls in enumerate(unique_classes):
                class_mask = y_full == cls
                mask = class_mask & np.isin(range(len(X_full)), training_indices)
                if np.any(mask):
                    ax.scatter(X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2],
                              c=[colors[i]], alpha=0.8, s=30, label=f'Class {cls}',
                              edgecolors='black', linewidth=0.5)

            ax.set_title(f'Adaptive Learning - Round {round_num + 1}\nTraining Samples: {len(training_indices)}',
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_zlabel('Component 3')

            frames.append(fig)
            plt.close()

        # Create animation (you'll need to install imageio: pip install imageio)
        try:
            import imageio
            images = []
            for fig in frames:
                fig.canvas.draw()
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                images.append(image)

            imageio.mimsave(f'{self.output_dir}/adaptive_learning_evolution.gif',
                           images, fps=2, loop=0)
            print("✅ Adaptive learning animation saved: adaptive_learning_evolution.gif")

        except ImportError:
            print("⚠️ imageio not installed, skipping animation creation")

class AdaptiveDBNNGUI:
    """
    Enhanced GUI for Adaptive DBNN with feature selection and hyperparameter configuration.
    Provides an interactive interface for the adaptive learning system.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Adaptive DBNN with Feature Selection")
        self.root.geometry("1400x900")

        self.training_process = None
        self.training_queue = queue.Queue()
        self.training_active = False

        self.adaptive_model = None
        self.model_trained = False
        self.data_loaded = False
        self.current_data_file = None
        self.original_data = None

        # Feature selection state
        self.feature_vars = {}
        self.target_var = tk.StringVar()

        # Configuration management
        self.config_vars = {}

        # Data file variable
        self.data_file_var = tk.StringVar()

        # Adaptive learning parameters
        self.max_rounds_var = tk.StringVar(value="20")
        self.max_samples_var = tk.StringVar(value="25")
        self.initial_samples_var = tk.StringVar(value="5")

        # DBNN core parameters
        self.resolution_var = tk.StringVar(value="100")
        self.gain_var = tk.StringVar(value="2.0")
        self.margin_var = tk.StringVar(value="0.2")
        self.patience_var = tk.StringVar(value="10")

        # Adaptive learning options - ADD VISUALIZATION TOGGLE
        self.enable_acid_var = tk.BooleanVar(value=True)
        self.enable_kl_var = tk.BooleanVar(value=False)
        self.disable_sample_limit_var = tk.BooleanVar(value=False)
        self.enable_visualization_var = tk.BooleanVar(value=True)  # NEW: Visualization toggle

        self.setup_gui()
        self.setup_common_controls()

    def run_adaptive_learning_async(self):
        """Start training in separate thread"""
        if self.training_active:
            messagebox.showwarning("Warning", "Training already in progress")
            return

        self.training_active = True
        self.training_process = threading.Thread(
            target=self._training_worker,
            args=(self.training_queue, self.get_training_config()),
            daemon=True  # Make it a daemon thread
        )
        self.training_process.start()
        self.log_output("🚀 Training started in background thread...")
        self.log_output("💡 You can continue using the GUI freely")

        # Start monitoring progress
        self.root.after(100, self._check_training_progress)

    def _training_worker(self, queue, config):
        """Worker thread that runs training"""
        try:
            # Initialize model in thread
            adaptive_model = AdaptiveDBNN(config['dataset_name'], config)

            # Setup progress reporting - use thread-safe callbacks
            adaptive_model.set_progress_callback(lambda msg: queue.put(('progress', msg)))

            # Run training
            results = adaptive_model.adaptive_learn(
                feature_columns=config['feature_columns']
            )

            queue.put(('complete', results))

        except Exception as e:
            queue.put(('error', str(e)))

    def get_training_config(self):
        """Get training configuration from GUI"""
        if not self.data_loaded or self.adaptive_model is None:
            raise ValueError("Data not loaded or model not initialized")

        return {
            'dataset_name': self.dataset_name,
            'target_column': self.target_var.get(),
            'feature_columns': [col for col, var in self.feature_vars.items()
                              if var.get() and col != self.target_var.get()],
            'resol': int(self.config_vars["dbnn_resolution"].get()),
            'gain': float(self.config_vars["dbnn_gain"].get()),
            'margin': float(self.config_vars["dbnn_margin"].get()),
            'patience': int(self.config_vars["dbnn_patience"].get()),
            'max_epochs': int(self.config_vars["dbnn_max_epochs"].get()),
            'min_improvement': float(self.config_vars["dbnn_min_improvement"].get()),
            'adaptive_learning': {
                'enable_adaptive': True,
                'initial_samples_per_class': int(self.initial_samples_var.get()),
                'max_adaptive_rounds': int(self.max_rounds_var.get()),
                'max_margin_samples_per_class': int(self.max_samples_var.get()),
                'enable_acid_test': self.enable_acid_var.get(),
                'enable_kl_divergence': self.enable_kl_var.get(),
                'disable_sample_limit': self.disable_sample_limit_var.get(),
                'enable_visualization': self.enable_visualization_var.get(),
            }
        }

    def _training_completed(self, results):
        """Handle training completion"""
        self.training_active = False
        self.model_trained = True

        # Update results display
        self.display_results(results)

        self.log_output("✅ Training completed successfully!")
        self.status_var.set("Training completed")

    def _training_failed(self, error_msg):
        """Handle training failure"""
        self.training_active = False
        self.log_output(f"❌ Training failed: {error_msg}")
        self.status_var.set("Training failed")
        messagebox.showerror("Training Error", f"Training failed:\n{error_msg}")

    def log_output(self, message: str):
        """Thread-safe output logging"""
        def update_log():
            self.output_text.insert(tk.END, f"{message}\n")
            self.output_text.see(tk.END)
            self.status_var.set(message)

        # Use thread-safe GUI update
        self.root.after(0, update_log)

    def safe_exit(self):
        """Safely exit the application with confirmation"""
        if self.training_active:
            if not messagebox.askyesno("Training in Progress",
                                      "Training is still in progress. Are you sure you want to exit?"):
                return

        if messagebox.askyesno("Exit", "Are you sure you want to exit?"):
            try:
                # Stop any active training
                self.training_active = False

                # Clean up threads
                if self.training_process and self.training_process.is_alive():
                    self.training_process.join(timeout=2.0)  # Wait max 2 seconds

                # Clean up any temporary files or resources
                if hasattr(self, 'adaptive_model'):
                    del self.adaptive_model

                self.root.quit()
                self.root.destroy()

            except Exception as e:
                # Force exit even if cleanup fails
                import traceback
                traceback.print_exc()
                self.root.quit()
                self.root.destroy()


    def _check_training_progress(self):
        """Enhanced progress monitoring with timeouts"""
        try:
            # Use non-blocking get with timeout
            try:
                msg_type, data = self.training_queue.get_nowait()

                if msg_type == 'progress':
                    self.log_output(f"📊 {data}")
                    self.status_var.set(data)

                elif msg_type == 'round_update':
                    round_num, accuracy, samples = data
                    self.log_output(f"🔄 Round {round_num}: Accuracy={accuracy:.4f}, Samples={samples}")

                elif msg_type == 'training_stats':
                    epoch, accuracy = data
                    progress = f"Epoch {epoch}: {accuracy:.2f}%"
                    self.status_var.set(progress)

                elif msg_type == 'complete':
                    self._training_completed(data)
                    return

                elif msg_type == 'error':
                    self._training_failed(data)
                    return

            except queue.Empty:
                # No message in queue, continue checking
                pass

        except Exception as e:
            self.log_output(f"❌ Error in progress monitoring: {e}")

        # Continue monitoring if training is still active
        if self.training_active:
            self.root.after(300, self._check_training_progress)  # Reduced frequency


    def setup_common_controls(self):
        """Setup common window controls including exit button"""
        # Create a common control frame at the bottom
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=5)

        ttk.Button(control_frame, text="🔄 Refresh GUI",
                   command=self.refresh_gui_values).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="💾 Save All Settings",
                   command=self.save_all_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="❌ Exit",
                   command=self.safe_exit, width=10).pack(side=tk.RIGHT, padx=5)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(control_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)

    def refresh_gui_values(self):
        """Refresh all GUI values to ensure they are current"""
        try:
            # Force update of all variables
            self.root.update()
            self.log_output("✅ GUI values refreshed and effective")
        except Exception as e:
            self.log_output(f"❌ Error refreshing GUI: {e}")

    def save_all_settings(self):
        """Save all current settings to configuration"""
        try:
            if self.current_data_file:
                self.save_configuration_for_file(self.current_data_file)
                self.apply_hyperparameters()
                self.log_output("✅ All settings saved and applied")
            else:
                messagebox.showinfo("Info", "Please load a data file first.")
        except Exception as e:
            self.log_output(f"❌ Error saving settings: {e}")

    def safe_exit(self):
        """Safely exit the application with confirmation"""
        if messagebox.askyesno("Exit", "Are you sure you want to exit?"):
            try:
                # Clean up any temporary files or resources
                if hasattr(self, 'adaptive_model'):
                    del self.adaptive_model
                self.root.quit()
                self.root.destroy()
            except Exception as e:
                # Force exit even if cleanup fails
                self.root.quit()
                self.root.destroy()

    def setup_gui(self):
        """Setup the main GUI interface with tabs and horizontal navigation."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create horizontal navigation frame for tab buttons
        nav_frame = ttk.Frame(main_frame)
        nav_frame.pack(fill=tk.X, pady=5)

        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)

        # Data Management Tab
        self.data_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.data_tab, text="📊 Data Management")

        # Hyperparameters Tab
        self.hyperparams_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.hyperparams_tab, text="⚙️ Hyperparameters")

        # Training Tab
        self.training_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.training_tab, text="🚀 Training & Evaluation")

        # Create navigation buttons for tabs
        ttk.Button(nav_frame, text="📊 Data",
                   command=lambda: self.notebook.select(0)).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="⚙️ Parameters",
                   command=lambda: self.notebook.select(1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="🚀 Training",
                   command=lambda: self.notebook.select(2)).pack(side=tk.LEFT, padx=2)

        # Setup each tab
        self.setup_data_tab()
        self.setup_hyperparameters_tab()
        self.setup_training_tab()

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def setup_data_tab(self):
        """Setup data management tab with feature selection."""
        # Dataset selection frame
        dataset_frame = ttk.LabelFrame(self.data_tab, text="Dataset Selection", padding="10")
        dataset_frame.pack(fill=tk.X, pady=5)

        ttk.Label(dataset_frame, text="Data File:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.data_file_entry = ttk.Entry(dataset_frame, textvariable=self.data_file_var, width=50)
        self.data_file_entry.grid(row=0, column=1, padx=5, sticky=tk.EW)

        ttk.Button(dataset_frame, text="Browse", command=self.browse_data_file).grid(row=0, column=2, padx=5)
        ttk.Button(dataset_frame, text="Load Data", command=self.load_data_file).grid(row=0, column=3, padx=5)

        # Feature selection frame
        feature_frame = ttk.LabelFrame(self.data_tab, text="Feature Selection", padding="10")
        feature_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Target selection
        ttk.Label(feature_frame, text="Target Column:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.target_combo = ttk.Combobox(feature_frame, textvariable=self.target_var, width=20, state="readonly")
        self.target_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        self.target_combo.bind('<<ComboboxSelected>>', self.on_target_selected)

        # Feature selection area with scrollbar
        ttk.Label(feature_frame, text="Feature Columns:").grid(row=1, column=0, sticky=tk.NW, padx=5, pady=5)

        # Create frame for feature list with scrollbar
        feature_list_frame = ttk.Frame(feature_frame)
        feature_list_frame.grid(row=1, column=1, columnspan=3, sticky=tk.NSEW, padx=5, pady=5)

        # Create canvas and scrollbar for feature list
        self.feature_canvas = tk.Canvas(feature_list_frame, height=200)
        feature_scrollbar = ttk.Scrollbar(feature_list_frame, orient="vertical", command=self.feature_canvas.yview)
        self.feature_scroll_frame = ttk.Frame(self.feature_canvas)

        self.feature_scroll_frame.bind(
            "<Configure>",
            lambda e: self.feature_canvas.configure(scrollregion=self.feature_canvas.bbox("all"))
        )

        self.feature_canvas.create_window((0, 0), window=self.feature_scroll_frame, anchor="nw")
        self.feature_canvas.configure(yscrollcommand=feature_scrollbar.set)

        self.feature_canvas.pack(side="left", fill="both", expand=True)
        feature_scrollbar.pack(side="right", fill="y")

        # Feature selection buttons
        button_frame = ttk.Frame(feature_frame)
        button_frame.grid(row=2, column=1, columnspan=3, sticky=tk.W, padx=5, pady=5)

        ttk.Button(button_frame, text="Select All Features",
                  command=self.select_all_features).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Deselect All Features",
                  command=self.deselect_all_features).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Select Only Numeric",
                  command=self.select_numeric_features).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Apply Selection",
                  command=self.apply_feature_selection).pack(side=tk.LEFT, padx=2)

        # Data info display
        self.data_info_text = scrolledtext.ScrolledText(feature_frame, height=8, width=80)
        self.data_info_text.grid(row=3, column=0, columnspan=4, sticky=tk.NSEW, pady=5)
        self.data_info_text.config(state=tk.DISABLED)

        # Configure grid weights
        self.data_tab.columnconfigure(0, weight=1)
        self.data_tab.rowconfigure(0, weight=1)
        feature_frame.columnconfigure(1, weight=1)
        feature_frame.rowconfigure(1, weight=1)
        feature_list_frame.columnconfigure(0, weight=1)
        feature_list_frame.rowconfigure(0, weight=1)

    def setup_hyperparameters_tab(self):
        """Setup hyperparameters configuration tab."""
        # Create main frame with scrollbar
        main_frame = ttk.Frame(self.hyperparams_tab)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create canvas and scrollbar
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # DBNN Core Parameters Frame
        core_frame = ttk.LabelFrame(scrollable_frame, text="DBNN Core Parameters", padding="10")
        core_frame.pack(fill=tk.X, pady=5, padx=10)

        # Core parameters
        core_params = [
            ("resolution", "Resolution:", "100", "Number of bins for feature discretization"),
            ("gain", "Gain:", "2.0", "Weight update intensity"),
            ("margin", "Margin:", "0.2", "Classification tolerance"),
            ("patience", "Patience:", "10", "Early stopping rounds"),
            ("max_epochs", "Max Epochs:", "100", "Maximum training epochs"),
            ("min_improvement", "Min Improvement:", "0.1", "Minimum improvement threshold")
        ]

        for i, (key, label, default, help_text) in enumerate(core_params):
            ttk.Label(core_frame, text=label).grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
            var = tk.StringVar(value=default)
            entry = ttk.Entry(core_frame, textvariable=var, width=12)
            entry.grid(row=i, column=1, padx=5, pady=2)
            ttk.Label(core_frame, text=help_text, foreground="gray").grid(row=i, column=2, sticky=tk.W, padx=5, pady=2)
            self.config_vars[f"dbnn_{key}"] = var

        # Adaptive Learning Parameters Frame
        adaptive_frame = ttk.LabelFrame(scrollable_frame, text="Adaptive Learning Parameters", padding="10")
        adaptive_frame.pack(fill=tk.X, pady=5, padx=10)

        # Adaptive parameters
        ttk.Label(adaptive_frame, text="Max Adaptive Rounds:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        max_rounds_entry = ttk.Entry(adaptive_frame, textvariable=self.max_rounds_var, width=12)
        max_rounds_entry.grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(adaptive_frame, text="Maximum adaptive learning rounds", foreground="gray").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)

        ttk.Label(adaptive_frame, text="Max Samples/Round:").grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)
        max_samples_entry = ttk.Entry(adaptive_frame, textvariable=self.max_samples_var, width=12)
        max_samples_entry.grid(row=0, column=4, padx=5, pady=2)
        ttk.Label(adaptive_frame, text="Maximum samples to add per round", foreground="gray").grid(row=0, column=5, sticky=tk.W, padx=5, pady=2)

        ttk.Label(adaptive_frame, text="Initial Samples/Class:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        initial_samples_entry = ttk.Entry(adaptive_frame, textvariable=self.initial_samples_var, width=12)
        initial_samples_entry.grid(row=1, column=1, padx=5, pady=2)
        ttk.Label(adaptive_frame, text="Initial samples per class for training", foreground="gray").grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)

        # Additional adaptive parameters
        adaptive_params = [
            ("margin_tolerance", "Margin Tolerance:", "0.15", "Tolerance for margin-based selection"),
            ("kl_threshold", "KL Threshold:", "0.1", "Threshold for KL divergence"),
            ("training_convergence_epochs", "Convergence Epochs:", "50", "Epochs for training convergence"),
            ("min_training_accuracy", "Min Training Accuracy:", "0.95", "Minimum training accuracy"),
            ("adaptive_margin_relaxation", "Margin Relaxation:", "0.1", "Margin relaxation factor")
        ]

        for i, (key, label, default, help_text) in enumerate(adaptive_params):
            row = i + 2
            ttk.Label(adaptive_frame, text=label).grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
            var = tk.StringVar(value=default)
            entry = ttk.Entry(adaptive_frame, textvariable=var, width=12)
            entry.grid(row=row, column=1, padx=5, pady=2)
            ttk.Label(adaptive_frame, text=help_text, foreground="gray").grid(row=row, column=2, sticky=tk.W, padx=5, pady=2)
            self.config_vars[f"adaptive_{key}"] = var

        # Advanced Adaptive Options
        ttk.Label(adaptive_frame, text="Advanced Options:").grid(row=6, column=0, sticky=tk.W, padx=5, pady=2)

        ttk.Checkbutton(adaptive_frame, text="Enable Acid Test", variable=self.enable_acid_var).grid(row=6, column=1, sticky=tk.W, padx=5)
        ttk.Checkbutton(adaptive_frame, text="Enable KL Divergence", variable=self.enable_kl_var).grid(row=6, column=2, sticky=tk.W, padx=5)
        ttk.Checkbutton(adaptive_frame, text="Disable Sample Limit", variable=self.disable_sample_limit_var).grid(row=6, column=3, sticky=tk.W, padx=5)

        # Control buttons frame
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill=tk.X, pady=10, padx=10)

        ttk.Button(button_frame, text="Load Default Parameters",
                  command=self.load_default_parameters).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Current Parameters",
                  command=self.save_current_parameters).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Apply Parameters",
                  command=self.apply_hyperparameters).pack(side=tk.RIGHT, padx=5)

        # Advanced Adaptive Options - ADD VISUALIZATION TOGGLE
        ttk.Label(adaptive_frame, text="Advanced Options:").grid(row=6, column=0, sticky=tk.W, padx=5, pady=2)

        ttk.Checkbutton(adaptive_frame, text="Enable Acid Test", variable=self.enable_acid_var).grid(row=6, column=1, sticky=tk.W, padx=5)
        ttk.Checkbutton(adaptive_frame, text="Enable KL Divergence", variable=self.enable_kl_var).grid(row=6, column=2, sticky=tk.W, padx=5)
        ttk.Checkbutton(adaptive_frame, text="Disable Sample Limit", variable=self.disable_sample_limit_var).grid(row=6, column=3, sticky=tk.W, padx=5)
        ttk.Checkbutton(adaptive_frame, text="Enable Visualization", variable=self.enable_visualization_var).grid(row=7, column=1, sticky=tk.W, padx=5)  # NEW

    def generate_final_visualizations(self):
        """Generate comprehensive final visualizations on-demand"""
        if not self.model_trained or self.adaptive_model is None:
            messagebox.showwarning("Warning", "No trained model available. Please run adaptive learning first.")
            return

        try:
            self.log_output("🏆 Generating comprehensive final visualizations...")

            # Show progress
            self.status_var.set("Generating final visualizations...")
            self.root.update()

            # Generate comprehensive visualizations
            if hasattr(self.adaptive_model, 'comprehensive_visualizer'):
                self.adaptive_model.comprehensive_visualizer.create_comprehensive_visualizations(
                    self.adaptive_model,
                    self.adaptive_model.X_full,
                    self.adaptive_model.y_full,
                    self.adaptive_model.training_history,
                    self.adaptive_model.round_stats,
                    self.adaptive_model.feature_columns
                )
                self.log_output("✅ Comprehensive visualizations generated successfully!")
            else:
                self.log_output("⚠️ Comprehensive visualizer not available")

            # Generate advanced 3D visualizations
            if hasattr(self.adaptive_model, 'advanced_visualizer'):
                self.adaptive_model.advanced_visualizer.create_advanced_3d_dashboard(
                    self.adaptive_model.X_full,
                    self.adaptive_model.y_full,
                    self.adaptive_model.training_history,
                    self.adaptive_model.feature_columns,
                    round_num=None  # Final visualization
                )
                self.log_output("✅ Advanced 3D dashboard generated!")
            else:
                self.log_output("⚠️ Advanced 3D visualizer not available")

            # Generate final model analysis
            if hasattr(self.adaptive_model, 'comprehensive_visualizer'):
                self.adaptive_model.comprehensive_visualizer.plot_final_model_analysis(
                    self.adaptive_model,
                    self.adaptive_model.X_full,
                    self.adaptive_model.y_full,
                    self.adaptive_model.feature_columns
                )
                self.log_output("✅ Final model analysis generated!")

            # Open the visualization location
            self.open_visualization_location()

            self.status_var.set("Final visualizations completed!")
            self.log_output("🎉 All final visualizations completed and folder opened!")

        except Exception as e:
            self.log_output(f"❌ Error generating final visualizations: {e}")
            self.status_var.set("Visualization error")

    def setup_training_tab(self):
        """Setup training and evaluation tab with enhanced visualization controls"""
        # Control frame
        control_frame = ttk.LabelFrame(self.training_tab, text="Model Control", padding="10")
        control_frame.pack(fill=tk.X, pady=5)

        ttk.Button(control_frame, text="Initialize Model",
                  command=self.initialize_model, width=15).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Run Adaptive Learning",
                  command=self.run_adaptive_learning, width=18).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Evaluate Model",
                  command=self.evaluate_model, width=15).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Save Model",
                  command=self.save_model, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Load Model",
                  command=self.load_model, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Predict",
              command=self.predict_with_loaded_model, width=12).pack(side=tk.LEFT, padx=2)

        # Enhanced Visualization frame
        viz_frame = ttk.LabelFrame(self.training_tab, text="Advanced Visualization", padding="10")
        viz_frame.pack(fill=tk.X, pady=5)

        # Visualization controls in a grid
        viz_control_frame = ttk.Frame(viz_frame)
        viz_control_frame.pack(fill=tk.X)

        # Row 1: Basic and Advanced Visualizations
        ttk.Button(viz_control_frame, text="📊 Basic Visualizations",
                  command=self.show_visualizations, width=18).grid(row=0, column=0, padx=2, pady=2)
        ttk.Button(viz_control_frame, text="🔬 Advanced Analysis",
                  command=self.show_advanced_analysis, width=18).grid(row=0, column=1, padx=2, pady=2)
        ttk.Button(viz_control_frame, text="🌐 Interactive 3D",
                  command=self.show_interactive_3d, width=18).grid(row=0, column=2, padx=2, pady=2)

        # Row 2: Final Visualizations and Folder Access
        ttk.Button(viz_control_frame, text="🏆 Final Comprehensive Viz",
                  command=self.generate_final_visualizations, width=18).grid(row=1, column=0, padx=2, pady=2)
        ttk.Button(viz_control_frame, text="📁 Open Viz Location",
                  command=self.open_visualization_location, width=18).grid(row=1, column=1, padx=2, pady=2)
        ttk.Button(viz_control_frame, text="🎬 Show Animations",
                  command=self.show_animations, width=18).grid(row=1, column=2, padx=2, pady=2)

        # Create a notebook for output and results
        output_notebook = ttk.Notebook(self.training_tab)
        output_notebook.pack(fill=tk.BOTH, expand=True, pady=5)

        # Output frame
        output_frame = ttk.Frame(output_notebook)
        output_notebook.add(output_frame, text="📝 Output Log")

        self.output_text = scrolledtext.ScrolledText(output_frame, height=15, width=100)
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Results frame
        results_frame = ttk.Frame(output_notebook)
        output_notebook.add(results_frame, text="📊 Results")

        self.results_text = scrolledtext.ScrolledText(results_frame, height=15, width=100)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.results_text.config(state=tk.DISABLED)

    def load_adaptive_model_for_prediction(self, model_path):
        """Load adaptive_dbnn model using the same structure as adaptive_dbnn"""
        try:
            import gzip
            import pickle

            self.log_output(f"📥 Loading adaptive model: {model_path}")

            # Load the model data using the same method as adaptive_dbnn
            with gzip.open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            # Extract the adaptive model components
            self.adaptive_model = AdaptiveDBNN("prediction_mode")

            # Load the core DBNN model
            if 'model' in model_data:
                # The model is stored in the 'model' key in adaptive_dbnn
                self.adaptive_model.model = DBNNWrapper("prediction_mode")
                self.adaptive_model.model.core = DBNNCore()

                # Load the core model data using the adaptive_dbnn structure
                core_data = model_data['model']

                # Manually load the core components
                if 'core' in core_data:
                    # Load from the core sub-structure
                    core_model_data = core_data['core']
                    self.load_core_model_data(self.adaptive_model.model.core, core_model_data)
                else:
                    # Load directly from model data
                    self.load_core_model_data(self.adaptive_model.model.core, core_data)

            # Load feature information
            if 'feature_columns' in model_data:
                self.adaptive_model.feature_columns = model_data['feature_columns']
                self.log_output(f"📊 Loaded feature columns: {len(self.adaptive_model.feature_columns)}")

            if 'target_column' in model_data:
                self.adaptive_model.target_column = model_data['target_column']
                self.log_output(f"🎯 Target column: {self.adaptive_model.target_column}")

            # Load configuration
            if 'config' in model_data:
                self.adaptive_model.config = model_data['config']
                self.log_output("⚙️ Model configuration loaded")

            # Verify the encoder is properly fitted
            if (hasattr(self.adaptive_model.model.core, 'class_encoder') and
                hasattr(self.adaptive_model.model.core.class_encoder, 'is_fitted')):
                self.log_output(f"🔤 Class encoder fitted: {self.adaptive_model.model.core.class_encoder.is_fitted}")
                if self.adaptive_model.model.core.class_encoder.is_fitted:
                    self.log_output(f"📊 Encoded classes: {len(self.adaptive_model.model.core.class_encoder.encoded_to_class)}")

            self.log_output("✅ Adaptive model components loaded successfully")
            return True

        except Exception as e:
            self.log_output(f"❌ Error loading adaptive model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_core_model_data(self, core_instance, model_data):
        """Load core model data with proper encoder handling"""
        try:
            # Load basic configuration
            core_instance.config = model_data.get('config', {})

            # Load arrays
            array_mappings = [
                ('anti_net', np.int32),
                ('anti_wts', np.float64),
                ('binloc', np.float64),
                ('max_val', np.float64),
                ('min_val', np.float64),
                ('dmyclass', np.float64),
                ('resolution_arr', np.int32)
            ]

            for field_name, dtype in array_mappings:
                if field_name in model_data and model_data[field_name] is not None:
                    if isinstance(model_data[field_name], list):
                        loaded_array = np.array(model_data[field_name], dtype=dtype)
                    else:
                        loaded_array = model_data[field_name]
                    setattr(core_instance, field_name, loaded_array)

            # Infer dimensions from arrays
            if hasattr(core_instance, 'anti_net') and core_instance.anti_net is not None:
                core_instance.innodes = core_instance.anti_net.shape[0] - 2
                core_instance.outnodes = core_instance.anti_net.shape[4] - 2
                self.log_output(f"📊 Model dimensions: {core_instance.innodes} inputs, {core_instance.outnodes} outputs")

            # Load class encoder with robust error handling
            if 'class_encoder' in model_data:
                encoder_data = model_data['class_encoder']
                self.load_class_encoder(core_instance.class_encoder, encoder_data)
            else:
                # Try to infer encoder from dmyclass
                self.infer_encoder_from_dmyclass(core_instance)

            core_instance.is_trained = True
            return True

        except Exception as e:
            self.log_output(f"❌ Error loading core model data: {e}")
            return False

    def load_class_encoder(self, encoder_instance, encoder_data):
        """Load class encoder with proper error handling"""
        try:
            if 'encoded_to_class' in encoder_data and 'class_to_encoded' in encoder_data:
                # Convert keys to appropriate types
                encoded_to_class = {}
                for k, v in encoder_data['encoded_to_class'].items():
                    try:
                        key = float(k) if isinstance(k, (int, float, str)) else k
                        encoded_to_class[key] = v
                    except (ValueError, TypeError):
                        self.log_output(f"⚠️ Could not convert encoder key: {k}")

                class_to_encoded = {}
                for k, v in encoder_data['class_to_encoded'].items():
                    try:
                        value = float(v) if isinstance(v, (int, float, str)) else v
                        class_to_encoded[k] = value
                    except (ValueError, TypeError):
                        self.log_output(f"⚠️ Could not convert encoder value: {v}")

                encoder_instance.encoded_to_class = encoded_to_class
                encoder_instance.class_to_encoded = class_to_encoded
                encoder_instance.is_fitted = True

                self.log_output(f"✅ Class encoder loaded with {len(encoded_to_class)} classes")
                if encoded_to_class:
                    sample = list(encoded_to_class.items())[:3]
                    self.log_output(f"📋 Sample classes: {sample}")
            else:
                self.log_output("⚠️ No encoder mapping found in model data")
                encoder_instance.is_fitted = False

        except Exception as e:
            self.log_output(f"❌ Error loading class encoder: {e}")
            encoder_instance.is_fitted = False

    def infer_encoder_from_dmyclass(self, core_instance):
        """Infer encoder from dmyclass values as fallback"""
        try:
            if hasattr(core_instance, 'dmyclass') and core_instance.dmyclass is not None:
                # Extract class values from dmyclass (skip margin at index 0)
                class_values = []
                for i in range(1, min(len(core_instance.dmyclass), core_instance.outnodes + 1)):
                    if core_instance.dmyclass[i] != 0:  # Skip zero values
                        class_values.append(core_instance.dmyclass[i])

                if class_values:
                    # Create basic encoder mapping
                    encoded_to_class = {}
                    class_to_encoded = {}
                    for i, class_val in enumerate(class_values, 1):
                        encoded_to_class[float(i)] = f"Class_{class_val}"
                        class_to_encoded[f"Class_{class_val}"] = float(i)

                    core_instance.class_encoder.encoded_to_class = encoded_to_class
                    core_instance.class_encoder.class_to_encoded = class_to_encoded
                    core_instance.class_encoder.is_fitted = True

                    self.log_output(f"✅ Inferred encoder from dmyclass with {len(class_values)} classes")
                    return True

            self.log_output("⚠️ Could not infer encoder from dmyclass")
            core_instance.class_encoder.is_fitted = False
            return False

        except Exception as e:
            self.log_output(f"❌ Error inferring encoder from dmyclass: {e}")
            core_instance.class_encoder.is_fitted = False
            return False

    def predict_with_adaptive_model(self):
        """Make predictions using the loaded adaptive model with better error handling"""
        try:
            if not hasattr(self, 'adaptive_model') or not self.adaptive_model:
                self.log_output("❌ No adaptive model loaded")
                return None

            # Prepare data for prediction
            if self.original_data is None:
                self.log_output("❌ No data available for prediction")
                return None

            # Use the feature columns from the model
            feature_columns = getattr(self.adaptive_model, 'feature_columns', None)
            if not feature_columns:
                # Fallback to current feature selection
                feature_columns = [col for col, var in self.feature_vars.items()
                                 if var.get() and col != self.target_var.get()]
                self.log_output(f"⚠️ Using current feature selection: {feature_columns}")

            # Create feature matrix
            X_pred = self.original_data[feature_columns].values

            self.log_output(f"📊 Making predictions on {len(X_pred)} samples...")

            # Use the adaptive model's predict method with encoder fallback
            if hasattr(self.adaptive_model, 'model') and self.adaptive_model.model:
                predictions = self.adaptive_model.model.predict(X_pred)

                # Try to decode predictions, with fallback if encoder fails
                decoded_predictions = self.safe_decode_predictions(predictions)

                self.log_output(f"✅ Generated {len(decoded_predictions)} predictions")
                return decoded_predictions
            else:
                self.log_output("❌ No model available for prediction")
                return None

        except Exception as e:
            self.log_output(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def safe_decode_predictions(self, predictions):
        """Safely decode predictions with multiple fallback strategies"""
        try:
            # Check if we have a valid encoder
            if (hasattr(self.adaptive_model.model.core, 'class_encoder') and
                self.adaptive_model.model.core.class_encoder.is_fitted):

                # Try to decode using the encoder
                decoded = self.adaptive_model.model.core.class_encoder.inverse_transform(predictions)
                self.log_output("✅ Predictions decoded using class encoder")
                return decoded

            else:
                # Fallback: use raw predictions with labeling
                self.log_output("⚠️ Using raw predictions (encoder not available)")
                decoded = [f"Class_{int(p)}" if isinstance(p, (int, float)) else str(p)
                          for p in predictions]
                return decoded

        except Exception as e:
            self.log_output(f"⚠️ Encoder decoding failed, using raw predictions: {e}")
            # Final fallback: convert to string
            return [str(p) for p in predictions]

    def load_model(self):
        """Load a model - UPDATED to handle both regular and adaptive_dbnn models"""
        model_path = filedialog.askopenfilename(
            title="Load Model",
            filetypes=[("Model files", "*.bin"), ("JSON files", "*.json"), ("All files", "*.*")]
        )

        if model_path:
            try:
                self.log_output(f"📥 Loading model: {model_path}")

                # First try to load as adaptive_dbnn model
                success = self.load_adaptive_model_for_prediction(model_path)

                if success:
                    self.log_output("✅ Adaptive DBNN model loaded successfully!")
                    self.model_loaded = True
                    self.model_type = "adaptive"

                    # Log model information
                    if hasattr(self.adaptive_model, 'feature_columns') and self.adaptive_model.feature_columns:
                        self.log_output(f"📊 Model features: {len(self.adaptive_model.feature_columns)}")
                        self.log_output(f"📊 Feature names: {self.adaptive_model.feature_columns}")
                    if hasattr(self.adaptive_model, 'target_column'):
                        self.log_output(f"🎯 Target column: {self.adaptive_model.target_column}")

                else:
                    # Fall back to regular DBNN model loading
                    self.log_output("🔄 Trying regular DBNN model format...")
                    cmd_interface = DBNNCommandLine()
                    success = cmd_interface.load_model(model_path)

                    if success:
                        self.log_output("✅ Regular DBNN model loaded successfully!")
                        self.cmd_interface = cmd_interface
                        self.model_loaded = True
                        self.model_type = "regular"

                        # Log basic model info
                        if hasattr(cmd_interface.core, 'innodes'):
                            self.log_output(f"📊 Input nodes: {cmd_interface.core.innodes}")
                        if hasattr(cmd_interface.core, 'outnodes'):
                            self.log_output(f"📊 Output nodes: {cmd_interface.core.outnodes}")
                    else:
                        self.log_output("❌ Failed to load model in any format")
                        return

                # Enable predict button if data is loaded
                if self.data_loaded:
                    self.log_output("🎯 Model ready for prediction on current data")
                else:
                    self.log_output("💡 Load data to make predictions")

            except Exception as e:
                self.log_output(f"❌ Error loading model: {e}")
                import traceback
                traceback.print_exc()

    def predict_with_loaded_model(self):
        """Predict using whichever model is loaded (adaptive or regular)"""
        if not self.model_loaded:
            messagebox.showwarning("Warning", "Please load a model first.")
            return

        if not self.data_loaded:
            messagebox.showwarning("Warning", "Please load data first.")
            return

        # Ask for output file
        output_file = filedialog.asksaveasfilename(
            title="Save Predictions As",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if not output_file:
            return

        try:
            self.log_output("🔮 Making predictions...")

            if self.model_type == "adaptive":
                predictions = self.predict_with_adaptive_model()
            else:  # regular model
                predictions = self.predict_with_regular_model()

            if predictions is not None:
                self.save_predictions(predictions, output_file)
                self.log_output("✅ Predictions completed successfully!")
            else:
                self.log_output("❌ Prediction failed")

        except Exception as e:
            self.log_output(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()

    def predict_with_regular_model(self):
        """Predict using regular DBNN model"""
        try:
            if not hasattr(self, 'cmd_interface') or not self.cmd_interface:
                self.log_output("❌ No regular model loaded")
                return None

            # Create temporary data file for prediction
            temp_data_file = "temp_prediction_data.csv"

            # Save current data to temporary file
            if hasattr(self.cmd_interface.core, 'feature_columns') and self.cmd_interface.core.feature_columns:
                # Use model's feature columns
                feature_columns = self.cmd_interface.core.feature_columns
            else:
                # Use current feature selection
                feature_columns = [col for col, var in self.feature_vars.items()
                                 if var.get() and col != self.target_var.get()]

            prediction_data = self.original_data[feature_columns]
            prediction_data.to_csv(temp_data_file, index=False)

            # Create args for prediction
            class Args:
                def __init__(self):
                    self.predict = temp_data_file
                    self.output = "temp_predictions.csv"
                    self.format = 'csv'
                    self.target = None  # No target for prediction
                    self.features = feature_columns
                    self.verbose = True

            args = Args()

            # Make prediction
            success = self.cmd_interface.predict_data(args)

            # Read results
            if success and os.path.exists("temp_predictions.csv"):
                import pandas as pd
                results = pd.read_csv("temp_predictions.csv")
                predictions = results['Prediction'].tolist()

                # Clean up temp files
                import os
                if os.path.exists(temp_data_file):
                    os.remove(temp_data_file)
                if os.path.exists("temp_predictions.csv"):
                    os.remove("temp_predictions.csv")

                return predictions
            else:
                return None

        except Exception as e:
            self.log_output(f"❌ Regular model prediction error: {e}")
            return None

    def save_predictions(self, predictions, output_file):
        """Save predictions to CSV file (works for both model types)"""
        try:
            # Create results DataFrame
            results_df = self.original_data.copy()
            results_df['Prediction'] = predictions

            # Add model type information
            results_df['Model_Type'] = self.model_type

            # Save to CSV
            results_df.to_csv(output_file, index=False)

            self.log_output(f"💾 Predictions saved to: {output_file}")
            self.log_output(f"📈 File contains {len(results_df)} predictions")

            # Show prediction distribution
            from collections import Counter
            pred_counts = Counter(predictions)
            self.log_output("Prediction distribution:")
            for pred, count in pred_counts.most_common():
                pct = (count / len(predictions)) * 100
                self.log_output(f"  {pred}: {count} ({pct:.1f}%)")

        except Exception as e:
            self.log_output(f"❌ Error saving predictions: {e}")

    # NEW METHODS FOR VISUALIZATION CONTROLS
    def open_visualization_location(self):
        """Open the visualization directory in file explorer"""
        try:
            if hasattr(self, 'adaptive_model') and self.adaptive_model:
                # Try comprehensive visualizer first, then fallback to basic
                if hasattr(self.adaptive_model, 'comprehensive_visualizer'):
                    viz_dir = self.adaptive_model.comprehensive_visualizer.output_dir
                elif hasattr(self.adaptive_model, 'visualizer'):
                    viz_dir = self.adaptive_model.visualizer.output_dir
                else:
                    self.log_output("❌ No visualizer found. Run adaptive learning first.")
                    return

                if viz_dir.exists():
                    import subprocess
                    import platform

                    system = platform.system()
                    if system == "Windows":
                        subprocess.Popen(f'explorer "{viz_dir}"')
                    elif system == "Darwin":  # macOS
                        subprocess.Popen(['open', str(viz_dir)])
                    else:  # Linux
                        subprocess.Popen(['xdg-open', str(viz_dir)])

                    self.log_output(f"📁 Opened visualization directory: {viz_dir}")

                    # List available visualization files
                    html_files = list(viz_dir.rglob("*.html"))
                    png_files = list(viz_dir.rglob("*.png"))
                    gif_files = list(viz_dir.rglob("*.gif"))

                    self.log_output(f"📊 Found: {len(html_files)} HTML, {len(png_files)} PNG, {len(gif_files)} GIF files")

                else:
                    self.log_output("❌ Visualization directory not found. Generate visualizations first.")
            else:
                self.log_output("❌ No model available. Please run adaptive learning first.")
        except Exception as e:
            self.log_output(f"❌ Error opening visualization location: {e}")

    def show_animations(self):
        """Show available animations and offer to open them"""
        try:
            if hasattr(self, 'adaptive_model') and self.adaptive_model:
                if hasattr(self.adaptive_model, 'comprehensive_visualizer'):
                    viz_dir = self.adaptive_model.comprehensive_visualizer.output_dir
                else:
                    self.log_output("❌ No visualizer found.")
                    return

                animation_files = list(viz_dir.rglob("*.gif")) + list(viz_dir.rglob("*.mp4"))

                if animation_files:
                    self.log_output("🎬 Available animations:")
                    for anim_file in animation_files:
                        self.log_output(f"   📹 {anim_file.relative_to(viz_dir)}")

                    # Ask if user wants to open the animations directory
                    if messagebox.askyesno("Open Animations",
                                          f"Found {len(animation_files)} animations. Open folder?"):
                        self.open_visualization_location()
                else:
                    self.log_output("❌ No animations found. Generate final visualizations first.")
            else:
                self.log_output("❌ No model available. Please run adaptive learning first.")
        except Exception as e:
            self.log_output(f"❌ Error showing animations: {e}")

    def show_interactive_3d(self):
        """Show interactive 3D visualizations"""
        try:
            if hasattr(self, 'adaptive_model') and self.adaptive_model:
                viz_dir = self.adaptive_model.comprehensive_visualizer.output_dir
                html_files = list(viz_dir.rglob("*.html"))

                interactive_3d_files = [f for f in html_files if "interactive" in f.name.lower() or "3d" in f.name.lower()]

                if interactive_3d_files:
                    self.log_output("🌐 Interactive 3D visualizations:")
                    for html_file in interactive_3d_files:
                        self.log_output(f"   🔗 {html_file.relative_to(viz_dir)}")

                    # Open the first interactive 3D file in default browser
                    import webbrowser
                    webbrowser.open(f"file://{interactive_3d_files[0].absolute()}")
                    self.log_output(f"📂 Opening: {interactive_3d_files[0].name}")
                else:
                    self.log_output("❌ No interactive 3D visualizations found. Run adaptive learning first.")
            else:
                self.log_output("❌ No model available. Please run adaptive learning first.")
        except Exception as e:
            self.log_output(f"❌ Error showing interactive 3D: {e}")

    def browse_data_file(self):
        """Browse for data file."""
        file_path = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=[("CSV files", "*.csv"), ("DAT files", "*.dat"), ("All files", "*.*")]
        )
        if file_path:
            self.data_file_var.set(file_path)
            self.current_data_file = file_path
            self.log_output(f"📁 Selected file: {file_path}")

            # Try to load configuration automatically
            self.load_configuration_for_file(file_path)

    def load_data_file(self):
        """Load data file and populate feature selection."""
        file_path = self.data_file_var.get()
        if not file_path or not os.path.exists(file_path):
            messagebox.showwarning("Warning", "Please select a valid data file.")
            return

        try:
            # Load data
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.dat'):
                # Try to load DAT file with space separation
                try:
                    df = pd.read_csv(file_path, delimiter=r'\s+')
                except:
                    df = pd.read_csv(file_path)  # Fallback to default
            else:
                messagebox.showerror("Error", "Unsupported file format.")
                return

            self.current_data_file = file_path
            self.original_data = df.copy()

            # Update data info
            self.update_data_info(df)

            # Update feature selection UI
            self.update_feature_selection_ui(df)

            self.data_loaded = True
            self.log_output(f"✅ Data loaded successfully: {len(df)} samples, {len(df.columns)} columns")

        except Exception as e:
            self.log_output(f"❌ Error loading data: {e}")

    def update_data_info(self, df):
        """Update data information display."""
        self.data_info_text.config(state=tk.NORMAL)
        self.data_info_text.delete(1.0, tk.END)

        info_text = f"""📊 DATA INFORMATION
{'='*40}
Samples: {len(df)}
Features: {len(df.columns)}
Columns: {', '.join(df.columns.tolist())}

Data Types:
"""
        for col in df.columns:
            dtype = df[col].dtype
            unique_count = df[col].nunique()
            info_text += f"  {col}: {dtype} (unique: {unique_count})\n"

        missing_values = df.isnull().sum()
        if missing_values.any():
            info_text += f"\nMissing Values:\n"
            for col in df.columns:
                if missing_values[col] > 0:
                    info_text += f"  {col}: {missing_values[col]}\n"

        self.data_info_text.insert(1.0, info_text)
        self.data_info_text.config(state=tk.DISABLED)

    def update_feature_selection_ui(self, df):
        """Update the feature selection UI with available columns."""
        # Clear existing feature checkboxes
        for widget in self.feature_scroll_frame.winfo_children():
            widget.destroy()

        self.feature_vars = {}
        columns = df.columns.tolist()

        # Update target combo with ALL columns
        self.target_combo['values'] = columns

        # Auto-select target if not set
        if not self.target_var.get() and columns:
            # Try common target column names
            target_candidates = ['target', 'class', 'label', 'y', 'output', 'result']
            for candidate in target_candidates + [columns[-1]]:
                if candidate in columns:
                    self.target_var.set(candidate)
                    break

        # Create feature checkboxes
        for i, col in enumerate(columns):
            var = tk.BooleanVar(value=col != self.target_var.get())  # Auto-select non-target columns
            self.feature_vars[col] = var

            # Determine column type for styling
            if pd.api.types.is_numeric_dtype(df[col]):
                col_type = "numeric"
                color = "blue"
            elif pd.api.types.is_string_dtype(df[col]):
                col_type = "categorical"
                color = "green"
            else:
                col_type = "other"
                color = "gray"

            display_text = f"{col} ({col_type})"

            # Highlight target column
            if col == self.target_var.get():
                display_text = f"🎯 {display_text} [TARGET]"
                # Don't allow target to be selected as feature
                cb = ttk.Checkbutton(self.feature_scroll_frame, text=display_text, variable=var, state="disabled")
            else:
                cb = ttk.Checkbutton(self.feature_scroll_frame, text=display_text, variable=var)

            cb.pack(anchor=tk.W, padx=5, pady=2)

        self.log_output(f"🔧 Available columns: {len(columns)} total")
        self.log_output(f"🎯 Current target: {self.target_var.get()}")

    def on_target_selected(self, event):
        """Handle target column selection."""
        # When target changes, update feature selection states
        if hasattr(self, 'feature_vars') and self.target_var.get():
            for col, var in self.feature_vars.items():
                if col == self.target_var.get():
                    var.set(False)
                else:
                    var.set(True)

    def select_all_features(self):
        """Select all features."""
        for col, var in self.feature_vars.items():
            if col != self.target_var.get():  # Don't select target as feature
                var.set(True)

    def deselect_all_features(self):
        """Deselect all features."""
        for var in self.feature_vars.values():
            var.set(False)

    def select_numeric_features(self):
        """Select only numeric features."""
        if not hasattr(self, 'original_data'):
            return

        df = self.original_data
        for col, var in self.feature_vars.items():
            if col != self.target_var.get() and pd.api.types.is_numeric_dtype(df[col]):
                var.set(True)
            else:
                var.set(False)

    def apply_feature_selection(self):
        """Apply the current feature selection."""
        if not self.data_loaded:
            messagebox.showwarning("Warning", "Please load data first.")
            return

        try:
            # Get selected features
            selected_features = []
            for col, var in self.feature_vars.items():
                if var.get() and col != self.target_var.get():
                    selected_features.append(col)

            # Get target column
            target_column = self.target_var.get()

            if not selected_features:
                messagebox.showwarning("Warning", "Please select at least one feature.")
                return

            if not target_column:
                messagebox.showwarning("Warning", "Please select a target column.")
                return

            # Initialize adaptive model
            dataset_name = os.path.splitext(os.path.basename(self.current_data_file))[0]

            # Create configuration
            config = {
                'target_column': target_column,
                'feature_columns': selected_features,
                'resol': int(self.config_vars.get('dbnn_resolution', tk.StringVar(value="100")).get()),
                'gain': float(self.config_vars.get('dbnn_gain', tk.StringVar(value="2.0")).get()),
                'margin': float(self.config_vars.get('dbnn_margin', tk.StringVar(value="0.2")).get()),
                'patience': int(self.config_vars.get('dbnn_patience', tk.StringVar(value="10")).get()),
                'max_epochs': int(self.config_vars.get('dbnn_max_epochs', tk.StringVar(value="100")).get()),
                'min_improvement': float(self.config_vars.get('dbnn_min_improvement', tk.StringVar(value="0.0000001")).get()),

                'adaptive_learning': {
                    'enable_adaptive': True,
                    'initial_samples_per_class': int(self.initial_samples_var.get()),
                    'max_adaptive_rounds': int(self.max_rounds_var.get()),
                    'max_margin_samples_per_class': int(self.max_samples_var.get()),
                    'enable_acid_test': self.enable_acid_var.get(),
                    'enable_kl_divergence': self.enable_kl_var.get(),
                    'disable_sample_limit': self.disable_sample_limit_var.get(),
                    'enable_visualization': self.enable_visualization_var.get(),  # NEW
                    'margin_tolerance': float(self.config_vars.get('adaptive_margin_tolerance', tk.StringVar(value="0.15")).get()),
                    'kl_threshold': float(self.config_vars.get('adaptive_kl_threshold', tk.StringVar(value="0.1")).get()),
                    'patience': int(self.config_vars.get('dbnn_patience', tk.StringVar(value="10")).get()),
                    'min_improvement': float(self.config_vars.get('dbnn_min_improvement', tk.StringVar(value="0.0000001")).get()),
                }
            }

            self.adaptive_model = AdaptiveDBNN(dataset_name, config)

            self.log_output(f"✅ Feature selection applied")
            self.log_output(f"🎯 Target: {target_column}")
            self.log_output(f"📊 Selected features: {len(selected_features)}")
            self.log_output(f"🔧 Features: {', '.join(selected_features)}")

            # Save configuration
            self.save_configuration_for_file(self.current_data_file)

        except Exception as e:
            self.log_output(f"❌ Error applying feature selection: {e}")

    def load_default_parameters(self):
        """Load default hyperparameters."""
        try:
            # Set the instance variables directly
            self.max_rounds_var.set("20")
            self.max_samples_var.set("25")
            self.initial_samples_var.set("5")

            # DBNN Core defaults
            self.config_vars["dbnn_resolution"].set("100")
            self.config_vars["dbnn_gain"].set("2.0")
            self.config_vars["dbnn_margin"].set("0.2")
            self.config_vars["dbnn_patience"].set("10")
            self.config_vars["dbnn_max_epochs"].set("100")
            self.config_vars["dbnn_min_improvement"].set("0.0000001")

            # Adaptive learning defaults
            self.config_vars["adaptive_margin_tolerance"].set("0.15")
            self.config_vars["adaptive_kl_threshold"].set("0.1")
            self.config_vars["adaptive_training_convergence_epochs"].set("50")
            self.config_vars["adaptive_min_training_accuracy"].set("0.95")
            self.config_vars["adaptive_adaptive_margin_relaxation"].set("0.1")

            self.enable_acid_var.set(True)
            self.enable_kl_var.set(False)
            self.disable_sample_limit_var.set(False)

            self.log_output("✅ Loaded default parameters")

        except Exception as e:
            self.log_output(f"❌ Error loading default parameters: {e}")

    def save_current_parameters(self):
        """Save current hyperparameters to configuration file."""
        if not self.current_data_file:
            messagebox.showwarning("Warning", "Please load a data file first.")
            return

        try:
            self.save_configuration_for_file(self.current_data_file)
            self.log_output("✅ Current parameters saved to configuration file")

        except Exception as e:
            self.log_output(f"❌ Error saving parameters: {e}")

    def save_configuration_for_file(self, file_path):
        """Save configuration for specific data file."""
        try:
            config_file = self.get_config_file_path(file_path)

            config = {
                'dataset_name': os.path.splitext(os.path.basename(file_path))[0],
                'target_column': self.target_var.get(),
                'feature_columns': [col for col, var in self.feature_vars.items() if var.get() and col != self.target_var.get()],

                'resol': int(self.config_vars["dbnn_resolution"].get()),
                'gain': float(self.config_vars["dbnn_gain"].get()),
                'margin': float(self.config_vars["dbnn_margin"].get()),
                'patience': int(self.config_vars["dbnn_patience"].get()),
                'max_epochs': int(self.config_vars["dbnn_max_epochs"].get()),
                'min_improvement': float(self.config_vars["dbnn_min_improvement"].get()),

                'adaptive_learning': {
                    'enable_adaptive': True,
                    'initial_samples_per_class': int(self.initial_samples_var.get()),
                    'max_adaptive_rounds': int(self.max_rounds_var.get()),
                    'max_margin_samples_per_class': int(self.max_samples_var.get()),
                    'enable_acid_test': self.enable_acid_var.get(),
                    'enable_kl_divergence': self.enable_kl_var.get(),
                    'disable_sample_limit': self.disable_sample_limit_var.get(),
                    'margin_tolerance': float(self.config_vars["adaptive_margin_tolerance"].get()),
                    'kl_threshold': float(self.config_vars["adaptive_kl_threshold"].get()),
                    'training_convergence_epochs': int(self.config_vars["adaptive_training_convergence_epochs"].get()),
                    'min_training_accuracy': float(self.config_vars["adaptive_min_training_accuracy"].get()),
                    'adaptive_margin_relaxation': float(self.config_vars["adaptive_adaptive_margin_relaxation"].get()),
                }
            }

            with open(config_file, 'w') as f:
                json.dump(config, f, indent=4)

            self.log_output(f"💾 Configuration saved to: {config_file}")

        except Exception as e:
            self.log_output(f"❌ Error saving configuration: {e}")

    def load_configuration_for_file(self, file_path):
        """Load configuration for specific data file."""
        try:
            config_file = self.get_config_file_path(file_path)

            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)

                # Apply configuration
                if 'target_column' in config:
                    self.target_var.set(config['target_column'])

                if 'resol' in config:
                    self.config_vars["dbnn_resolution"].set(str(config['resol']))
                if 'gain' in config:
                    self.config_vars["dbnn_gain"].set(str(config['gain']))
                if 'margin' in config:
                    self.config_vars["dbnn_margin"].set(str(config['margin']))
                if 'patience' in config:
                    self.config_vars["dbnn_patience"].set(str(config['patience']))
                if 'max_epochs' in config:
                    self.config_vars["dbnn_max_epochs"].set(str(config['max_epochs']))
                if 'min_improvement' in config:
                    self.config_vars["dbnn_min_improvement"].set(str(config['min_improvement']))

                if 'adaptive_learning' in config:
                    adaptive_config = config['adaptive_learning']
                    self.max_rounds_var.set(str(adaptive_config.get('max_adaptive_rounds', 20)))
                    self.max_samples_var.set(str(adaptive_config.get('max_margin_samples_per_class', 25)))
                    self.initial_samples_var.set(str(adaptive_config.get('initial_samples_per_class', 5)))

                    self.enable_acid_var.set(adaptive_config.get('enable_acid_test', True))
                    self.enable_kl_var.set(adaptive_config.get('enable_kl_divergence', False))
                    self.disable_sample_limit_var.set(adaptive_config.get('disable_sample_limit', False))

                    self.config_vars["adaptive_margin_tolerance"].set(str(adaptive_config.get('margin_tolerance', 0.15)))
                    self.config_vars["adaptive_kl_threshold"].set(str(adaptive_config.get('kl_threshold', 0.1)))
                    self.config_vars["adaptive_training_convergence_epochs"].set(str(adaptive_config.get('training_convergence_epochs', 50)))
                    self.config_vars["adaptive_min_training_accuracy"].set(str(adaptive_config.get('min_training_accuracy', 0.95)))
                    self.config_vars["adaptive_adaptive_margin_relaxation"].set(str(adaptive_config.get('adaptive_margin_relaxation', 0.1)))

                self.log_output(f"📂 Loaded configuration from: {config_file}")
            else:
                self.log_output("ℹ️ No existing configuration found. Using defaults.")

        except Exception as e:
            self.log_output(f"❌ Error loading configuration: {e}")

    def get_config_file_path(self, data_file_path):
        """Get configuration file path for data file."""
        base_name = os.path.splitext(data_file_path)[0]
        return f"{base_name}_adaptive_config.json"

    def apply_hyperparameters(self):
        """Apply current hyperparameters to the model and make them immediately effective"""
        if not self.data_loaded or self.adaptive_model is None:
            messagebox.showwarning("Warning", "Please load data and apply feature selection first.")
            return

        try:
            # Update model configuration with ALL current GUI values
            config = {
                'resol': int(self.config_vars["dbnn_resolution"].get()),
                'gain': float(self.config_vars["dbnn_gain"].get()),
                'margin': float(self.config_vars["dbnn_margin"].get()),
                'patience': int(self.config_vars["dbnn_patience"].get()),
                'max_epochs': int(self.config_vars["dbnn_max_epochs"].get()),
                'min_improvement': float(self.config_vars["dbnn_min_improvement"].get()),

                'adaptive_learning': {
                    'initial_samples_per_class': int(self.initial_samples_var.get()),
                    'max_adaptive_rounds': int(self.max_rounds_var.get()),
                    'max_margin_samples_per_class': int(self.max_samples_var.get()),
                    'enable_acid_test': self.enable_acid_var.get(),
                    'enable_kl_divergence': self.enable_kl_var.get(),
                    'disable_sample_limit': self.disable_sample_limit_var.get(),
                    'enable_visualization': self.enable_visualization_var.get(),
                    'margin_tolerance': float(self.config_vars["adaptive_margin_tolerance"].get()),
                    'kl_threshold': float(self.config_vars["adaptive_kl_threshold"].get()),
                    'training_convergence_epochs': int(self.config_vars["adaptive_training_convergence_epochs"].get()),
                    'min_training_accuracy': float(self.config_vars["adaptive_min_training_accuracy"].get()),
                    'adaptive_margin_relaxation': float(self.config_vars["adaptive_adaptive_margin_relaxation"].get()),
                }
            }

            # Update the model configuration
            self.adaptive_model.config.update(config)
            if hasattr(self.adaptive_model, 'adaptive_config'):
                self.adaptive_model.adaptive_config.update(config.get('adaptive_learning', {}))

            self.log_output("✅ Hyperparameters applied and effective immediately")
            self.log_output(f"   Resolution: {self.config_vars['dbnn_resolution'].get()}")
            self.log_output(f"   Max Rounds: {self.max_rounds_var.get()}")
            self.log_output(f"   Acid Test: {'Enabled' if self.enable_acid_var.get() else 'Disabled'}")
            self.log_output(f"   Visualization: {'Enabled' if self.enable_visualization_var.get() else 'Disabled'}")

            # Force GUI refresh
            self.refresh_gui_values()

        except Exception as e:
            self.log_output(f"❌ Error applying hyperparameters: {e}")

    def initialize_model(self):
        """Initialize the model."""
        if not self.data_loaded or self.adaptive_model is None:
            messagebox.showwarning("Warning", "Please load data and apply feature selection first.")
            return

        try:
            # Prepare data with selected features
            feature_columns = [col for col, var in self.feature_vars.items() if var.get() and col != self.target_var.get()]
            self.adaptive_model.prepare_full_data(feature_columns=feature_columns)

            self.log_output("✅ Model initialized successfully")
            self.log_output(f"📊 Dataset: {self.adaptive_model.X_full.shape[0]} samples, {self.adaptive_model.X_full.shape[1]} features")

        except Exception as e:
            self.log_output(f"❌ Error initializing model: {e}")

    def display_results(self, results):
        """Display adaptive learning results."""
        if results is None:
            return

        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)

        # Format results
        self.results_text.insert(tk.END, "🏆 ADAPTIVE LEARNING RESULTS\n")
        self.results_text.insert(tk.END, "=" * 60 + "\n\n")

        # Basic results
        self.results_text.insert(tk.END, f"📁 Dataset: {results.get('dataset_name', 'Unknown')}\n")
        self.results_text.insert(tk.END, f"🎯 Target Column: {results.get('target_column', 'Unknown')}\n")
        self.results_text.insert(tk.END, f"🔧 Features Used: {len(results.get('feature_names', []))}\n")
        self.results_text.insert(tk.END, f"📦 Total Samples: {len(self.adaptive_model.X_full) if self.adaptive_model and self.adaptive_model.X_full is not None else 'Unknown'}\n\n")

        # Performance results
        self.results_text.insert(tk.END, "📊 PERFORMANCE SUMMARY\n")
        self.results_text.insert(tk.END, "-" * 40 + "\n")
        self.results_text.insert(tk.END, f"🎯 Final Accuracy: {results.get('final_accuracy', 0.0):.4f}\n")
        self.results_text.insert(tk.END, f"🏆 Best Accuracy: {results.get('best_accuracy', 0.0):.4f}\n")
        self.results_text.insert(tk.END, f"🔄 Best Round: {results.get('best_round', 0)}\n")
        self.results_text.insert(tk.END, f"📊 Final Training Size: {results.get('final_training_size', 0)}\n")
        self.results_text.insert(tk.END, f"⏱️ Total Training Time: {results.get('total_training_time', 0.0):.2f} seconds\n")
        self.results_text.insert(tk.END, f"🔄 Total Rounds: {results.get('total_rounds', 0)}\n\n")

        # Feature information
        feature_names = results.get('feature_names', [])
        if feature_names:
            self.results_text.insert(tk.END, "🔧 FEATURES USED\n")
            self.results_text.insert(tk.END, "-" * 40 + "\n")
            features_text = ", ".join(feature_names)
            # Split long feature lists into multiple lines
            if len(features_text) > 80:
                words = features_text.split(', ')
                lines = []
                current_line = ""
                for word in words:
                    if len(current_line + word) > 80:
                        lines.append(current_line)
                        current_line = word + ", "
                    else:
                        current_line += word + ", "
                if current_line:
                    lines.append(current_line.rstrip(', '))

                for line in lines:
                    self.results_text.insert(tk.END, f"  {line}\n")
            else:
                self.results_text.insert(tk.END, f"  {features_text}\n")
            self.results_text.insert(tk.END, "\n")

        # Configuration summary
        self.results_text.insert(tk.END, "⚙️ CONFIGURATION SUMMARY\n")
        self.results_text.insert(tk.END, "-" * 40 + "\n")

        adaptive_config = results.get('adaptive_config', {})
        if adaptive_config:
            self.results_text.insert(tk.END, "Adaptive Learning:\n")
            self.results_text.insert(tk.END, f"  • Max Rounds: {adaptive_config.get('max_adaptive_rounds', 'N/A')}\n")
            self.results_text.insert(tk.END, f"  • Samples/Round: {adaptive_config.get('max_margin_samples_per_class', 'N/A')}\n")
            self.results_text.insert(tk.END, f"  • Initial Samples/Class: {adaptive_config.get('initial_samples_per_class', 'N/A')}\n")
            self.results_text.insert(tk.END, f"  • Acid Test: {adaptive_config.get('enable_acid_test', 'N/A')}\n")
            self.results_text.insert(tk.END, f"  • KL Divergence: {adaptive_config.get('enable_kl_divergence', 'N/A')}\n")

        model_config = results.get('model_config', {})
        if model_config:
            self.results_text.insert(tk.END, "DBNN Model:\n")
            self.results_text.insert(tk.END, f"  • Resolution: {model_config.get('resol', 'N/A')}\n")
            self.results_text.insert(tk.END, f"  • Gain: {model_config.get('gain', 'N/A')}\n")
            self.results_text.insert(tk.END, f"  • Margin: {model_config.get('margin', 'N/A')}\n")
            self.results_text.insert(tk.END, f"  • Patience: {model_config.get('patience', 'N/A')}\n")

        # Add timestamp
        self.results_text.insert(tk.END, f"\n📅 Results generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        self.results_text.config(state=tk.DISABLED)

    def run_adaptive_learning(self):
        """Run adaptive learning."""
        if not self.data_loaded or self.adaptive_model is None:
            messagebox.showwarning("Warning", "Please load data, apply feature selection, and initialize model first.")
            return

        try:
            self.log_output("🚀 Starting adaptive learning...")

            # Get selected features for adaptive learning
            feature_columns = [col for col, var in self.feature_vars.items() if var.get() and col != self.target_var.get()]

            # Run adaptive learning
            X_train, y_train, X_test, y_test = self.adaptive_model.adaptive_learn(feature_columns=feature_columns)

            # Create results dictionary for display
            results = {
                'dataset_name': self.adaptive_model.dataset_name,
                'target_column': self.adaptive_model.target_column,
                'feature_names': self.adaptive_model.feature_columns,
                'final_accuracy': self.adaptive_model.best_accuracy,
                'best_accuracy': self.adaptive_model.best_accuracy,
                'best_round': getattr(self.adaptive_model, 'best_round', 0),
                'final_training_size': len(getattr(self.adaptive_model, 'best_training_indices', [])),
                'total_training_time': getattr(self.adaptive_model, 'total_training_time', 0),
                'total_rounds': getattr(self.adaptive_model, 'adaptive_round', 0),
                'round_stats': getattr(self.adaptive_model, 'round_stats', []),
                'adaptive_config': getattr(self.adaptive_model, 'adaptive_config', {}),
                'model_config': self.adaptive_model.config,
                'training_indices': getattr(self.adaptive_model, 'best_training_indices', [])
            }

            # Display results
            self.display_results(results)

            self.model_trained = True
            self.log_output("✅ Adaptive learning completed successfully!")
            self.log_output(f"🏆 Best accuracy: {self.adaptive_model.best_accuracy:.4f}")
            self.log_output(f"📊 Final training size: {len(X_train)} samples")
            self.log_output(f"📊 Test set size: {len(X_test)} samples")

        except Exception as e:
            self.log_output(f"❌ Error during adaptive learning: {e}")

    def evaluate_model(self):
        """Evaluate the model."""
        if not self.model_trained or self.adaptive_model is None:
            messagebox.showwarning("Warning", "Please run adaptive learning first.")
            return

        try:
            self.log_output("📊 Evaluating model...")

            # Use the test set from adaptive learning
            if hasattr(self.adaptive_model, 'X_test') and hasattr(self.adaptive_model, 'y_test'):
                X_test = self.adaptive_model.X_test
                y_test = self.adaptive_model.y_test

                predictions = self.adaptive_model.model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)

                self.log_output(f"🎯 Test accuracy: {accuracy:.4f}")
                self.log_output(f"📊 Test set size: {len(X_test)} samples")
            else:
                self.log_output("⚠️ No test set available for evaluation")

        except Exception as e:
            self.log_output(f"❌ Error during evaluation: {e}")

    def show_visualizations(self):
        """Show basic model visualizations."""
        if not self.model_trained or self.adaptive_model is None:
            messagebox.showwarning("Warning", "No trained model available for visualization.")
            return

        try:
            self.log_output("📊 Generating basic visualizations...")

            # Create basic visualizations using the existing visualizer
            if hasattr(self.adaptive_model, 'adaptive_visualizer'):
                self.adaptive_model.adaptive_visualizer.create_visualizations(
                    self.adaptive_model.X_full,
                    self.adaptive_model.y_full
                )
                self.log_output("✅ Basic visualizations created in 'adaptive_visualizations' directory")
            else:
                self.log_output("⚠️ Visualizer not available")

        except Exception as e:
            self.log_output(f"❌ Error showing visualizations: {e}")

    def show_advanced_analysis(self):
        """Show advanced analysis."""
        if not self.model_trained or self.adaptive_model is None:
            messagebox.showwarning("Warning", "No trained model available for analysis.")
            return

        try:
            self.log_output("🔬 Generating advanced analysis...")

            # Generate adaptive learning report
            if hasattr(self.adaptive_model, '_generate_adaptive_learning_report'):
                self.adaptive_model._generate_adaptive_learning_report()
                self.log_output("✅ Advanced analysis report generated")
            else:
                self.log_output("⚠️ Advanced analysis not available")

        except Exception as e:
            self.log_output(f"❌ Error during advanced analysis: {e}")

    def save_model(self):
        """Save the model."""
        if not self.model_trained or self.adaptive_model is None:
            messagebox.showwarning("Warning", "No trained model to save.")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Model As",
            defaultextension=".bin",
            filetypes=[("Model files", "*.bin"), ("All files", "*.*")]
        )

        if file_path:
            try:
                # Use the model's save functionality
                success = self.adaptive_model.model.core.save_model_auto(
                    model_dir=os.path.dirname(file_path),
                    data_filename=self.current_data_file,
                    feature_columns=self.adaptive_model.feature_columns,
                    target_column=self.adaptive_model.target_column
                )

                if success:
                    self.log_output(f"✅ Model saved to: {file_path}")
                else:
                    self.log_output(f"❌ Failed to save model")

            except Exception as e:
                self.log_output(f"❌ Error saving model: {e}")

    def log_output(self, message: str):
        """Add message to output text."""
        self.output_text.insert(tk.END, f"{message}\n")
        self.output_text.see(tk.END)
        self.root.update()
        self.status_var.set(message)

def launch_adaptive_gui():
    """Launch the Adaptive DBNN GUI."""
    root = tk.Tk()
    app = AdaptiveDBNNGUI(root)
    root.mainloop()


class DatasetConfig:
    """Dataset configuration handler"""

    @staticmethod
    def get_available_datasets():
        """Get list of available datasets from configuration files"""
        config_files = glob.glob("*.conf") + glob.glob("*.json")
        datasets = []
        for f in config_files:
            # Remove both .conf and .json extensions
            base_name = f.replace('.conf', '').replace('.json', '')
            if base_name not in datasets:  # Avoid duplicates
                datasets.append(base_name)
        return datasets

    @staticmethod
    def load_config(dataset_name):
        """Load configuration for a dataset - supports both .conf and .json"""
        # Try .json first, then .conf
        config_paths = [
            f"{dataset_name}.json",
            f"{dataset_name}.conf"
        ]

        for config_path in config_paths:
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    print(f"⚠️  Warning: Could not load config from {config_path}: {e}")
                    continue
        return {}

    @staticmethod
    def get_available_config_files():
        """Get all available configuration files with their types"""
        config_files = []
        # Look for JSON config files
        json_files = glob.glob("*.json")
        for f in json_files:
            # Skip the auto-saved config to avoid confusion
            if not f.endswith('_run_config.json') and not f.endswith('adaptive_dbnn_config.json'):
                config_files.append({'file': f, 'type': 'JSON'})

        # Look for CONF config files
        conf_files = glob.glob("*.conf")
        for f in conf_files:
            config_files.append({'file': f, 'type': 'CONF'})

        return config_files

class DataPreprocessor:
    """Comprehensive data preprocessing for DBNN"""

    def __init__(self, target_column: str = 'target', sentinel_value: float = -99999.0):
        self.target_column = target_column
        self.sentinel_value = sentinel_value
        self.feature_encoders = {}
        self.target_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_columns = []  # Store original feature names
        self.missing_value_indicators = {}
        self.column_dtypes = {}  # Track original data types

    def preprocess_dataset(self, data: pd.DataFrame, feature_columns: List[str] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Preprocess entire dataset with specified feature columns"""
        print("🔧 Preprocessing dataset...")

        if self.target_column not in data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset")

        # Use specified feature columns or all non-target columns
        if feature_columns is None:
            self.feature_columns = [col for col in data.columns if col != self.target_column]
        else:
            # Validate that specified feature columns exist
            missing_cols = [col for col in feature_columns if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Feature columns not found in data: {missing_cols}")
            self.feature_columns = feature_columns

        print(f"📊 Using features: {self.feature_columns}")

        # Create a copy with only the selected features and target
        data_clean = data[self.feature_columns + [self.target_column]].copy()

        # Preprocess features
        X_processed = self._preprocess_features_with_names(data_clean[self.feature_columns])

        # Preprocess target
        y_processed = self.preprocess_target(data_clean[self.target_column])

        # Remove samples with missing target values
        valid_mask = ~np.isnan(y_processed)
        if not np.all(valid_mask):
            removed_count = len(y_processed) - np.sum(valid_mask)
            print(f"⚠️  Removed {removed_count} samples with invalid target values")
            X_processed = X_processed[valid_mask]
            y_processed = y_processed[valid_mask]

        print(f"✅ Preprocessing complete: {X_processed.shape[0]} samples, {X_processed.shape[1]} features")
        print(f"📊 Features used: {self.feature_columns}")

        return X_processed, y_processed, self.feature_columns

    def _preprocess_features_with_names(self, X: pd.DataFrame) -> np.ndarray:
        """Preprocess feature columns while preserving original names"""
        processed_features = []

        for col in X.columns:
            feature_data = X[col].copy()

            # Store original dtype
            self.column_dtypes[col] = feature_data.dtype

            # Handle missing values
            missing_mask = self._detect_missing_values(feature_data)

            # Convert to numeric
            numeric_data = self._convert_to_numeric(feature_data, col)

            # Store missing value information
            self.missing_value_indicators[col] = {
                'missing_mask': missing_mask,
                'has_missing': np.any(missing_mask)
            }

            processed_features.append(numeric_data)

        # Stack all features
        if processed_features:
            X_processed = np.column_stack(processed_features)
        else:
            X_processed = np.empty((len(X), 0))

        return X_processed

    def preprocess_features(self, X: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Preprocess feature columns - handle mixed types, missing values, etc."""
        processed_features = []
        feature_names = []

        for col in X.columns:
            if col == self.target_column:
                continue

            feature_data = X[col].copy()

            # Handle missing values
            missing_mask = self._detect_missing_values(feature_data)

            # Convert to numeric, handling errors
            numeric_data = self._convert_to_numeric(feature_data, col)

            # Store missing value information
            self.missing_value_indicators[col] = {
                'missing_mask': missing_mask,
                'has_missing': np.any(missing_mask)
            }

            processed_features.append(numeric_data)
            feature_names.append(col)

        # Stack all features
        if processed_features:
            X_processed = np.column_stack(processed_features)
        else:
            X_processed = np.empty((len(X), 0))

        return X_processed, feature_names

    def _detect_missing_values(self, data: pd.Series) -> np.ndarray:
        """Detect various types of missing values"""
        # Standard missing values
        missing_mask = data.isna()

        # String representations of missing values
        if data.dtype == 'object':
            missing_strings = ['', 'NA', 'N/A', 'null', 'NULL', 'None', 'NaN', 'nan', 'ERROR', 'error', 'MISSING', 'missing']
            missing_mask = missing_mask | data.isin(missing_strings)

        return missing_mask.values

    def _convert_to_numeric(self, data: pd.Series, col_name: str) -> np.ndarray:
        """Convert data to numeric, handling various data types"""
        # If already numeric, return as is
        if pd.api.types.is_numeric_dtype(data):
            numeric_data = data.values.astype(float)
            # Replace any remaining NaN with sentinel value
            numeric_data = np.where(np.isnan(numeric_data), self.sentinel_value, numeric_data)
            return numeric_data

        # For categorical/string data
        if data.dtype == 'object':
            try:
                # Try direct conversion to numeric first
                numeric_data = pd.to_numeric(data, errors='coerce').values
                # Replace NaN with sentinel value
                numeric_data = np.where(np.isnan(numeric_data), self.sentinel_value, numeric_data)
                return numeric_data
            except:
                # Use label encoding for categorical data
                if col_name not in self.feature_encoders:
                    self.feature_encoders[col_name] = LabelEncoder()

                # Handle missing values before encoding
                clean_data = data.fillna('MISSING')
                encoded_data = self.feature_encoders[col_name].fit_transform(clean_data)
                return encoded_data.astype(float)

        # Fallback: convert to string then label encode
        str_data = data.astype(str)
        if col_name not in self.feature_encoders:
            self.feature_encoders[col_name] = LabelEncoder()
        encoded_data = self.feature_encoders[col_name].fit_transform(str_data)
        return encoded_data.astype(float)

    def preprocess_target(self, y: pd.Series) -> np.ndarray:
        """Preprocess target column"""
        # Handle missing target values
        if y.isna().any():
            print(f"⚠️  Warning: Found {y.isna().sum()} missing target values. They will be removed.")
            # We'll handle this at the dataset level by removing these samples

        # Convert to numeric if needed
        if not pd.api.types.is_numeric_dtype(y):
            try:
                y_processed = pd.to_numeric(y, errors='coerce')
                if y_processed.isna().any():
                    print(f"⚠️  Some target values couldn't be converted to numeric. Using label encoding.")
                    y_processed = self.target_encoder.fit_transform(y.fillna('MISSING'))
                else:
                    y_processed = y_processed.values
            except:
                y_processed = self.target_encoder.fit_transform(y.fillna('MISSING'))
        else:
            y_processed = y.values

        return y_processed.astype(int)

class DBNNVisualizer:
    """Visualization system for DBNN"""

    def __init__(self, model, output_dir='visualizations', enabled=False):
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

class DBNNWrapper:
    """
    Wrapper for dbnn.py module that implements the exact adaptive learning requirements
    """

    def __init__(self, dataset_name: str = None, config: Dict = None):
        self.dataset_name = dataset_name
        self.config = config or {}

        # Initialize the core DBNN
        dbnn_config = {
            'resol': self.config.get('resol', 100),
            'gain': self.config.get('gain', 2.0),
            'margin': self.config.get('margin', 0.2),
            'patience': self.config.get('patience', 10),
            'epochs': self.config.get('max_epochs', 100),
            'min_improvement': self.config.get('min_improvement',0.0000001)
        }
        self.core = dbnn.DBNNCore(dbnn_config)

        # Store feature information
        self.feature_columns = []  # Original feature column names
        self.target_column = self.config.get('target_column', 'target')
        self.preprocessor = DataPreprocessor(target_column=self.target_column)

        # Training state
        self.initialized_with_full_data = False

        # Store architectural components separately for freezing
        self.architecture_frozen = False
        self.frozen_components = {}

        # Store data and preprocessing
        self.data = None
        self.preprocessor = DataPreprocessor(target_column=self.target_column)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

        # Training state
        self.train_enabled = True
        self.max_epochs = self.config.get('max_epochs', 100)
        self.test_size = self.config.get('test_size', 0.2)
        self.random_state = self.config.get('random_state', 42)

        # Feature information
        self.feature_names = []
        self.initialized_with_full_data = False

    def load_data(self, file_path: str = None, feature_columns: List[str] = None):
        """Load data from file with original column names"""
        if file_path is None:
            # Auto-detect data file (existing logic)
            possible_files = [
                f"{self.dataset_name}.csv",
                f"{self.dataset_name}.data",
                f"wine.data",
                f"wine.csv",
                "data.csv",
                "train.csv"
            ]
            for file in possible_files:
                if os.path.exists(file):
                    file_path = file
                    print(f"📁 Found data file: {file_path}")
                    break

        if file_path is None:
            csv_files = glob.glob("*.csv")
            dat_files = glob.glob("*.dat")
            all_files = csv_files + dat_files
            if all_files:
                file_path = all_files[0]
                print(f"📁 Auto-selected data file: {file_path}")
            else:
                raise ValueError("No data file found.")

        # Load data
        if file_path.endswith('.csv'):
            self.data = pd.read_csv(file_path)
            print(f"✅ Loaded CSV data: {self.data.shape[0]} samples, {self.data.shape[1]} columns")
            print(f"📊 Columns: {list(self.data.columns)}")
        else:
            # For .dat files, create proper column names
            print(f"📊 Loading DAT file: {file_path}")
            try:
                data = np.loadtxt(file_path)
                if feature_columns is None:
                    n_features = data.shape[1] - 1
                    feature_columns = [f'feature_{i}' for i in range(n_features)]
                columns = feature_columns + [self.target_column]
                self.data = pd.DataFrame(data, columns=columns)
                print(f"✅ Loaded DAT data: {self.data.shape[0]} samples, {self.data.shape[1]} columns")
            except Exception as e:
                print(f"❌ Error loading DAT file: {e}")
                raise

        return self.data

    def preprocess_data(self, feature_columns: List[str] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Preprocess the loaded data with specified feature columns"""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        X, y, feature_columns_used = self.preprocessor.preprocess_dataset(self.data, feature_columns)
        self.feature_columns = feature_columns_used  # Store for prediction
        return X, y, feature_columns_used


    def initialize_with_full_data(self, X: np.ndarray, y: np.ndarray, feature_columns: List[str]):
        """Step 1: Initialize DBNN architecture with full dataset and feature names"""
        print("🏗️ Initializing DBNN architecture with full dataset...")
        self.feature_columns = feature_columns  # Store feature names

        # Create temporary file with full data and original column names
        temp_file = f"temp_full_init_{int(time.time())}.csv"
        full_df = pd.DataFrame(X, columns=self.feature_columns)
        full_df[self.target_column] = y
        full_df.to_csv(temp_file, index=False)

        try:
            # Initialize DBNN architecture with feature names
            self._initialize_dbnn_architecture(X, y, self.feature_columns)

            # Build network structure
            print("🔧 Building network structure...")
            success = self._build_network_structure_only(temp_file, self.feature_columns)

            if success:
                print("✅ DBNN architecture initialized with full dataset")
                self.initialized_with_full_data = True
                self.freeze_architecture()
            else:
                print("❌ Failed to initialize DBNN architecture")

        except Exception as e:
            print(f"❌ Initialization error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)


    def predict_with_original_columns(self, X: np.ndarray, input_features: List[str] = None):
        """Predict classes for input data with feature column validation"""
        if not hasattr(self.core, 'is_trained') or not self.core.is_trained:
            unique_classes = np.unique(self.y_full) if hasattr(self, 'y_full') else [1, 2, 3]
            return np.random.choice(unique_classes, size=len(X))

        try:
            # If input_features provided, reorder columns to match training features
            if input_features is not None and hasattr(self, 'feature_columns'):
                print(f"🔧 Reordering features to match training configuration")
                print(f"   Training features: {self.feature_columns}")
                print(f"   Input features: {input_features}")

                # Create a DataFrame to facilitate column reordering
                if isinstance(X, np.ndarray):
                    X_df = pd.DataFrame(X, columns=input_features)
                else:
                    X_df = X.copy()

                # Reorder columns to match training feature order
                missing_cols = [col for col in self.feature_columns if col not in X_df.columns]
                if missing_cols:
                    raise ValueError(f"Missing feature columns: {missing_cols}")

                X_ordered = X_df[self.feature_columns].values
                print(f"✅ Features reordered for prediction")
            else:
                X_ordered = X

            # Use the core prediction with ordered features
            return self.predict(X_ordered)

        except Exception as e:
            print(f"❌ Prediction error with column reordering: {e}")
            # Fallback to direct prediction
            return self.predict(X)

    def _build_network_structure_only(self, train_file: str, feature_cols: List[str]):
        """Build network structure without training - just initialize counts"""
        try:
            # Load data without re-initializing
            features_batches, targets_batches, feature_columns_used, original_targets_batches = self.core.load_data(
                train_file, self.target_column, feature_cols
            )

            if not features_batches:
                return False

            # Encode targets using existing encoder
            encoded_targets_batches = []
            for batch in original_targets_batches:
                encoded_batch = self.core.class_encoder.transform(batch)
                encoded_targets_batches.append(encoded_batch)

            # Initialize training parameters (minimal setup)
            resol = self.core.config.get('resol', 100)
            omax, omin = self._initialize_training_params(features_batches, encoded_targets_batches, resol)

            # Process just a few samples to build the network structure
            # We don't need to process all samples for architecture setup
            total_samples = sum(len(batch) for batch in features_batches)
            sample_limit = min(100, total_samples)  # Process only up to 100 samples

            print(f"📊 Building network structure with {sample_limit} samples...")
            processed_count = 0

            for batch_idx, (features_batch, targets_batch) in enumerate(zip(features_batches, encoded_targets_batches)):
                batch_size = len(features_batch)

                for sample_idx in range(min(batch_size, sample_limit - processed_count)):
                    vects = np.zeros(self.core.innodes + self.core.outnodes + 2)
                    for i in range(1, self.core.innodes + 1):
                        vects[i] = features_batch[sample_idx, i-1]
                    tmpv = targets_batch[sample_idx]

                    # Use the core's processing function to build network counts
                    self.core.anti_net = dbnn.process_training_sample(
                        vects, tmpv, self.core.anti_net, self.core.anti_wts, self.core.binloc,
                        self.core.resolution_arr, self.core.dmyclass, self.core.min_val, self.core.max_val,
                        self.core.innodes, self.core.outnodes
                    )

                    processed_count += 1

                if processed_count >= sample_limit:
                    break

            print(f"✅ Network structure built with {processed_count} samples")

            # Mark as trained to allow predictions, but note it's architecture-only
            self.core.is_trained = True
            self.core.is_architecture_only = True  # Add this flag

            return True

        except Exception as e:
            print(f"❌ Network structure building error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _initialize_dbnn_architecture(self, X: np.ndarray, y: np.ndarray, feature_cols: List[str]):
        """Manually initialize DBNN architecture to avoid the dmyclass error"""
        print("🔧 Manually initializing DBNN architecture...")

        # Create temporary file for initialization
        temp_file = f"temp_init_{int(time.time())}.csv"
        init_df = pd.DataFrame(X, columns=feature_cols)
        init_df[self.target_column] = y
        init_df.to_csv(temp_file, index=False)

        try:
            # Load data to get feature information
            features_batches, targets_batches, feature_columns_used, original_targets_batches = self.core.load_data(
                temp_file, self.target_column, feature_cols
            )

            if not features_batches:
                raise ValueError("No data loaded for initialization")

            # Fit encoder first
            all_original_targets = np.concatenate(original_targets_batches) if original_targets_batches else np.array([])
            self.core.class_encoder.fit(all_original_targets)

            # Get encoded classes
            encoded_classes = self.core.class_encoder.get_encoded_classes()
            self.core.outnodes = len(encoded_classes)
            self.core.innodes = len(feature_cols)

            # Initialize arrays with proper dimensions
            resol = self.core.config.get('resol', 100)
            self.core.initialize_arrays(self.core.innodes, resol, self.core.outnodes)

            # Now set dmyclass values safely
            self.core.dmyclass[0] = self.core.config.get('margin', 0.2)
            for i, encoded_val in enumerate(encoded_classes, 1):
                if i < len(self.core.dmyclass):
                    self.core.dmyclass[i] = float(encoded_val)

            print(f"✅ Manual initialization complete: {self.core.innodes} inputs, {self.core.outnodes} outputs")

        except Exception as e:
            print(f"❌ Manual initialization failed: {e}")
            raise
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def _train_with_initialized_architecture(self, train_file: str, feature_cols: List[str]):
        """Train with acid test-based early stopping"""
        try:
            # Load data without re-initializing
            features_batches, targets_batches, feature_columns_used, original_targets_batches = self.core.load_data(
                train_file, self.target_column, feature_cols
            )

            if not features_batches:
                return False

            # Encode targets using existing encoder
            encoded_targets_batches = []
            for batch in original_targets_batches:
                encoded_batch = self.core.class_encoder.transform(batch)
                encoded_targets_batches.append(encoded_batch)

            # Initialize training parameters
            resol = self.core.config.get('resol', 100)
            omax, omin = self._initialize_training_params(features_batches, encoded_targets_batches, resol)

            # Process training data for initial APF
            total_samples = sum(len(batch) for batch in features_batches)
            total_processed = 0

            for batch_idx, (features_batch, targets_batch) in enumerate(zip(features_batches, encoded_targets_batches)):
                processed = self._process_training_batch(features_batch, targets_batch)
                total_processed += processed

            # Training with acid test-based early stopping
            gain = self.core.config.get('gain', 2.0)
            max_epochs = self.core.config.get('epochs', 100)
            patience = self.core.config.get('patience', 10)
            min_improvement = self.core.config.get('min_improvement', 0.0000001)

            print(f"Starting weight training (max {max_epochs} epochs)...")
            best_accuracy = 0.0
            self.best_weights = None
            self.best_round = 0
            patience_counter = 0

            for rnd in range(max_epochs + 1):
                if rnd == 0:
                    # Initial evaluation
                    current_accuracy, correct_predictions, _ = self._evaluate_model(features_batches, encoded_targets_batches)
                    print(f"Epoch {rnd:3d}: Accuracy = {current_accuracy:.2f}% ({correct_predictions}/{total_samples})")
                    self.best_accuracy = current_accuracy
                    self.best_weights = self.core.anti_wts.copy()
                    self.best_round = rnd

                    # Stop immediately if we already have 100% accuracy
                    if current_accuracy >= 100.0:
                        print("🎉 Already at 100% accuracy - stopping training!")
                        break
                    continue

                # Training pass
                self._train_epoch(features_batches, encoded_targets_batches, gain)

                # Evaluation after training epoch
                current_accuracy, correct_predictions, _ = self._evaluate_model(features_batches, encoded_targets_batches)

                # Stop immediately if we reach 100% accuracy
                if current_accuracy >= 100.0:
                    print(f"Epoch {rnd:3d}: Accuracy = {current_accuracy:.2f}% - 🎉 100% ACCURACY REACHED!")
                    self.best_accuracy = current_accuracy
                    self.best_weights = self.core.anti_wts.copy()
                    self.best_round = rnd
                    break

                print(f"Epoch {rnd:3d}: Accuracy = {current_accuracy:.2f}% ({correct_predictions}/{total_samples})")

                # Check for improvement
                if current_accuracy > self.best_accuracy:
                    improvement = current_accuracy - self.best_accuracy
                    self.best_accuracy = current_accuracy
                    self.best_weights = self.core.anti_wts.copy()
                    self.best_round = rnd
                    patience_counter = 0
                    print(f"  → Improved by {improvement:.2f}%")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"  → Early stopping after {rnd} epochs (no improvement for {patience} epochs)")
                        break
                    else:
                        print(f"  → No improvement (patience: {patience_counter}/{patience})")

            # Restore best weights
            if self.best_weights is not None:
                self.core.anti_wts = self.best_weights
                print(f"✅ Training completed - Best accuracy: {self.best_accuracy:.2f}% (epoch {self.best_round})")

            self.core.is_trained = True
            return True

        except Exception as e:
            print(f"❌ Training error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _initialize_training_params(self, features_batches, encoded_targets_batches, resol: int):
        """Initialize training parameters without re-initializing arrays"""
        # Find min/max values
        omax = -400.0
        omin = 400.0

        for batch_idx, (features_batch, targets_batch) in enumerate(zip(features_batches, encoded_targets_batches)):
            for i in range(1, self.core.innodes + 1):
                batch_max = np.max(features_batch[:, i-1])
                batch_min = np.min(features_batch[:, i-1])
                if batch_max > self.core.max_val[i]:
                    self.core.max_val[i] = batch_max
                if batch_min < self.core.min_val[i]:
                    self.core.min_val[i] = batch_min

            # Update omax/omin from targets
            batch_omax = np.max(targets_batch)
            batch_omin = np.min(targets_batch)
            if batch_omax > omax:
                omax = batch_omax
            if batch_omin < omin:
                omin = batch_omin

        # Set resolutions
        for i in range(1, self.core.innodes + 1):
            self.core.resolution_arr[i] = resol
            for j in range(self.core.resolution_arr[i] + 1):
                self.core.binloc[i][j+1] = j * 1.0

        # Initialize network counts
        self.core.anti_wts.fill(1.0)
        for k in range(1, self.core.outnodes + 1):
            for i in range(1, self.core.innodes + 1):
                for j in range(self.core.resolution_arr[i] + 1):
                    for l in range(1, self.core.innodes + 1):
                        for m in range(self.core.resolution_arr[l] + 1):
                            self.core.anti_net[i, j, l, m, k] = 1

        return omax, omin

    def _process_training_batch(self, features_batch, targets_batch):
        """Process a single batch of training data"""
        batch_size = len(features_batch)
        processed_count = 0

        for sample_idx in range(batch_size):
            vects = np.zeros(self.core.innodes + self.core.outnodes + 2)
            for i in range(1, self.core.innodes + 1):
                vects[i] = features_batch[sample_idx, i-1]
            tmpv = targets_batch[sample_idx]

            # Use the core's processing function
            self.core.anti_net = dbnn.process_training_sample(
                vects, tmpv, self.core.anti_net, self.core.anti_wts, self.core.binloc,
                self.core.resolution_arr, self.core.dmyclass, self.core.min_val, self.core.max_val,
                self.core.innodes, self.core.outnodes
            )

            processed_count += 1

        return processed_count

    def _train_epoch(self, features_batches, encoded_targets_batches, gain: float):
        """Train for one epoch"""
        for batch_idx, (features_batch, targets_batch) in enumerate(zip(features_batches, encoded_targets_batches)):
            batch_size = len(features_batch)

            for sample_idx in range(batch_size):
                vects = np.zeros(self.core.innodes + self.core.outnodes + 2)
                for i in range(1, self.core.innodes + 1):
                    vects[i] = features_batch[sample_idx, i-1]
                tmpv = targets_batch[sample_idx]

                # Compute probabilities
                classval = dbnn.compute_class_probabilities_numba(
                    vects, self.core.anti_net, self.core.anti_wts, self.core.binloc, self.core.resolution_arr,
                    self.core.dmyclass, self.core.min_val, self.core.max_val, self.core.innodes, self.core.outnodes
                )

                # Find predicted class
                kmax = 1
                cmax = 0.0
                for k in range(1, self.core.outnodes + 1):
                    if classval[k] > cmax:
                        cmax = classval[k]
                        kmax = k

                # Update weights if wrong classification
                if abs(self.core.dmyclass[kmax] - tmpv) > self.core.dmyclass[0]:
                    self.core.anti_wts = dbnn.update_weights_numba(
                        vects, tmpv, classval, self.core.anti_wts, self.core.binloc, self.core.resolution_arr,
                        self.core.dmyclass, self.core.min_val, self.core.max_val, self.core.innodes, self.core.outnodes, gain
                    )

    def _evaluate_model(self, features_batches, encoded_targets_batches):
        """Evaluate model accuracy"""
        correct_predictions = 0
        total_samples = 0
        all_predictions = []

        for batch_idx, (features_batch, targets_batch) in enumerate(zip(features_batches, encoded_targets_batches)):
            batch_size = len(features_batch)
            total_samples += batch_size

            for sample_idx in range(batch_size):
                vects = np.zeros(self.core.innodes + self.core.outnodes + 2)
                for i in range(1, self.core.innodes + 1):
                    vects[i] = features_batch[sample_idx, i-1]
                actual = targets_batch[sample_idx]

                # Compute class probabilities
                classval = dbnn.compute_class_probabilities_numba(
                    vects, self.core.anti_net, self.core.anti_wts, self.core.binloc, self.core.resolution_arr,
                    self.core.dmyclass, self.core.min_val, self.core.max_val, self.core.innodes, self.core.outnodes
                )

                # Find predicted class
                kmax = 1
                cmax = 0.0
                for k in range(1, self.core.outnodes + 1):
                    if classval[k] > cmax:
                        cmax = classval[k]
                        kmax = k

                predicted = self.core.dmyclass[kmax]
                all_predictions.append(predicted)

                # Check if prediction is correct
                if abs(actual - predicted) <= self.core.dmyclass[0]:
                    correct_predictions += 1

        accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
        return accuracy, correct_predictions, all_predictions

    def train_with_data(self, X_train: np.ndarray, y_train: np.ndarray, reset_weights: bool = True):
        """Step 2: Train with given data (no train/test split)"""
        if not self.initialized_with_full_data:
            # Try to initialize if not already done
            print("⚠️  DBNN not initialized, attempting initialization...")
            self.initialize_with_full_data(X_train, y_train, self.feature_columns)  # Pass feature columns
            if not self.initialized_with_full_data:
                raise ValueError("DBNN must be initialized with full data first")

        if reset_weights:
            print("🔄 Resetting weights for new training...")
            self._reset_weights()

        print(f"🎯 Training with {len(X_train)} samples...")

        # Create temporary file with training data - USE ORIGINAL FEATURE NAMES
        temp_file = f"temp_train_{int(time.time())}.csv"

        # Use original feature columns if available, otherwise fallback to generic names
        if hasattr(self, 'feature_columns') and self.feature_columns:
            feature_cols = self.feature_columns
            print(f"📊 Using original feature names: {feature_cols}")
        else:
            feature_cols = [f'feature_{i}' for i in range(X_train.shape[1])]
            print(f"⚠️  Using generic feature names: {feature_cols}")

        train_df = pd.DataFrame(X_train, columns=feature_cols)
        train_df[self.target_column] = y_train
        train_df.to_csv(temp_file, index=False)

        try:
            # Train using our custom training method that preserves architecture
            success = self._train_with_initialized_architecture(temp_file, feature_cols)

            if success:
                train_accuracy = self._compute_accuracy(X_train, y_train)
                print(f"✅ Training completed - Accuracy: {train_accuracy:.2f}%")
                return train_accuracy
            else:
                print("❌ Training failed")
                return 0.0

        except Exception as e:
            print(f"❌ Training error: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def _compute_accuracy(self, X: np.ndarray, y: np.ndarray):
        """Compute accuracy on given data"""
        if not hasattr(self.core, 'is_trained') or not self.core.is_trained:
            return 0.0

        try:
            predictions = self.predict(X)
            accuracy = accuracy_score(y, predictions)
            return accuracy * 100
        except Exception as e:
            print(f"❌ Accuracy computation error: {e}")
            return 0.0

    def predict(self, X: np.ndarray):
        """Predict classes for input data"""
        if not hasattr(self.core, 'is_trained') or not self.core.is_trained:
            # Return random predictions if not trained
            unique_classes = np.unique(self.y_full) if hasattr(self, 'y_full') else [1, 2, 3]
            return np.random.choice(unique_classes, size=len(X))

        try:
            # Convert to batches for prediction
            batch_size = 1000
            all_predictions = []

            for i in range(0, len(X), batch_size):
                batch = X[i:i+batch_size]
                batch_predictions, _ = self.core.predict_batch(batch)
                all_predictions.extend(batch_predictions)

            return np.array(all_predictions)
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            # Return random predictions as fallback
            unique_classes = np.unique(self.y_full) if hasattr(self, 'y_full') else [1, 2, 3]
            return np.random.choice(unique_classes, size=len(X))

    def freeze_architecture(self):
        """Freeze the current architecture for later restoration"""
        if not self.initialized_with_full_data:
            raise ValueError("Cannot freeze architecture: DBNN not initialized")

        print("❄️  Freezing DBNN architecture...")

        # Store critical components
        self.frozen_components = {
            'innodes': self.core.innodes,
            'outnodes': self.core.outnodes,
            'anti_net': self.core.anti_net.copy(),
            'dmyclass': self.core.dmyclass.copy(),
            'binloc': self.core.binloc.copy(),
            'max_val': self.core.max_val.copy(),
            'min_val': self.core.min_val.copy(),
            'resolution_arr': self.core.resolution_arr.copy(),
            'class_encoder': copy.deepcopy(self.core.class_encoder),
            'feature_columns': self.feature_columns.copy()
        }

        self.architecture_frozen = True
        print(f"✅ Architecture frozen: {self.core.innodes} inputs, {self.core.outnodes} outputs")

    def restore_architecture(self):
        """Restore the frozen architecture"""
        if not self.architecture_frozen:
            raise ValueError("No frozen architecture to restore")

        print("🔄 Restoring frozen architecture...")

        # Restore critical components
        self.core.innodes = self.frozen_components['innodes']
        self.core.outnodes = self.frozen_components['outnodes']
        self.core.anti_net = self.frozen_components['anti_net'].copy()
        self.core.dmyclass = self.frozen_components['dmyclass'].copy()
        self.core.binloc = self.frozen_components['binloc'].copy()
        self.core.max_val = self.frozen_components['max_val'].copy()
        self.core.min_val = self.frozen_components['min_val'].copy()
        self.core.resolution_arr = self.frozen_components['resolution_arr'].copy()
        self.core.class_encoder = copy.deepcopy(self.frozen_components['class_encoder'])
        self.feature_columns = self.frozen_components['feature_columns'].copy()

        # Reset weights but keep architecture
        self._reset_weights()

        print(f"✅ Architecture restored: {self.core.innodes} inputs, {self.core.outnodes} outputs")

    def _reset_weights(self):
        """Reset weights while preserving architecture"""
        if hasattr(self.core, 'anti_wts'):
            self.core.anti_wts.fill(1.0)
        if hasattr(self.core, 'antit_wts'):
            self.core.antit_wts.fill(1.0)
        if hasattr(self.core, 'antip_wts'):
            self.core.antip_wts.fill(1.0)

    def adaptive_train(self, X_train: np.ndarray, y_train: np.ndarray, reset_weights: bool = True):
        """Adaptive training that preserves architecture"""
        if not self.initialized_with_full_data:
            raise ValueError("DBNN must be initialized with full data first")

        if reset_weights:
            self._reset_weights()

        return self.train_with_data(X_train, y_train, reset_weights=False)

class AdaptiveDBNN:
    """
    Advanced Adaptive Learning DBNN with comprehensive feature support
    """

    def __init__(self, dataset_name: str = None, config: Dict = None):
        self.dataset_name = dataset_name
        self.config = config or {}
        self.adaptive_config = self._setup_adaptive_config()
        self.progress_callback = None

        # Initialize DBNN wrapper - THIS IS THE MAIN MODEL
        self.model = DBNNWrapper(dataset_name, config)  # Changed from self.model to self.model

        # Adaptive learning state
        self.adaptive_round = 0
        self.best_accuracy = 0.0
        self.convergence_history = []
        self.margin_history = []
        self.kl_divergence_history = []
        self.adaptive_samples_added = 0

        # Data storage
        self.X_full = None
        self.y_full = None
        self.X_train_current = None
        self.y_train_current = None
        self.X_test = None
        self.y_test = None

        # Feature information
        self.feature_columns = []
        self.target_column = self.config.get('target_column', 'target')

        # Preprocessor
        self.preprocessor = DataPreprocessor(target_column=self.target_column)

        # Visualization
        self.visualizer = None
        self.enable_3d = self.adaptive_config.get('enable_3d_visualization', False)

        # Adaptive learning metrics
        self.margin_samples_per_class = defaultdict(list)
        self.divergence_samples_per_class = defaultdict(list)
        self.failed_samples = []

        # Comprehensive visualizer
        self.comprehensive_visualizer = ComprehensiveAdaptiveVisualizer(dataset_name)

        # Model saving configuration
        self.models_dir = Path('Models')
        self.models_dir.mkdir(exist_ok=True)

        # Training history tracking
        self.training_history = []
        self.round_stats = []

        # Enhanced visualization
        self.advanced_visualizer = AdvancedInteractiveVisualizer(dataset_name)

        print("🎯 Adaptive DBNN initialized with configuration:")
        for key, value in self.adaptive_config.items():
            print(f"  {key:40}: {value}")

    def set_progress_callback(self, callback):
        """Set callback for progress updates"""
        self.progress_callback = callback


    def _setup_adaptive_config(self) -> Dict[str, Any]:
        """Setup adaptive learning configuration with defaults"""
        default_config = {
            'enable_adaptive': True,
            'initial_samples_per_class': 10,
            'margin': 0.1,
            'max_adaptive_rounds': 10,
            'patience': 3,
            'min_improvement': 0.0000001,
            'max_margin_samples_per_class': 3,
            'margin_tolerance': 0.15,
            'kl_threshold': 0.1,
            'training_convergence_epochs': 50,
            'min_training_accuracy': 0.95,
            'min_samples_to_add_per_class': 5,
            'adaptive_margin_relaxation': 0.1,
            'max_divergence_samples_per_class': 5,
            'exhaust_all_failed': True,
            'min_failed_threshold': 10,
            'enable_kl_divergence': False,
            'max_samples_per_class_fallback': 2,
            'enable_3d_visualization': False,
            '3d_snapshot_interval': 10,
            'learning_rate': 1.0,
            'enable_acid_test': True,
            'min_training_percentage_for_stopping': 10.0,
            'max_training_percentage': 90.0,
            'kl_divergence_threshold': 0.1,
            'max_kl_samples_per_class': 5,
            'disable_sample_limit': False,
            'architecture_freeze_epochs': 50,
            'adaptive_training_epochs': 20
        }

        # Update with provided config
        adaptive_config = default_config.copy()
        adaptive_config.update(self.config.get('adaptive', {}))

        # Update with direct config values
        for key in default_config:
            if key in self.config:
                adaptive_config[key] = self.config[key]

        return adaptive_config

    def _should_create_visualizations(self, round_num: int) -> bool:
        """Determine whether to create visualizations based on round and configuration"""
        if not self.adaptive_config.get('enable_visualization', False):
            return False

        # Create visualizations only at strategic points to save time
        if round_num == 1:  # Always create first round
            return True
        elif round_num <= 10 and round_num % 2 == 0:  # Every 2 rounds for first 10
            return True
        elif round_num <= 50 and round_num % 5 == 0:  # Every 5 rounds for next 40
            return True
        elif round_num % 10 == 0:  # Every 10 rounds after that
            return True

        return False

    def load_and_preprocess_data(self, file_path: str = None, feature_columns: List[str] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load and preprocess data with feature column support"""
        print("📊 Loading and preprocessing data...")

        # Load data
        data = self.model.load_data(file_path, feature_columns)

        # Preprocess data
        X, y, feature_columns_used = self.model.preprocess_data(feature_columns)

        # Store full dataset
        self.X_full = X
        self.y_full = y
        self.feature_columns = feature_columns_used

        print(f"✅ Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"📊 Feature columns: {self.feature_columns}")
        print(f"🎯 Classes: {np.unique(y)}")

        return X, y, feature_columns_used

    def prepare_full_data(self, feature_columns: List[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare full dataset with feature columns - FIXED METHOD"""
        print("📊 Preparing full dataset...")

        # Load and preprocess data if not already done
        if self.X_full is None or self.y_full is None:
            self.X_full, self.y_full, self.feature_columns = self.load_and_preprocess_data(feature_columns=feature_columns)

        # Return the data
        return self.X_full, self.y_full, self.y_full  # Return y_full twice for compatibility

    def initialize_with_full_data(self, feature_columns: List[str] = None):
        """Initialize DBNN with full dataset architecture"""
        print("🏗️ Initializing DBNN with full dataset architecture...")

        # Prepare full data
        X_full, y_full, _ = self.prepare_full_data(feature_columns)

        # Initialize DBNN with full data
        self.model.initialize_with_full_data(X_full, y_full, self.feature_columns)

        print(f"✅ DBNN initialized with full dataset: {X_full.shape[0]} samples")

    def create_initial_training_set(self, initial_samples_per_class: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Create initial training set with specified samples per class"""
        if initial_samples_per_class is None:
            initial_samples_per_class = self.adaptive_config['initial_samples_per_class']

        print(f"🎯 Creating initial training set ({initial_samples_per_class} samples per class)...")

        X_initial = []
        y_initial = []

        unique_classes = np.unique(self.y_full)
        for class_label in unique_classes:
            class_indices = np.where(self.y_full == class_label)[0]
            n_samples = min(initial_samples_per_class, len(class_indices))

            if n_samples > 0:
                selected_indices = np.random.choice(class_indices, n_samples, replace=False)
                X_initial.append(self.X_full[selected_indices])
                y_initial.append(self.y_full[selected_indices])

        X_train = np.vstack(X_initial)
        y_train = np.hstack(y_initial)

        print(f"✅ Initial training set: {X_train.shape[0]} samples")
        print(f"📊 Class distribution: {np.unique(y_train, return_counts=True)}")

        return X_train, y_train


    def _select_initial_training_samples(self, X: np.ndarray, y: np.ndarray, initial_samples_per_class: int = None) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """Select initial diverse training samples using K-means clustering"""
        if initial_samples_per_class is None:
            initial_samples_per_class = self.adaptive_config['initial_samples_per_class']

        print(f"🎯 Selecting initial training samples ({initial_samples_per_class} samples per class)...")

        X_initial = []
        y_initial = []
        initial_indices = []

        unique_classes = np.unique(y)

        for class_label in unique_classes:
            class_indices = np.where(y == class_label)[0]
            n_samples = min(initial_samples_per_class, len(class_indices))

            if n_samples > 0:
                if len(class_indices) > n_samples:
                    # Use k-means to select diverse samples
                    class_data = X[class_indices]

                    try:
                        from sklearn.cluster import KMeans
                        kmeans = KMeans(n_clusters=n_samples, init='k-means++', n_init=1, random_state=42)
                        kmeans.fit(class_data)

                        # Find samples closest to cluster centers
                        distances = kmeans.transform(class_data)
                        closest_indices = np.argmin(distances, axis=0)
                        selected_class_indices = class_indices[closest_indices]
                    except:
                        # Fallback: random selection
                        selected_class_indices = np.random.choice(class_indices, n_samples, replace=False)
                else:
                    # Use all available samples for this class
                    selected_class_indices = class_indices

                X_initial.append(X[selected_class_indices])
                y_initial.append(y[selected_class_indices])
                initial_indices.extend(selected_class_indices.tolist())

        if X_initial:
            X_train = np.vstack(X_initial)
            y_train = np.hstack(y_initial)
        else:
            X_train = np.array([]).reshape(0, X.shape[1])
            y_train = np.array([])

        print(f"✅ Initial training set: {X_train.shape[0]} samples")
        print(f"📊 Class distribution: {np.unique(y_train, return_counts=True)}")

        return X_train, y_train, initial_indices

    def _select_divergent_samples(self, X_remaining: np.ndarray, y_remaining: np.ndarray,
                                predictions: np.ndarray, posteriors: np.ndarray,
                                misclassified_indices: np.ndarray, remaining_indices: List[int]) -> List[int]:
        """Select most divergent failed candidates from each class"""
        print("🔍 Selecting most divergent failed candidates...")

        samples_to_add = []
        unique_classes = np.unique(y_remaining)

        # Group misclassified samples by true class
        class_samples = defaultdict(list)

        for idx_in_remaining in misclassified_indices:
            original_idx = remaining_indices[idx_in_remaining]
            true_class = y_remaining[idx_in_remaining]
            pred_class = predictions[idx_in_remaining]

            # Convert class labels to indices for array access
            true_class_idx_result = np.where(unique_classes == true_class)[0]
            pred_class_idx_result = np.where(unique_classes == pred_class)[0]

            if len(true_class_idx_result) == 0 or len(pred_class_idx_result) == 0:
                continue

            true_class_idx = true_class_idx_result[0]
            pred_class_idx = pred_class_idx_result[0]

            # Calculate margin (divergence)
            true_posterior = posteriors[idx_in_remaining, true_class_idx] if posteriors is not None and posteriors.shape[1] > true_class_idx else 0.0
            pred_posterior = posteriors[idx_in_remaining, pred_class_idx] if posteriors is not None and posteriors.shape[1] > pred_class_idx else 0.0
            margin = pred_posterior - true_posterior

            class_samples[true_class].append({
                'index': original_idx,
                'margin': margin,
                'true_posterior': true_posterior,
                'pred_posterior': pred_posterior
            })

        # For each class, select most divergent samples
        max_samples = self.adaptive_config.get('max_margin_samples_per_class', 2)

        for class_id in unique_classes:
            if class_id not in class_samples or not class_samples[class_id]:
                continue

            class_data = class_samples[class_id]

            # Sort by margin (most negative first - most divergent)
            class_data.sort(key=lambda x: x['margin'])

            # Select top divergent samples
            selected_for_class = class_data[:max_samples]

            for sample in selected_for_class:
                samples_to_add.append(sample['index'])

            if selected_for_class:
                print(f"   ✅ Class {class_id}: Selected {len(selected_for_class)} divergent samples")

        print(f"📥 Total divergent samples to add: {len(samples_to_add)}")
        return samples_to_add

    def adaptive_learn(self, feature_columns: List[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Main adaptive learning method with CORRECT best model selection"""
        print("\n🚀 STARTING ADAPTIVE LEARNING")
        print("=" * 60)

        # Use provided data or prepare full data
        if self.X_full is None or self.y_full is None:
            print("📊 Preparing dataset...")
            self.X_full, self.y_full, _ = self.prepare_full_data(feature_columns)

        X = self.X_full
        y = self.y_full

        print(f"📦 Total samples: {len(X)}")
        print(f"🎯 Classes: {np.unique(y)}")
        print(f"📊 Features: {self.feature_columns}")

        # STEP 1: Initialize DBNN architecture with full dataset and feature names
        self.model.initialize_with_full_data(X, y, self.feature_columns)

        # STEP 2: Select initial diverse training samples
        X_train, y_train, initial_indices = self._select_initial_training_samples(X, y)
        remaining_indices = [i for i in range(len(X)) if i not in initial_indices]

        print(f"📊 Initial training set: {len(X_train)} samples")
        print(f"📊 Remaining test set: {len(remaining_indices)} samples")

        # Initialize tracking variables - FIXED: Track best model state
        self.best_accuracy = 0.0
        self.best_training_indices = initial_indices.copy()
        self.best_round = 0
        self.best_model_state = None  # NEW: Store best model weights
        self.round_stats = []
        self.training_history = [initial_indices.copy()]
        acid_test_history = []
        patience_counter = 0

        max_rounds = self.adaptive_config['max_adaptive_rounds']
        patience = self.adaptive_config['patience']
        # REMOVED: min_improvement - any improvement is valuable
        enable_acid_test = self.adaptive_config.get('enable_acid_test', True)
        enable_visualization = self.adaptive_config.get('enable_visualization', False)

        print(f"\n🔄 Starting adaptive learning for up to {max_rounds} rounds...")
        print(f"📊 Stopping criteria: 100% accuracy OR patience {patience} rounds OR max rounds {max_rounds}")
        print(f"🔬 Acid Test: {'ENABLED' if enable_acid_test else 'DISABLED'}")
        print(f"🎨 Visualization: {'ENABLED' if enable_visualization else 'DISABLED'}")
        self.adaptive_start_time = datetime.now()

        for round_num in range(1, max_rounds + 1):
            self.adaptive_round = round_num

            print(f"\n🎯 Round {round_num}/{max_rounds}")
            print("-" * 40)

            # Train with current training data
            print("🎯 Training with current training data...")
            train_accuracy = self.model.train_with_data(X_train, y_train, reset_weights=True)

            if train_accuracy == 0.0:
                print("❌ Training failed, stopping...")
                break

            # Determine the accuracy metric to use for model selection
            if enable_acid_test:
                print("🧪 Running acid test on entire dataset...")
                try:
                    all_predictions = self.model.predict(X)
                    current_accuracy = accuracy_score(y, all_predictions)
                    accuracy_type = "acid test"
                    print(f"📊 Training accuracy: {train_accuracy:.2f}%")
                    print(f"📊 Acid test accuracy: {current_accuracy:.4f}")
                except Exception as e:
                    print(f"❌ Acid test failed: {e}")
                    # Fallback to training accuracy if acid test fails
                    current_accuracy = train_accuracy / 100.0
                    accuracy_type = "training (fallback)"
                    print(f"📊 Using training accuracy as fallback: {current_accuracy:.4f}")
            else:
                # If acid test disabled, use accuracy on remaining data
                if len(remaining_indices) > 0:
                    X_remaining = X[remaining_indices]
                    y_remaining = y[remaining_indices]
                    remaining_predictions = self.model.predict(X_remaining)
                    current_accuracy = accuracy_score(y_remaining, remaining_predictions)
                    accuracy_type = "remaining data"
                else:
                    current_accuracy = train_accuracy / 100.0
                    accuracy_type = "training"
                print(f"📊 Training accuracy: {train_accuracy:.2f}%")
                print(f"📊 {accuracy_type.title()} accuracy: {current_accuracy:.4f}")

            acid_test_history.append(current_accuracy)

            # Store round statistics
            round_stat = {
                'round': round_num,
                'training_size': len(X_train),
                'train_accuracy': train_accuracy / 100.0,
                'test_accuracy': current_accuracy,
                'new_samples': 0,
                'improvement': 0.0,
                'accuracy_type': accuracy_type
            }

            # STOPPING CRITERION 1: 100% accuracy
            if current_accuracy >= 0.9999:
                print("🎉 REACHED 100% ACCURACY! Stopping adaptive learning.")
                self._update_best_model(current_accuracy, initial_indices.copy(), round_num, self.model)
                round_stat['improvement'] = current_accuracy - self.best_accuracy
                self.round_stats.append(round_stat)
                self.training_history.append(initial_indices.copy())
                break

            # Check if we have any remaining samples to process
            if not remaining_indices:
                print("💤 No more samples to add to training set")
                break

            # Find samples to add from remaining data
            X_remaining = X[remaining_indices]
            y_remaining = y[remaining_indices]

            # Get predictions for remaining data
            remaining_predictions = self.model.predict(X_remaining)

            # Find misclassified samples
            misclassified_mask = remaining_predictions != y_remaining
            misclassified_indices = np.where(misclassified_mask)[0]

            if len(misclassified_indices) == 0:
                print("✅ No misclassified samples in remaining data!")
                print("🎉 PERFECT CLASSIFICATION ON REMAINING DATA! Stopping adaptive learning.")
                self._update_best_model(current_accuracy, initial_indices.copy(), round_num, self.model)
                round_stat['improvement'] = current_accuracy - self.best_accuracy
                self.round_stats.append(round_stat)
                self.training_history.append(initial_indices.copy())
                break

            print(f"📊 Found {len(misclassified_indices)} misclassified samples in remaining data")

            # Select samples to add (limit by configuration)
            max_samples_to_add = self.adaptive_config.get('max_margin_samples_per_class', 3) * len(np.unique(y))
            n_samples_to_add = min(len(misclassified_indices), max_samples_to_add)

            selected_indices = np.random.choice(misclassified_indices, n_samples_to_add, replace=False)
            samples_to_add_indices = [remaining_indices[i] for i in selected_indices]

            # Update training set
            initial_indices.extend(samples_to_add_indices)
            remaining_indices = [i for i in remaining_indices if i not in samples_to_add_indices]

            X_train = X[initial_indices]
            y_train = y[initial_indices]

            # Update training history
            self.training_history.append(initial_indices.copy())

            # Update round statistics with new samples
            round_stat['new_samples'] = len(samples_to_add_indices)

            print(f"📈 Added {len(samples_to_add_indices)} samples. New training set: {len(X_train)} samples")
            print(f"📊 Remaining set size: {len(remaining_indices)} samples")

            # CRITICAL FIX: Update best model and check for improvement - ANY improvement counts!
            improvement = current_accuracy - self.best_accuracy
            round_stat['improvement'] = improvement

            # DEBUG: Show detailed tracking information
            print(f"🔍 BEST TRACKING: Current={current_accuracy:.4f}, Best={self.best_accuracy:.4f}, Δ={improvement:.4f}")

            if current_accuracy > self.best_accuracy:
                # ANY improvement updates the best model and resets patience
                self._update_best_model(current_accuracy, initial_indices.copy(), round_num, self.model)
                patience_counter = 0  # Reset patience on ANY improvement
                print(f"🏆 NEW BEST {accuracy_type} accuracy: {current_accuracy:.4f} (+{improvement:.4f})")
            else:
                # No improvement - increment patience
                patience_counter += 1
                if current_accuracy == self.best_accuracy:
                    print(f"🔄 Same accuracy - Patience: {patience_counter}/{patience}")
                else:
                    print(f"📉 Worse accuracy: {current_accuracy:.4f} (best: {self.best_accuracy:.4f}) - Patience: {patience_counter}/{patience}")

            # Add round statistics
            self.round_stats.append(round_stat)

            # Create intermediate visualizations only if enabled and at strategic points
            if enable_visualization and self._should_create_visualizations(round_num):
                self._create_intermediate_visualizations(round_num)

            # STOPPING CRITERION: No improvement for patience rounds
            if patience_counter >= patience:
                print(f"🛑 PATIENCE EXCEEDED: No improvement for {patience} rounds")
                print(f"   Best {accuracy_type} accuracy: {self.best_accuracy:.4f} (round {self.best_round})")
                break

        # Finalize with best configuration
        print(f"\n🎉 Adaptive learning completed after {self.adaptive_round} rounds!")

        # Ensure we have valid best values - FIXED: Use the actual best we tracked
        if self.best_accuracy == 0.0 and acid_test_history:
            # Fallback: use the last accuracy if no best was set
            self.best_accuracy = acid_test_history[-1]
            self.best_training_indices = initial_indices.copy()
            self.best_round = self.adaptive_round
            print(f"⚠️  Using fallback best accuracy: {self.best_accuracy:.4f}")

        print(f"🏆 Best accuracy: {self.best_accuracy:.4f} (round {self.best_round})")
        print(f"📊 Final training set: {len(self.best_training_indices)} samples ({len(self.best_training_indices)/len(X)*100:.1f}% of total)")

        # Use best configuration for final model - RESTORE BEST MODEL STATE
        if self.best_model_state is not None:
            print("🔄 Restoring best model state...")
            self._restore_best_model_state()
        else:
            print("⚠️  No best model state saved - using current model")

        X_train_best = X[self.best_training_indices]
        y_train_best = y[self.best_training_indices]
        X_test_best = X[[i for i in range(len(X)) if i not in self.best_training_indices]]
        y_test_best = y[[i for i in range(len(X)) if i not in self.best_training_indices]]

        # Store test sets for evaluation
        self.X_test = X_test_best
        self.y_test = y_test_best

        # Train final model with best configuration (quick fine-tuning)
        print("🔧 Fine-tuning final model with best configuration...")
        final_train_accuracy = self.model.train_with_data(X_train_best, y_train_best, reset_weights=False)

        # Final verification
        final_predictions = self.model.predict(X)
        final_accuracy = accuracy_score(y, final_predictions)

        # Calculate total training time
        self.total_training_time = (datetime.now() - self.adaptive_start_time).total_seconds()

        print(f"📊 Final training accuracy: {final_train_accuracy:.2f}%")
        print(f"📊 Final acid test accuracy: {final_accuracy:.4f}")
        print(f"📈 Final training set size: {len(X_train_best)}")
        print(f"📊 Final test set size: {len(X_test_best)}")
        print(f"⏱️  Total training time: {self.total_training_time:.2f} seconds")

        # Final visualizations only if enabled
        if enable_visualization:
            self._finalize_adaptive_learning()
        else:
            print("🎨 Visualization disabled - skipping final visualizations")

        return X_train_best, y_train_best, X_test_best, y_test_best

    def _update_best_model(self, accuracy: float, training_indices: List[int], round_num: int, model):
        """Update the best model state - CRITICAL FIX"""
        self.best_accuracy = accuracy
        self.best_training_indices = training_indices.copy()
        self.best_round = round_num

        # Store the actual model weights for later restoration
        try:
            if hasattr(db, 'core') and hasattr(db.core, 'anti_wts'):
                self.best_model_state = {
                    'anti_wts': db.core.anti_wts.copy(),
                    'anti_net': db.core.anti_net.copy() if hasattr(db.core, 'anti_net') else None
                }
            else:
                self.best_model_state = None
        except Exception as e:
            print(f"⚠️ Could not save model state: {e}")
            self.best_model_state = None

    def _restore_best_model_state(self):
        """Restore the best model state"""
        if self.best_model_state is not None and hasattr(self.model, 'core'):
            try:
                self.model.core.anti_wts = self.best_model_state['anti_wts'].copy()
                if self.best_model_state['anti_net'] is not None and hasattr(self.model.core, 'anti_net'):
                    self.model.core.anti_net = self.best_model_state['anti_net'].copy()
                print("✅ Best model state restored")
            except Exception as e:
                print(f"⚠️ Could not restore model state: {e}")

    def _create_intermediate_visualizations(self, round_num):
        """Create intermediate visualizations including advanced 3D"""
        try:
            current_indices = self.training_history[-1]

            # Create comprehensive visualizations
            self.comprehensive_visualizer.plot_3d_networks(
                self.X_full, self.y_full, [current_indices],
                self.feature_columns
            )

            # Create advanced interactive 3D visualizations
            if self.adaptive_config.get('enable_advanced_3d', True):
                self.advanced_visualizer.create_advanced_3d_dashboard(
                    self.X_full, self.y_full, self.training_history,
                    self.feature_columns, round_num
                )

            print(f"🎨 Created advanced visualizations for round {round_num}")
        except Exception as e:
            print(f"⚠️ Advanced visualization failed: {e}")

    def _finalize_adaptive_learning(self):
        """Finalize adaptive learning with comprehensive outputs"""
        print("\n" + "="*60)
        print("🏁 FINALIZING ADAPTIVE LEARNING")
        print("="*60)

        # 1. Create comprehensive visualizations
        try:
            self.comprehensive_visualizer.create_comprehensive_visualizations(
                self, self.X_full, self.y_full,
                self.training_history, self.round_stats, self.feature_columns
            )
        except Exception as e:
            print(f"⚠️ Comprehensive visualization failed: {e}")

        # 2. Save model with automatic naming
        self._save_adaptive_model()

        # 3. Save configuration
        self._save_adaptive_configuration()

        # 4. Generate final report
        self._generate_final_report()

    def _save_adaptive_model(self):
        """Save adaptive model with automatic naming"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{self.dataset_name}_adbnn_{timestamp}.bin"
        model_path = self.models_dir / model_filename

        try:
            # Use the DBNN core's save capability - FIXED: use self.model instead of self.model
            success = self.model.core.save_model_auto(
                model_dir=str(self.models_dir),
                data_filename=f"{self.dataset_name}.csv",
                feature_columns=self.feature_columns,
                target_column=self.target_column
            )

            if success:
                print(f"💾 Adaptive model saved: {model_path}")

                # Also save adaptive learning metadata
                metadata = {
                    'dataset_name': self.dataset_name,
                    'adaptive_config': self.adaptive_config,
                    'best_accuracy': self.best_accuracy,
                    'best_round': self.best_round,
                    'final_training_size': len(self.best_training_indices),
                    'total_rounds': self.adaptive_round,
                    'feature_columns': self.feature_columns,
                    'target_column': self.target_column,
                    'save_timestamp': timestamp
                }

                metadata_path = self.models_dir / f"{self.dataset_name}_adbnn_{timestamp}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=4)

                print(f"💾 Adaptive metadata saved: {metadata_path}")
            else:
                print("❌ Failed to save adaptive model")

        except Exception as e:
            print(f"❌ Error saving adaptive model: {e}")

    def _save_adaptive_configuration(self):
        """Save adaptive learning configuration"""
        config_path = self.models_dir / f"{self.dataset_name}_adbnn_config.json"

        try:
            config_data = {
                'dataset_name': self.dataset_name,
                'adaptive_config': self.adaptive_config,
                'feature_columns': self.feature_columns,
                'target_column': self.target_column,
                'save_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=4)

            print(f"💾 Adaptive configuration saved: {config_path}")

        except Exception as e:
            print(f"❌ Error saving adaptive configuration: {e}")

    def _generate_final_report(self):
        """Generate final adaptive learning report"""
        report_path = self.comprehensive_visualizer.output_dir / "adaptive_learning_final_report.txt"

        try:
            with open(report_path, 'w') as f:
                f.write("="*60 + "\n")
                f.write("ADAPTIVE DBNN FINAL REPORT\n")
                f.write("="*60 + "\n\n")

                f.write(f"Dataset: {self.dataset_name}\n")
                f.write(f"Total Samples: {len(self.X_full)}\n")
                f.write(f"Features: {len(self.feature_columns)}\n")
                f.write(f"Classes: {np.unique(self.y_full)}\n\n")

                f.write("ADAPTIVE LEARNING RESULTS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Best Accuracy: {self.best_accuracy:.4f}\n")
                f.write(f"Best Round: {self.best_round}\n")
                f.write(f"Total Rounds: {self.adaptive_round}\n")
                f.write(f"Final Training Size: {len(self.best_training_indices)}\n")
                f.write(f"Training Percentage: {len(self.best_training_indices)/len(self.X_full)*100:.1f}%\n")
                f.write(f"Total Training Time: {self.total_training_time:.2f} seconds\n\n")

                f.write("FEATURE COLUMNS:\n")
                f.write("-" * 40 + "\n")
                for feature in self.feature_columns:
                    f.write(f"  {feature}\n")

                f.write(f"\nReport generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            print(f"📋 Final report saved: {report_path}")

        except Exception as e:
            print(f"❌ Error generating final report: {e}")

    def finalize_adaptive_learning(self):
        """Create final visualizations after adaptive learning"""
        # Final 3D visualization
        self.visualizer_3d.create_3d_training_network(
            self.X_full, self.y_full, self.best_training_indices,
            feature_names=self.feature_columns,
            round_num=None  # Final visualization
        )

        # Create animation of the entire process
        if len(self.training_history) > 1:
            self.visualizer_3d.create_adaptive_learning_animation(
                self.X_full, self.y_full, self.training_history
            )

    def _run_adaptive_rounds(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_full: np.ndarray, y_full: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run adaptive learning rounds"""
        max_rounds = self.adaptive_config['max_adaptive_rounds']
        patience = self.adaptive_config['patience']
        min_improvement = self.adaptive_config['min_improvement']

        self.best_accuracy = 0.0
        patience_counter = 0
        current_X_train = X_train.copy()
        current_y_train = y_train.copy()

        for round_num in range(max_rounds):
            self.adaptive_round = round_num
            print(f"\n🔄 Adaptive Round {round_num + 1}/{max_rounds}")

            # Train on current dataset
            round_accuracy = self.model.adaptive_train(current_X_train, current_y_train)

            print(f"📊 Round accuracy: {round_accuracy:.2f}%")

            # Check for convergence
            if round_accuracy >= self.adaptive_config['min_training_accuracy'] * 100:
                print(f"🎯 Target accuracy reached: {round_accuracy:.2f}%")
                break

            # Find samples to add
            new_samples_X, new_samples_y = self._find_samples_to_add(current_X_train, current_y_train, X_full, y_full)

            if len(new_samples_X) == 0:
                print("💡 No new informative samples found")
                patience_counter += 1
                if patience_counter >= patience:
                    print("🛑 Early stopping - no improvement")
                    break
                continue

            # Add samples to training set
            current_X_train = np.vstack([current_X_train, new_samples_X])
            current_y_train = np.hstack([current_y_train, new_samples_y])

            print(f"📈 Added {len(new_samples_X)} samples. New training set: {current_X_train.shape[0]} samples")

            # Check for improvement
            if round_accuracy > self.best_accuracy + min_improvement:
                self.best_accuracy = round_accuracy
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("🛑 Early stopping - no significant improvement")
                break

        return current_X_train, current_y_train

    def _find_samples_to_add(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_full: np.ndarray, y_full: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Find informative samples to add to training set"""
        print("🔍 Finding informative samples to add...")

        # Get predictions on full dataset
        predictions = self.model.predict(X_full)

        # Find misclassified samples
        misclassified_mask = predictions != y_full
        misclassified_indices = np.where(misclassified_mask)[0]

        print(f"📊 Misclassified samples: {len(misclassified_indices)}")

        if len(misclassified_indices) == 0:
            return np.array([]), np.array([])

        # Select samples to add (limit by configuration)
        max_samples_to_add = self.adaptive_config.get('max_samples_per_class_fallback', 2)
        n_samples_to_add = min(len(misclassified_indices), max_samples_to_add * len(np.unique(y_full)))

        selected_indices = np.random.choice(misclassified_indices, n_samples_to_add, replace=False)

        return X_full[selected_indices], y_full[selected_indices]

    def _create_test_set(self, X_train: np.ndarray, y_train: np.ndarray,
                       X_full: np.ndarray, y_full: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create test set from samples not in training set"""
        # Create a mask for samples not in training set
        train_mask = np.zeros(len(X_full), dtype=bool)

        # For each training sample, find its index in the full dataset
        for i in range(len(X_train)):
            # Find matching sample in full dataset
            for j in range(len(X_full)):
                if np.array_equal(X_train[i], X_full[j]) and y_train[i] == y_full[j]:
                    train_mask[j] = True
                    break

        test_mask = ~train_mask
        X_test = X_full[test_mask]
        y_test = y_full[test_mask]

        print(f"📊 Test set created: {len(X_test)} samples")

        return X_test, y_test

    def evaluate_adaptive_performance(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate adaptive learning performance"""
        if not hasattr(self.model.core, 'is_trained') or not self.model.core.is_trained:
            return {'accuracy': 0.0, 'error': 'Model not trained'}

        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions) * 100

        # Additional metrics
        cm = confusion_matrix(y_test, predictions)
        class_report = classification_report(y_test, predictions, output_dict=True)

        results = {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'adaptive_rounds': self.adaptive_round,
            'samples_added': self.adaptive_samples_added
        }

        print(f"📊 Adaptive Learning Results:")
        print(f"   Final Accuracy: {accuracy:.2f}%")
        print(f"   Adaptive Rounds: {self.adaptive_round}")
        print(f"   Samples Added: {self.adaptive_samples_added}")

        return results

def main():
    """Main function for adaptive DBNN"""
    print("🎯 Adaptive DBNN - Advanced Learning System")
    print("=" * 50)

    # Get available datasets
    available_configs = DatasetConfig.get_available_config_files()
    if not available_configs:
        print("❌ No configuration files found (.conf or .json)")
        return

    print("📁 Available datasets:")
    for i, config_info in enumerate(available_configs, 1):
        print(f"  {i:2d}. {config_info['file']} ({config_info['type']})")

    try:
        selection = int(input("\nSelect a dataset (1-{}): ".format(len(available_configs))))
        if selection < 1 or selection > len(available_configs):
            print("❌ Invalid selection")
            return

        selected_config = available_configs[selection - 1]
        dataset_name = selected_config['file'].replace('.conf', '').replace('.json', '')
        config_type = selected_config['type']

        print(f"🎯 Selected configuration: {dataset_name} ({config_type})")

        # Load configuration
        config = DatasetConfig.load_config(dataset_name)
        if not config:
            print(f"❌ Could not load configuration for {dataset_name}")
            return

        # Initialize adaptive DBNN
        adaptive_model = AdaptiveDBNN(dataset_name, config)

        # Get feature columns from config if available
        feature_columns = config.get('feature_columns', None)
        if feature_columns:
            print(f"📊 Using feature columns from config: {feature_columns}")

        # Run adaptive learning
        print("🚀 Starting adaptive learning...")
        X_train, y_train, X_test, y_test = adaptive_model.adaptive_learn(feature_columns=feature_columns)

        # Evaluate performance
        results = adaptive_model.evaluate_adaptive_performance(X_test, y_test)

        print("\n✅ Adaptive Learning Completed!")
        print(f"📊 Final Test Accuracy: {results['accuracy']:.2f}%")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("""
    ╔═════════════════════════════════════════════════════════════╗
    ║              🧠      DBNN CLASSIFIER                        ║
    ║         Difference Boosting Bayesian Neural Network         ║
    ║                 author: nsp@airis4d.com                     ║
    ║  Artificial Intelligence Research and Intelligent Systems   ║
    ║                 Thelliyoor 689544, India                    ║
    ║          INCREMENTAL LEARNING + FREEZE MECHANISM            ║
    ║                 implementation: deepseek                    ║
    ╚═════════════════════════════════════════════════════════════╝
    """)
    # Check for GUI flag
    if "--gui" in sys.argv or "-g" in sys.argv or len(sys.argv) == 1:
        launch_adaptive_gui()
    else:
        # Run command line version
        main()
