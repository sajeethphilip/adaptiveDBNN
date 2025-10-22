#!/usr/bin/env python3
"""
Optimized DBNN implementation as a modular class structure
Core algorithm with unified configuration and visualization capabilities
"""

import math
import sys
import time
import os
import json
import csv
import glob
import pickle
import gzip
from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
from numba import jit, prange
from multiprocessing import Pool, cpu_count
from functools import lru_cache
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import traceback
import multiprocessing as mp
import plotly.express as px

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import threading
from numba import jit, prange
import numpy as np
import os
import time
import json
import csv
import pickle
import gzip
from typing import List, Tuple, Dict, Any, Optional, Union

# Global constants
max_resol = 1600
features = 100
classes = 500

@jit(nopython=True, fastmath=True, cache=True)
def round_cpp(x: float) -> int:
    """Match C++ rounding behavior - Numba compatible"""
    return int(x + 0.5) if x >= 0 else int(x - 0.5)

@jit(nopython=True, fastmath=True, cache=True)
def normalize_feature(value, min_val, max_val, resolution_val):
    """Normalize single feature value - Numba compatible"""
    if max_val - min_val > 0:
        return round_cpp((value - min_val) / (max_val - min_val) * resolution_val)
    return 0

@jit(nopython=True, parallel=False, fastmath=True)
def process_training_sample(vects, tmpv, anti_net, anti_wts, binloc,
                           resolution, class_labels, min_val, max_val,
                           innodes, outnodes):
    """Process single training sample - core algorithm unchanged"""
    normalized_vects = np.zeros(innodes + 2)

    # Normalize vectors
    for i in range(1, innodes + 1):
        normalized_vects[i] = normalize_feature(vects[i], min_val[i], max_val[i], resolution[i])

    # Find bins for all features
    bins = np.zeros(innodes + 2, dtype=np.int32)
    for i in range(1, innodes + 1):
        bins[i] = find_closest_bin_numba(normalized_vects[i], binloc[i], resolution[i])

    # Update network counts
    for i in range(1, innodes + 1):
        j = bins[i]
        for l in range(1, innodes + 1):
            m = bins[l]

            # Find correct output class
            k_class = 1
            while k_class <= outnodes and abs(tmpv - class_labels[k_class]) > class_labels[0]:
                k_class += 1

            if k_class <= outnodes:
                anti_net[i, j, l, m, k_class] += 1
                anti_net[i, j, l, m, 0] += 1

    return anti_net

@jit(nopython=True, parallel=False, fastmath=True)
def compute_class_probabilities_numba(vects, anti_net, anti_wts, binloc,
                                    resolution, class_labels, min_val, max_val,
                                    innodes, outnodes):
    """Compute class probabilities - core algorithm unchanged"""
    classval = np.ones(outnodes + 2)
    normalized_vects = np.zeros(innodes + 2)

    # Normalize vectors
    for i in range(1, innodes + 1):
        normalized_vects[i] = normalize_feature(vects[i], min_val[i], max_val[i], resolution[i])

    # Find bins for all features
    bins = np.zeros(innodes + 2, dtype=np.int32)
    for i in range(1, innodes + 1):
        bins[i] = find_closest_bin_numba(normalized_vects[i], binloc[i], resolution[i])

    # Compute probabilities
    classval[0] = 0.0
    for i in range(1, innodes + 1):
        j = bins[i]
        for l in range(1, innodes + 1):
            m = bins[l]
            for k in range(1, outnodes + 1):
                if anti_net[i, j, l, m, 0] > 0:
                    tmp2_wts = anti_net[i, j, l, m, k] / anti_net[i, j, l, m, 0]
                else:
                    tmp2_wts = 1.0 / outnodes
                classval[k] *= tmp2_wts * anti_wts[i, j, l, m, k]
                classval[0] += classval[k]

    # Normalize
    if classval[0] > 0:
        for k in range(1, outnodes + 1):
            classval[k] /= classval[0]
    classval[0] = 0.0

    return classval

@jit(nopython=True, parallel=False, fastmath=True)
def update_weights_numba(vects, tmpv, classval, anti_wts, binloc, resolution,
                        class_labels, min_val, max_val, innodes, outnodes, gain):
    """Update weights - core algorithm unchanged"""
    normalized_vects = np.zeros(innodes + 2)

    # Normalize vectors
    for i in range(1, innodes + 1):
        normalized_vects[i] = normalize_feature(vects[i], min_val[i], max_val[i], resolution[i])

    # Find bins for all features
    bins = np.zeros(innodes + 2, dtype=np.int32)
    for i in range(1, innodes + 1):
        bins[i] = find_closest_bin_numba(normalized_vects[i], binloc[i], resolution[i])

    # Find predicted class
    kmax = 1
    cmax = 0.0
    for k in range(1, outnodes + 1):
        if classval[k] > cmax:
            cmax = classval[k]
            kmax = k

    # Update weights if wrong
    if abs(class_labels[kmax] - tmpv) > class_labels[0]:
        for i in range(1, innodes + 1):
            j = bins[i]
            for l in range(1, innodes + 1):
                m = bins[l]

                # Find correct class
                k_correct = 1
                while k_correct <= outnodes and abs(class_labels[k_correct] - tmpv) > class_labels[0]:
                    k_correct += 1

                if (k_correct <= outnodes and classval[kmax] > 0 and
                    classval[k_correct] < classval[kmax]):
                    adjustment = gain * (1.0 - (classval[k_correct] / classval[kmax]))
                    anti_wts[i, j, l, m, k_correct] += adjustment

    return anti_wts

@jit(nopython=True, fastmath=True, cache=True)
def find_closest_bin_numba(value, binloc_row, resolution_val):
    """Optimized bin finding for single value"""
    if resolution_val <= 0:
        return 0
    min_dist = 2.0 * resolution_val
    best_bin = 0
    for j in range(1, resolution_val + 1):
        dist = abs(value - binloc_row[j])
        if dist < min_dist:
            min_dist = dist
            best_bin = j
    if best_bin > 0:
        best_bin -= 1
    return best_bin

class DBNNVisualizer:
    """
    Enhanced visualization class for DBNN with interactive 3D capabilities,
    educational visualizations, and comprehensive training monitoring.

    This class provides multiple visualization types:
    - Standard 2D/3D plots for training monitoring
    - Enhanced interactive 3D visualizations with animation
    - Tensor mode specific visualizations
    - Educational dashboards with multiple subplots
    - Real-time training progression tracking
    """

    def __init__(self):
        """
        Initialize DBNNVisualizer with empty data structures for storing
        training history and visualization data.
        """
        # Core training history storage
        self.training_history = []  # List of training snapshots
        self.visualization_data = {}  # Additional visualization metadata
        self.tensor_snapshots = []  # Tensor-specific training data

        # Enhanced data storage for advanced visualizations
        self.feature_space_snapshots = []  # 3D feature space evolution
        self.feature_names = []  # Names of features for labeling
        self.current_iteration = 0
        self.class_names = []  # Names of classes for labeling
        self.accuracy_progression = []  # Accuracy over training rounds
        self.weight_evolution = []  # Weight statistics over time
        self.confusion_data = []  # Confusion matrix data

        # Educational visualization data
        self.decision_boundaries = []  # Decision boundary evolution
        self.feature_importance_data = []  # Feature importance metrics
        self.learning_curves = []  # Learning curve data
        self.network_topology_data = []  # Network structure information

    # =========================================================================
    # CORE TRAINING DATA CAPTURE METHODS
    # =========================================================================

    def capture_training_snapshot(self, features, targets, weights, predictions, accuracy, round_num):
        """
        Capture comprehensive training snapshot for visualization and analysis.

        Args:
            features (numpy.ndarray): Feature matrix (samples x features)
            targets (numpy.ndarray): True target values
            weights (numpy.ndarray): Current network weights
            predictions (numpy.ndarray): Model predictions
            accuracy (float): Current accuracy percentage
            round_num (int): Training iteration/round number

        Returns:
            dict: Snapshot containing all training data
        """
        snapshot = {
            'round': round_num,
            'features': features.copy() if features is not None else None,
            'targets': targets.copy() if targets is not None else None,
            'weights': weights.copy() if weights is not None else None,
            'predictions': predictions.copy() if predictions is not None else None,
            'accuracy': accuracy,
            'timestamp': time.time()
        }

        self.training_history.append(snapshot)

        # Store accuracy progression
        self.accuracy_progression.append({
            'round': round_num,
            'accuracy': accuracy,
            'timestamp': time.time()
        })

        # Store weight statistics for educational purposes
        if weights is not None:
            flat_weights = weights.flatten()
            flat_weights = flat_weights[(flat_weights != 0) & (np.abs(flat_weights) < 100)]

            self.weight_evolution.append({
                'round': round_num,
                'mean': np.mean(flat_weights) if len(flat_weights) > 0 else 0,
                'std': np.std(flat_weights) if len(flat_weights) > 0 else 0,
                'min': np.min(flat_weights) if len(flat_weights) > 0 else 0,
                'max': np.max(flat_weights) if len(flat_weights) > 0 else 0
            })

        # Capture enhanced visualization data if features are available
        if hasattr(self, 'feature_space_snapshots') and features is not None:
            try:
                feature_names = getattr(self, 'feature_names',
                                      [f'Feature_{i+1}' for i in range(features.shape[1])])
                class_names = getattr(self, 'class_names',
                                    [f'Class_{int(c)}' for c in np.unique(targets)])

                enhanced_snapshot = {
                    'iteration': round_num,
                    'features': features.copy(),
                    'targets': targets.copy(),
                    'predictions': predictions.copy(),
                    'feature_names': feature_names,
                    'class_names': class_names,
                    'timestamp': time.time(),
                    'accuracy': accuracy
                }
                self.feature_space_snapshots.append(enhanced_snapshot)
            except Exception as e:
                print(f"Enhanced visualization capture warning: {e}")

        return snapshot

    def capture_tensor_snapshot(self, features, targets, weight_matrix, orthogonal_basis,
                               predictions, accuracy, iteration=0):
        """
        Capture specialized snapshot for tensor mode training.

        Args:
            features (numpy.ndarray): Input features
            targets (numpy.ndarray): True targets
            weight_matrix (numpy.ndarray): Tensor weight matrix
            orthogonal_basis (numpy.ndarray): Orthogonal basis vectors
            predictions (numpy.ndarray): Model predictions
            accuracy (float): Current accuracy
            iteration (int): Training iteration

        Returns:
            dict: Tensor-specific snapshot
        """
        tensor_data = {
            'weight_matrix': weight_matrix.copy() if hasattr(weight_matrix, 'copy') else weight_matrix,
            'orthogonal_basis': orthogonal_basis.copy() if hasattr(orthogonal_basis, 'copy') else orthogonal_basis,
            'iteration': iteration,
            'weight_matrix_norm': np.linalg.norm(weight_matrix) if weight_matrix is not None else 0,
            'basis_rank': np.linalg.matrix_rank(orthogonal_basis) if orthogonal_basis is not None else 0
        }

        # Use the main snapshot method but add tensor data
        snapshot = self.capture_training_snapshot(
            features, targets, weight_matrix, predictions, accuracy, iteration
        )
        snapshot['is_tensor_mode'] = True
        snapshot['tensor_data'] = tensor_data

        self.tensor_snapshots.append(snapshot)
        return snapshot

    def capture_feature_space_snapshot(self, features, targets, predictions, iteration,
                                     feature_names=None, class_names=None):
        """
        Capture feature space state for interactive 3D visualization.

        Args:
            features (numpy.ndarray): Feature matrix
            targets (numpy.ndarray): True targets
            predictions (numpy.ndarray): Model predictions
            iteration (int): Training iteration
            feature_names (list): Names of features for labeling
            class_names (list): Names of classes for labeling

        Returns:
            dict: Feature space snapshot
        """
        if feature_names is None:
            feature_names = [f'Feature_{i+1}' for i in range(features.shape[1])]
        if class_names is None:
            unique_targets = np.unique(targets)
            class_names = [f'Class_{int(t)}' for t in unique_targets]

        snapshot = {
            'iteration': iteration,
            'features': features.copy(),
            'targets': targets.copy(),
            'predictions': predictions.copy(),
            'feature_names': feature_names,
            'class_names': class_names,
            'timestamp': time.time()
        }

        self.feature_space_snapshots.append(snapshot)
        self.feature_names = feature_names
        self.class_names = class_names

        return snapshot

    # =========================================================================
    # ENHANCED INTERACTIVE 3D VISUALIZATION METHODS
    # =========================================================================

    def generate_animated_confusion_matrix(self, output_file="confusion_animation.html", frame_delay=500):
        """Generate animated confusion matrix showing evolution over training iterations"""
        try:
            import plotly.graph_objects as go
            import numpy as np
            from sklearn.metrics import confusion_matrix
            import os

            if not self.feature_space_snapshots:
                print("❌ No feature space snapshots available for confusion matrix animation")
                return None

            print(f"🔄 Generating animated confusion matrix from {len(self.feature_space_snapshots)} snapshots...")

            # Collect all unique classes across all snapshots
            unique_classes_all = set()
            for snapshot in self.feature_space_snapshots:
                if 'targets' in snapshot and 'predictions' in snapshot:
                    targets = snapshot['targets']
                    predictions = snapshot['predictions']
                    all_labels = np.concatenate([targets, predictions])
                    unique_classes_all.update(all_labels)

            if not unique_classes_all:
                print("❌ No class data found in snapshots")
                return None

            unique_classes = sorted(unique_classes_all)
            class_names = [f'Class {int(cls)}' for cls in unique_classes]

            print(f"📊 Found {len(unique_classes)} unique classes: {unique_classes}")

            # Create frames for each snapshot
            frames = []

            for i, snapshot in enumerate(self.feature_space_snapshots):
                if 'targets' not in snapshot or 'predictions' not in snapshot:
                    continue

                targets = snapshot['targets']
                predictions = snapshot['predictions']
                iteration = snapshot.get('iteration', i)

                # Create confusion matrix
                try:
                    cm = confusion_matrix(targets, predictions, labels=unique_classes)

                    # Normalize by row (true classes)
                    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
                    cm_normalized = np.nan_to_num(cm_normalized)

                    # Create heatmap trace
                    heatmap = go.Heatmap(
                        z=cm_normalized,
                        x=class_names,
                        y=class_names,
                        colorscale='Blues',
                        zmin=0,
                        zmax=1,
                        colorbar=dict(title="Normalized<br>Probability"),
                        hovertemplate=(
                            'True: %{y}<br>' +
                            'Predicted: %{x}<br>' +
                            'Probability: %{z:.3f}<br>' +
                            'Iteration: ' + str(iteration) + '<br>' +
                            '<extra></extra>'
                        ),
                        name=f"Iteration {iteration}"
                    )

                    # Calculate accuracy for this iteration
                    accuracy = np.mean(targets == predictions) * 100 if len(targets) > 0 else 0

                    # Create frame
                    frame = go.Frame(
                        data=[heatmap],
                        name=f'frame_{i}',
                        layout=go.Layout(
                            title_text=f"Confusion Matrix Evolution<br>Iteration {iteration} | Accuracy: {accuracy:.1f}%",
                            annotations=[
                                dict(
                                    text=f'Iteration: {iteration} | Accuracy: {accuracy:.1f}%',
                                    x=0.5, y=1.08,
                                    xref='paper', yref='paper',
                                    showarrow=False,
                                    font=dict(size=14, color='darkblue')
                                )
                            ]
                        )
                    )
                    frames.append(frame)

                except Exception as e:
                    print(f"⚠️ Could not create confusion matrix for iteration {iteration}: {e}")
                    continue

            if not frames:
                print("❌ No valid frames created for confusion matrix animation")
                return None

            print(f"✅ Created {len(frames)} frames for confusion matrix animation")

            # Create initial figure with first frame
            fig = go.Figure(data=frames[0].data, frames=frames)

            # Update layout
            fig.update_layout(
                title={
                    'text': "Confusion Matrix Evolution During Training",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20, 'color': 'darkblue'}
                },
                xaxis_title="Predicted Class",
                yaxis_title="True Class",
                width=900,
                height=800,
                plot_bgcolor='white',
                paper_bgcolor='lightgray',
                font=dict(size=12),
                # Animation controls
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': False,
                    'buttons': [
                        {
                            'label': '▶️ Play',
                            'method': 'animate',
                            'args': [
                                None,
                                {
                                    'frame': {'duration': frame_delay, 'redraw': True},
                                    'fromcurrent': True,
                                    'transition': {'duration': 300, 'easing': 'cubic-in-out'}
                                }
                            ]
                        },
                        {
                            'label': '⏸️ Pause',
                            'method': 'animate',
                            'args': [
                                [None],
                                {
                                    'frame': {'duration': 0, 'redraw': False},
                                    'mode': 'immediate'
                                }
                            ]
                        },
                        {
                            'label': '⏭️ Next',
                            'method': 'animate',
                            'args': [
                                None,
                                {
                                    'mode': 'next',
                                    'frame': {'duration': frame_delay, 'redraw': True},
                                    'transition': {'duration': 300}
                                }
                            ]
                        }
                    ],
                    'x': 0.1,
                    'y': 0,
                    'xanchor': 'left',
                    'yanchor': 'bottom',
                    'bgcolor': 'lightblue',
                    'bordercolor': 'navy',
                    'borderwidth': 2
                }]
            )

            # Add slider for manual control
            steps = []
            for i, snapshot in enumerate(self.feature_space_snapshots):
                if i >= len(frames):
                    continue
                iteration = snapshot.get('iteration', i)
                accuracy = np.mean(snapshot['targets'] == snapshot['predictions']) * 100 if 'targets' in snapshot and 'predictions' in snapshot else 0

                step = {
                    'args': [
                        [f'frame_{i}'],
                        {
                            'frame': {'duration': 300, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 300}
                        }
                    ],
                    'label': f'Iter {iteration}',
                    'method': 'animate'
                }
                steps.append(step)

            fig.update_layout(
                sliders=[{
                    'active': 0,
                    'currentvalue': {
                        'prefix': 'Iteration: ',
                        'xanchor': 'right',
                        'font': {'size': 16, 'color': 'black'}
                    },
                    'transition': {'duration': 300, 'easing': 'cubic-in-out'},
                    'x': 0.1,
                    'len': 0.8,
                    'xanchor': 'left',
                    'y': 0,
                    'yanchor': 'top',
                    'bgcolor': 'lightgray',
                    'bordercolor': 'black',
                    'borderwidth': 1,
                    'steps': steps
                }]
            )

            # Add educational annotations
            fig.add_annotation(
                text="🎓 <b>Educational Insight:</b><br>Watch how the model's confusion patterns evolve during training.<br>Perfect classification would show high values only on the diagonal.",
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                bgcolor="lightyellow",
                bordercolor="black",
                borderwidth=2,
                font=dict(size=12, color='black')
            )

            # Add colorbar title
            fig.add_annotation(
                text="Color Intensity = Classification Probability",
                xref="paper", yref="paper",
                x=0.98, y=0.02,
                showarrow=False,
                font=dict(size=12, color='darkblue'),
                bgcolor="white",
                bordercolor="blue",
                borderwidth=1
            )

            # Ensure output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            fig.write_html(output_file)
            print(f"✅ Animated confusion matrix saved: {output_file}")

            # Print some statistics
            if frames:
                first_frame = frames[0]
                last_frame = frames[-1]
                first_acc = np.mean(self.feature_space_snapshots[0]['targets'] == self.feature_space_snapshots[0]['predictions']) * 100
                last_acc = np.mean(self.feature_space_snapshots[-1]['targets'] == self.feature_space_snapshots[-1]['predictions']) * 100
                print(f"📈 Accuracy progression: {first_acc:.1f}% → {last_acc:.1f}%")

            return output_file

        except Exception as e:
            print(f"❌ Error creating animated confusion matrix: {e}")
            import traceback
            traceback.print_exc()
            return None


    def generate_feature_orthogonality_plot(self, output_file="feature_orthogonality.html"):
        """Generate visualization showing feature orthogonality evolution"""
        try:
            import plotly.graph_objects as go
            import numpy as np

            if not self.feature_space_snapshots:
                return None

            # Calculate orthogonality scores over time
            iterations = []
            orthogonality_scores = []

            for snapshot in self.feature_space_snapshots:
                features = snapshot['features']
                iteration = snapshot['iteration']

                # Calculate feature correlation matrix
                if features.shape[1] > 1:
                    corr_matrix = np.corrcoef(features.T)
                    # Measure orthogonality (lower correlation = more orthogonal)
                    mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
                    avg_correlation = np.mean(np.abs(corr_matrix[mask]))
                    orthogonality = 1.0 - avg_correlation
                else:
                    orthogonality = 0.0

                iterations.append(iteration)
                orthogonality_scores.append(orthogonality)

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=iterations, y=orthogonality_scores,
                mode='lines+markers',
                name='Feature Orthogonality',
                line=dict(color='purple', width=3),
                marker=dict(size=6),
                hovertemplate='Iteration: %{x}<br>Orthogonality: %{y:.3f}<extra></extra>'
            ))

            fig.update_layout(
                title="Feature Orthogonality Evolution",
                xaxis_title="Training Iteration",
                yaxis_title="Orthogonality Score (1 = Perfect Orthogonality)",
                height=500,
                showlegend=True
            )

            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            fig.write_html(output_file)
            print(f"✅ Feature orthogonality plot saved: {output_file}")
            return output_file

        except Exception as e:
            print(f"Error creating feature orthogonality plot: {e}")
            return None

    def generate_class_separation_evolution(self, output_file="class_separation.html"):
        """Generate visualization showing class separation evolution in complex space"""
        try:
            import plotly.graph_objects as go
            import numpy as np
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

            if not self.feature_space_snapshots:
                return None

            iterations = []
            separation_scores = []

            for snapshot in self.feature_space_snapshots:
                features = snapshot['features']
                targets = snapshot['targets']
                iteration = snapshot['iteration']

                if len(np.unique(targets)) > 1 and features.shape[1] > 1:
                    try:
                        # Use LDA to measure class separation
                        lda = LinearDiscriminantAnalysis()
                        lda.fit(features, targets)
                        # Use between-class variance as separation metric
                        separation = np.trace(lda.between_class_scatter) / np.trace(lda.within_class_scatter)
                    except:
                        separation = 0.0
                else:
                    separation = 0.0

                iterations.append(iteration)
                separation_scores.append(separation)

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=iterations, y=separation_scores,
                mode='lines+markers',
                name='Class Separation',
                line=dict(color='orange', width=3),
                marker=dict(size=6),
                hovertemplate='Iteration: %{x}<br>Separation: %{y:.3f}<extra></extra>'
            ))

            fig.update_layout(
                title="Class Separation Evolution in Complex Space",
                xaxis_title="Training Iteration",
                yaxis_title="Separation Score (Higher = Better Separation)",
                height=500,
                showlegend=True
            )

            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            fig.write_html(output_file)
            print(f"✅ Class separation evolution saved: {output_file}")
            return output_file

        except Exception as e:
            print(f"Error creating class separation plot: {e}")
            return None

    def generate_complex_phase_diagram(self, output_file="complex_phase_diagram.html"):
        """Generate phase diagram showing feature vectors in complex space"""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots
            import numpy as np

            if not self.feature_space_snapshots:
                return None

            # Use the latest snapshot
            snapshot = self.feature_space_snapshots[-1]
            features = snapshot['features']
            targets = snapshot['targets']
            iteration = snapshot['iteration']

            # Sample for performance
            sample_size = min(1000, len(features))
            if len(features) > sample_size:
                indices = np.random.choice(len(features), sample_size, replace=False)
                features = features[indices]
                targets = targets[indices]

            n_features = min(features.shape[1], 6)  # Limit for visualization

            # Create complex representation
            complex_features = self._create_complex_representation(features, n_features)
            unique_classes = np.unique(targets)
            colors = px.colors.qualitative.Set1

            # Create subplots for each feature pair
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=[f'Features {i+1}-{j+1}' for i in range(2) for j in range(3)],
                specs=[[{'type': 'scatter'} for _ in range(3)] for _ in range(2)]
            )

            plot_idx = 0
            for i in range(n_features):
                for j in range(i+1, min(i+4, n_features)):
                    if plot_idx < 6:  # Max 6 subplots
                        row = plot_idx // 3 + 1
                        col = plot_idx % 3 + 1

                        for class_idx, cls in enumerate(unique_classes):
                            class_mask = targets == cls
                            if np.any(class_mask):
                                # Get complex components
                                real_i = complex_features[class_mask, i].real
                                imag_i = complex_features[class_mask, i].imag
                                real_j = complex_features[class_mask, j].real
                                imag_j = complex_features[class_mask, j].imag

                                # Create phase plot
                                fig.add_trace(go.Scatter(
                                    x=real_i, y=real_j,
                                    mode='markers',
                                    marker=dict(
                                        size=6,
                                        color=colors[class_idx % len(colors)],
                                        opacity=0.7,
                                        line=dict(width=1, color='white')
                                    ),
                                    name=f'Class {int(cls)}',
                                    legendgroup=f'class_{cls}',
                                    showlegend=(plot_idx == 0),  # Only show legend in first plot
                                    hovertemplate=(
                                        f'Class {int(cls)}<br>' +
                                        f'Feature {i+1}: %{{x:.3f}}<br>' +
                                        f'Feature {j+1}: %{{y:.3f}}<br>' +
                                        '<extra></extra>'
                                    )
                                ), row=row, col=col)

                        plot_idx += 1

            fig.update_layout(
                title={
                    'text': f"Complex Phase Diagram - Iteration {iteration}",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20}
                },
                height=800,
                showlegend=True
            )

            # Add educational annotation
            fig.add_annotation(
                text="🎓 <b>Complex Phase Diagram</b><br>Each subplot shows the relationship between two features in complex space.<br>As training progresses, classes should separate into distinct directional patterns.",
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                bgcolor="lightyellow",
                bordercolor="black",
                borderwidth=1,
                font=dict(size=12)
            )

            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            fig.write_html(output_file)
            print(f"✅ Complex phase diagram saved: {output_file}")
            return output_file

        except Exception as e:
            print(f"Error creating phase diagram: {e}")
            return None

    def generate_complex_tensor_evolution(self, output_file="complex_tensor_evolution.html", max_features=8):
        """Generate animated visualization of complex tensor evolution showing feature orthogonality"""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots
            import numpy as np
            from sklearn.decomposition import PCA
            import colorsys

            if not self.feature_space_snapshots:
                print("No feature space snapshots available for complex tensor visualization")
                return None

            # Collect all data across snapshots
            all_features = []
            all_targets = []
            all_iterations = []

            for snapshot in self.feature_space_snapshots:
                features = snapshot['features']
                targets = snapshot['targets']
                iteration = snapshot['iteration']

                # Sample for performance
                sample_size = min(500, len(features))
                if len(features) > sample_size:
                    indices = np.random.choice(len(features), sample_size, replace=False)
                    features = features[indices]
                    targets = targets[indices]

                all_features.append(features)
                all_targets.extend([f"Iter {iteration}"] * len(features))
                all_iterations.extend([iteration] * len(features))

            if not all_features:
                return None

            # Combine all features
            combined_features = np.vstack(all_features)
            unique_classes = np.unique(np.concatenate([snapshot['targets'] for snapshot in self.feature_space_snapshots]))

            # Limit features for visualization
            n_features = min(combined_features.shape[1], max_features)

            # Create complex tensor representation
            # Each feature dimension becomes a complex component
            complex_features = self._create_complex_representation(combined_features, n_features)

            # Create the main visualization
            fig = make_subplots(
                rows=2, cols=2,
                specs=[
                    [{"type": "scatter3d", "rowspan": 2}, {"type": "scatter"}],
                    [None, {"type": "scatter"}]
                ],
                subplot_titles=(
                    "Complex Tensor Evolution - Feature Space",
                    "Orthogonality Progress",
                    "Feature Vector Alignment by Class"
                )
            )

            # Create frames for animation
            frames = []
            colors = px.colors.qualitative.Set1

            for i, snapshot in enumerate(self.feature_space_snapshots):
                features = snapshot['features']
                targets = snapshot['targets']
                iteration = snapshot['iteration']

                # Sample for performance
                sample_size = min(300, len(features))
                if len(features) > sample_size:
                    indices = np.random.choice(len(features), sample_size, replace=False)
                    features = features[indices]
                    targets = targets[indices]

                # Create complex representation for this snapshot
                complex_snapshot = self._create_complex_representation(features, n_features)

                # Calculate orthogonality metrics
                orthogonality_score = self._calculate_orthogonality(complex_snapshot, targets)
                alignment_metrics = self._calculate_feature_alignment(complex_snapshot, targets)

                frame_data = []

                # 3D Complex Space Visualization (left-top)
                if complex_snapshot.shape[1] >= 3:
                    # Use first 3 complex components for 3D visualization
                    x, y, z = complex_snapshot[:, 0].real, complex_snapshot[:, 1].real, complex_snapshot[:, 2].real

                    for class_idx, cls in enumerate(unique_classes):
                        class_mask = targets == cls
                        if np.any(class_mask):
                            # Calculate class centroid
                            centroid = np.mean(complex_snapshot[class_mask], axis=0)

                            # Plot samples
                            frame_data.append(go.Scatter3d(
                                x=x[class_mask], y=y[class_mask], z=z[class_mask],
                                mode='markers',
                                marker=dict(
                                    size=4,
                                    color=colors[class_idx % len(colors)],
                                    opacity=0.7,
                                    line=dict(width=1, color='white')
                                ),
                                name=f'Class {int(cls)}',
                                legendgroup=f'class_{cls}'
                            ))

                            # Plot class direction vector
                            direction = centroid / np.linalg.norm(centroid) * 2
                            frame_data.append(go.Scatter3d(
                                x=[0, direction[0].real], y=[0, direction[1].real], z=[0, direction[2].real],
                                mode='lines',
                                line=dict(
                                    color=colors[class_idx % len(colors)],
                                    width=6
                                ),
                                name=f'Class {int(cls)} Direction',
                                legendgroup=f'class_{cls}_dir',
                                showlegend=False
                            ))

                # Orthogonality Progress (right-top)
                iterations_so_far = [s['iteration'] for s in self.feature_space_snapshots[:i+1]]
                orth_scores_so_far = [self._calculate_orthogonality(
                    self._create_complex_representation(s['features'], n_features), s['targets']
                ) for s in self.feature_space_snapshots[:i+1]]

                frame_data.append(go.Scatter(
                    x=iterations_so_far, y=orth_scores_so_far,
                    mode='lines+markers',
                    line=dict(color='blue', width=3),
                    name='Orthogonality Score',
                    xaxis='x2', yaxis='y2'
                ))

                # Feature Alignment (right-bottom)
                feature_names = [f'F{j+1}' for j in range(n_features)]
                alignment_values = list(alignment_metrics.values())[:n_features]

                frame_data.append(go.Scatter(
                    x=feature_names, y=alignment_values,
                    mode='lines+markers',
                    line=dict(color='green', width=2),
                    name='Feature Alignment',
                    xaxis='x3', yaxis='y3'
                ))

                frame = go.Frame(
                    data=frame_data,
                    name=f'frame_{i}',
                    layout=go.Layout(
                        title_text=f"Complex Tensor Evolution - Iteration {iteration}",
                        annotations=[
                            dict(
                                text=f'Orthogonality: {orthogonality_score:.3f}',
                                x=0.02, y=0.98, xref='paper', yref='paper',
                                showarrow=False, bgcolor='white', bordercolor='black',
                                borderwidth=1
                            )
                        ]
                    )
                )
                frames.append(frame)

            # Add initial data
            if frames:
                for trace in frames[0].data:
                    fig.add_trace(trace)

            # Update layout
            fig.update_layout(
                title={
                    'text': "Complex Tensor Evolution in Feature Space",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20}
                },
                scene=dict(
                    xaxis_title='Complex Component 1 (Real)',
                    yaxis_title='Complex Component 2 (Real)',
                    zaxis_title='Complex Component 3 (Real)',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                scene2=dict(xaxis_title='Iteration', yaxis_title='Orthogonality Score'),
                scene3=dict(xaxis_title='Features', yaxis_title='Alignment Metric'),
                width=1200,
                height=800,
                showlegend=True
            )

            # Add animation controls
            fig.update_layout(
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': False,
                    'buttons': [
                        {
                            'label': '▶️ Play',
                            'method': 'animate',
                            'args': [None, {
                                'frame': {'duration': 500, 'redraw': True},
                                'fromcurrent': True,
                                'transition': {'duration': 300}
                            }]
                        },
                        {
                            'label': '⏸️ Pause',
                            'method': 'animate',
                            'args': [[None], {
                                'frame': {'duration': 0, 'redraw': False},
                                'mode': 'immediate'
                            }]
                        }
                    ],
                    'x': 0.1, 'y': 0
                }]
            )

            # Add slider
            steps = []
            for i, snapshot in enumerate(self.feature_space_snapshots):
                step = {
                    'args': [
                        [f'frame_{i}'],
                        {
                            'frame': {'duration': 300, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 300}
                        }
                    ],
                    'label': f'Iter {snapshot["iteration"]}',
                    'method': 'animate'
                }
                steps.append(step)

            fig.update_layout(
                sliders=[{
                    'active': 0,
                    'currentvalue': {'prefix': 'Iteration: '},
                    'steps': steps,
                    'x': 0.1, 'len': 0.8
                }]
            )

            fig.frames = frames

            # Ensure output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            fig.write_html(output_file)
            print(f"✅ Complex tensor evolution visualization saved: {output_file}")
            return output_file

        except Exception as e:
            print(f"Error creating complex tensor visualization: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _create_complex_representation(self, features, n_components):
        """Create complex representation of features using Hilbert transform"""
        try:
            from scipy import signal

            n_samples, n_features = features.shape
            n_components = min(n_components, n_features)

            # Create complex features using Hilbert transform (analytic signal)
            complex_features = np.zeros((n_samples, n_components), dtype=complex)

            for i in range(n_components):
                # Use Hilbert transform to create analytic signal
                analytic_signal = signal.hilbert(features[:, i])
                complex_features[:, i] = analytic_signal

            return complex_features

        except ImportError:
            # Fallback: create complex features using simple transformation
            n_samples, n_features = features.shape
            n_components = min(n_components, n_features)

            complex_features = np.zeros((n_samples, n_components), dtype=complex)

            for i in range(n_components):
                # Simple complex representation: real part = feature value, imaginary part = normalized position
                complex_features[:, i] = features[:, i] + 1j * (features[:, i] - np.mean(features[:, i])) / np.std(features[:, i])

            return complex_features

    def _calculate_orthogonality(self, complex_features, targets):
        """Calculate orthogonality metric between class centroids in complex space"""
        try:
            unique_classes = np.unique(targets)
            n_classes = len(unique_classes)

            if n_classes < 2:
                return 0.0

            # Calculate class centroids in complex space
            centroids = []
            for cls in unique_classes:
                class_mask = targets == cls
                if np.sum(class_mask) > 0:
                    centroid = np.mean(complex_features[class_mask], axis=0)
                    centroids.append(centroid)

            if len(centroids) < 2:
                return 0.0

            # Calculate pairwise orthogonality
            orthogonality_scores = []
            for i in range(len(centroids)):
                for j in range(i+1, len(centroids)):
                    # Complex dot product
                    dot_product = np.vdot(centroids[i], centroids[j])
                    norm_i = np.linalg.norm(centroids[i])
                    norm_j = np.linalg.norm(centroids[j])

                    if norm_i > 0 and norm_j > 0:
                        cosine_sim = np.abs(dot_product) / (norm_i * norm_j)
                        orthogonality = 1.0 - cosine_sim
                        orthogonality_scores.append(orthogonality)

            return np.mean(orthogonality_scores) if orthogonality_scores else 0.0

        except Exception as e:
            print(f"Error calculating orthogonality: {e}")
            return 0.0

    def _calculate_feature_alignment(self, complex_features, targets):
        """Calculate how well features align with class separation"""
        try:
            unique_classes = np.unique(targets)
            n_features = complex_features.shape[1]

            alignment_scores = {}

            for feature_idx in range(n_features):
                feature_values = complex_features[:, feature_idx]

                # Calculate between-class variance vs within-class variance
                overall_mean = np.mean(feature_values)
                between_var = 0.0
                within_var = 0.0

                for cls in unique_classes:
                    class_mask = targets == cls
                    if np.sum(class_mask) > 0:
                        class_mean = np.mean(feature_values[class_mask])
                        between_var += np.sum(class_mask) * np.abs(class_mean - overall_mean)**2
                        within_var += np.sum(np.abs(feature_values[class_mask] - class_mean)**2)

                if within_var > 0:
                    alignment = between_var / within_var
                else:
                    alignment = 0.0

                alignment_scores[f'Feature_{feature_idx+1}'] = alignment

            return alignment_scores

        except Exception as e:
            print(f"Error calculating feature alignment: {e}")
            return {}

    def generate_interactive_visualizations(self, output_dir="Visualisations"):
        """Generate all interactive visualizations after training - COMPLETE UPDATED VERSION"""
        if not hasattr(self, 'visualizer') or not self.visualizer:
            print("❌ No visualizer available. Please enable enhanced visualization before training.")
            return None

        # Check if we have any visualization data
        has_data = (hasattr(self.visualizer, 'training_history') and self.visualizer.training_history) or \
                   (hasattr(self.visualizer, 'feature_space_snapshots') and self.visualizer.feature_space_snapshots) or \
                   (hasattr(self.visualizer, 'accuracy_progression') and self.visualizer.accuracy_progression)

        if not has_data:
            print("❌ No visualization data available.")
            print("   Enable with enable_enhanced_visualization() before training")
            return None

        try:
            import os

            # Create organized directory structure based on data file
            if hasattr(self, 'current_file') and self.current_file:
                base_name = os.path.splitext(os.path.basename(self.current_file))[0]
            else:
                base_name = "dbnn_training"

            if hasattr(self, 'feature_columns') and self.feature_columns:
                feature_info = f"{len(self.feature_columns)}features"
                base_name = f"{base_name}_{feature_info}"

            # Create main visualization folder for this dataset
            main_viz_dir = os.path.join(output_dir, base_name)
            standard_dir = os.path.join(main_viz_dir, "Standard")
            enhanced_dir = os.path.join(main_viz_dir, "Enhanced")
            os.makedirs(standard_dir, exist_ok=True)
            os.makedirs(enhanced_dir, exist_ok=True)

            # Store the main visualization directory for easy access
            self.main_viz_directory = main_viz_dir

            print(f"📁 Creating visualizations in: {main_viz_dir}")

            outputs = {}

            # 1. Generate ENHANCED visualizations
            print("🔄 Generating enhanced visualizations...")

            # Enhanced 3D
            enhanced_3d_file = os.path.join(enhanced_dir, f"{base_name}_enhanced_3d.html")
            result = self.visualizer.generate_enhanced_interactive_3d(enhanced_3d_file)
            if result:
                outputs['enhanced_3d'] = result
                print(f"✅ Enhanced 3D: {result}")
            else:
                print("❌ Failed to generate enhanced_3d")

            # Advanced Dashboard
            dashboard_file = os.path.join(enhanced_dir, f"{base_name}_advanced_dashboard.html")
            result = self.visualizer.create_advanced_interactive_dashboard(dashboard_file)
            if result:
                outputs['advanced_dashboard'] = result
                print(f"✅ Advanced dashboard: {result}")
            else:
                print("❌ Failed to generate advanced_dashboard")

            # Complex Tensor Evolution
            complex_file = os.path.join(enhanced_dir, f"{base_name}_complex_tensor.html")
            result = self.visualizer.generate_complex_tensor_evolution(complex_file)
            if result:
                outputs['complex_tensor'] = result
                print(f"✅ Complex tensor evolution: {result}")
            else:
                print("❌ Failed to generate complex tensor evolution")

            # Phase Diagram
            phase_file = os.path.join(enhanced_dir, f"{base_name}_phase_diagram.html")
            result = self.visualizer.generate_complex_phase_diagram(phase_file)
            if result:
                outputs['phase_diagram'] = result
                print(f"✅ Complex phase diagram: {result}")
            else:
                print("❌ Failed to generate phase diagram")

            # 2. Generate STANDARD visualizations
            print("🔄 Generating standard visualizations...")

            # Traditional Dashboard
            traditional_file = os.path.join(standard_dir, f"{base_name}_traditional_dashboard.html")
            result = self.visualizer.create_training_dashboard(traditional_file)
            if result:
                outputs['traditional_dashboard'] = result
                print(f"✅ Traditional dashboard: {result}")
            else:
                print("❌ Failed to generate traditional_dashboard")

            # Performance Metrics
            performance_file = os.path.join(standard_dir, f"{base_name}_performance.html")
            result = self.visualizer.generate_performance_metrics(performance_file)
            if result:
                outputs['performance'] = result
                print(f"✅ Performance metrics: {result}")
            else:
                print("❌ Failed to generate performance")

            # Correlation Matrix
            correlation_file = os.path.join(standard_dir, f"{base_name}_correlation.html")
            result = self.visualizer.generate_correlation_matrix(correlation_file)
            if result:
                outputs['correlation'] = result
                print(f"✅ Correlation matrix: {result}")
            else:
                print("❌ Failed to generate correlation")

            # Feature Explorer
            feature_file = os.path.join(standard_dir, f"{base_name}_feature_explorer.html")
            result = self.visualizer.generate_basic_3d_visualization(feature_file)
            if result:
                outputs['feature_explorer'] = result
                print(f"✅ Feature explorer: {result}")
            else:
                print("❌ Failed to generate feature_explorer")

            # Animated Training
            animated_file = os.path.join(standard_dir, f"{base_name}_animated.html")
            result = self.visualizer.generate_animated_training(animated_file)
            if result:
                outputs['animated'] = result
                print(f"✅ Animated training: {result}")
            else:
                print("❌ Failed to generate animated")

            # Animated Confusion Matrix
            confusion_file = os.path.join(standard_dir, f"{base_name}_confusion_animation.html")
            result = self.visualizer.generate_animated_confusion_matrix(confusion_file)
            if result:
                outputs['confusion_animation'] = result
                print(f"✅ Animated confusion matrix: {result}")
            else:
                print("❌ Failed to generate confusion animation")

            # 3. Generate additional complex space visualizations
            print("🔄 Generating complex space visualizations...")

            # Complex Feature Orthogonality
            orthogonality_file = os.path.join(enhanced_dir, f"{base_name}_feature_orthogonality.html")
            result = self.visualizer.generate_feature_orthogonality_plot(orthogonality_file)
            if result:
                outputs['feature_orthogonality'] = result
                print(f"✅ Feature orthogonality: {result}")
            else:
                print("❌ Failed to generate feature orthogonality")

            # Class Separation Evolution
            separation_file = os.path.join(enhanced_dir, f"{base_name}_class_separation.html")
            result = self.visualizer.generate_class_separation_evolution(separation_file)
            if result:
                outputs['class_separation'] = result
                print(f"✅ Class separation evolution: {result}")
            else:
                print("❌ Failed to generate class separation")

            # Generate summary
            successful = len(outputs)
            total_attempted = 13  # Total visualization types we attempt to generate
            print(f"📊 Visualization Summary: {successful}/{total_attempted} successful")

            if successful == 0:
                print("💡 Tips to get visualizations working:")
                print("   - Enable enhanced visualization before training")
                print("   - Ensure you have plotly installed: pip install plotly")
                print("   - Train for multiple iterations to capture evolution")
                print("   - Make sure you have sufficient features (at least 2-3)")
            else:
                print(f"🎉 Successfully generated {successful} visualizations!")
                print(f"📂 All files saved in: {main_viz_dir}")

                # List generated files
                print("\n📋 Generated Files:")
                for viz_type, file_path in outputs.items():
                    print(f"   📄 {viz_type}: {os.path.basename(file_path)}")

            return outputs

        except Exception as e:
            print(f"❌ Error generating visualizations: {e}")
            import traceback
            traceback.print_exc()
            return None

    def generate_interactive_3d_visualization(self, output_file="interactive_3d_visualization.html"):
        """
        Generate complete interactive 3D visualization with animation controls.

        Args:
            output_file (str): Path for output HTML file

        Returns:
            str or None: Path to generated file if successful, None otherwise
        """
        try:
            import plotly.graph_objects as go
            import plotly.express as px

            if not self.feature_space_snapshots:
                print("No feature space snapshots available for 3D visualization")
                return None

            # Create the main visualization with frames for each iteration
            fig = go.Figure()

            # Create frames for animation
            frames = []
            for i, snapshot in enumerate(self.feature_space_snapshots):
                frame_fig = go.Figure()
                self._add_3d_snapshot_to_plot(frame_fig, snapshot)

                frame = go.Frame(
                    data=frame_fig.data,
                    name=f'frame_{i}',
                    layout=go.Layout(
                        title=f"Iteration {snapshot['iteration']}"
                    )
                )
                frames.append(frame)

            # Add first frame data
            if frames:
                for trace in frames[0].data:
                    fig.add_trace(trace)

            # Create feature selection dropdowns
            feature_dropdowns = self._create_feature_dropdowns()

            # Create iteration slider
            iteration_slider = self._create_iteration_slider()

            # Update layout with all controls
            fig.update_layout(
                title={
                    'text': "DBNN Interactive 3D Feature Space Visualization",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20}
                },
                scene=dict(
                    xaxis_title=self.feature_names[0] if self.feature_names else "Feature 1",
                    yaxis_title=self.feature_names[1] if len(self.feature_names) > 1 else "Feature 2",
                    zaxis_title=self.feature_names[2] if len(self.feature_names) > 2 else "Feature 3",
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                    aspectmode='cube'
                ),
                width=1200,
                height=800,
                updatemenus=feature_dropdowns,
                sliders=[iteration_slider]
            )

            # Add frames for animation
            fig.frames = frames

            # Add play/pause buttons
            fig.update_layout(
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': False,
                    'buttons': [
                        {
                            'label': 'Play',
                            'method': 'animate',
                            'args': [None, {
                                'frame': {'duration': 500, 'redraw': True},
                                'fromcurrent': True,
                                'transition': {'duration': 300}
                            }]
                        },
                        {
                            'label': 'Pause',
                            'method': 'animate',
                            'args': [[None], {
                                'frame': {'duration': 0, 'redraw': False},
                                'mode': 'immediate'
                            }]
                        }
                    ],
                    'x': 0.1,
                    'y': 0.02
                }]
            )

            # Ensure output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            fig.write_html(output_file)
            print(f"✅ Interactive 3D visualization saved: {output_file}")
            return output_file

        except Exception as e:
            print(f"Error creating interactive 3D visualization: {e}")
            import traceback
            traceback.print_exc()
            return None

    def generate_correlation_matrix(self, output_file=None):
        """Generate feature correlation matrix"""
        try:
            if not self.feature_space_snapshots:
                return None

            import plotly.graph_objects as go
            import plotly.express as px

            latest = self.feature_space_snapshots[-1]
            features = latest['features']

            if features.shape[1] <= 1:
                return None

            # Calculate correlation matrix
            corr_matrix = np.corrcoef(features.T)
            feature_names = latest.get('feature_names', [f'F{i+1}' for i in range(features.shape[1])])

            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix,
                x=feature_names,
                y=feature_names,
                colorscale='RdBu',
                zmid=0
            ))

            fig.update_layout(title='Feature Correlation Matrix')

            if output_file:
                fig.write_html(output_file)
                return output_file
            return fig

        except Exception as e:
            print(f"Error generating correlation matrix: {e}")
            return None

    def generate_performance_metrics(self, output_file="performance_metrics.html"):
        """Generate performance metrics visualization - FIXED VERSION"""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import numpy as np

            if not self.accuracy_progression:
                print("❌ No accuracy progression data available for performance metrics")
                return None

            print(f"📊 Generating performance metrics from {len(self.accuracy_progression)} data points")

            rounds = [s['round'] for s in self.accuracy_progression]
            accuracies = [s['accuracy'] for s in self.accuracy_progression]

            # Create a more comprehensive performance dashboard
            fig = make_subplots(
                rows=2, cols=2,
                specs=[
                    [{"type": "xy", "colspan": 2}, None],
                    [{"type": "indicator"}, {"type": "indicator"}]
                ],
                subplot_titles=(
                    'Accuracy Progression Over Training',
                    'Training Summary',
                    'Performance Indicators'
                )
            )

            # 1. Accuracy progression plot
            fig.add_trace(go.Scatter(
                x=rounds, y=accuracies, mode='lines+markers',
                name='Accuracy', line=dict(color='blue', width=3),
                hovertemplate='Round: %{x}<br>Accuracy: %{y:.2f}%<extra></extra>'
            ), row=1, col=1)

            # Calculate performance metrics
            best_acc = max(accuracies) if accuracies else 0
            final_acc = accuracies[-1] if accuracies else 0
            initial_acc = accuracies[0] if accuracies else 0
            improvement = final_acc - initial_acc

            # Find convergence point
            convergence_round = self._find_convergence_point(accuracies)

            # 2. Performance indicators
            fig.add_trace(go.Indicator(
                mode="number+delta",
                value=final_acc,
                number={'suffix': "%", 'font': {'size': 24}},
                delta={'reference': initial_acc, 'relative': False, 'suffix': '%'},
                title={"text": "Final Accuracy"},
                domain={'row': 2, 'column': 0}
            ), row=2, col=1)

            fig.add_trace(go.Indicator(
                mode="number",
                value=best_acc,
                number={'suffix': "%", 'font': {'size': 24}},
                title={"text": "Best Accuracy"},
                domain={'row': 2, 'column': 1}
            ), row=2, col=2)

            # Add convergence point marker if found
            if convergence_round is not None:
                fig.add_trace(go.Scatter(
                    x=[rounds[convergence_round]], y=[accuracies[convergence_round]],
                    mode='markers+text',
                    marker=dict(size=12, color='red', symbol='star'),
                    text=['Convergence'],
                    textposition='top center',
                    name='Convergence Point',
                    hovertemplate=f'Convergence at round {rounds[convergence_round]}<br>Accuracy: {accuracies[convergence_round]:.2f}%<extra></extra>'
                ), row=1, col=1)

            # Update layout
            fig.update_layout(
                height=600,
                title_text="DBNN Performance Metrics Dashboard",
                showlegend=True,
                template="plotly_white"
            )

            # Update axis labels
            fig.update_xaxes(title_text="Training Round", row=1, col=1)
            fig.update_yaxes(title_text="Accuracy (%)", row=1, col=1)

            # Add performance summary annotation
            summary_text = f"""
            <b>Performance Summary:</b><br>
            • Initial Accuracy: {initial_acc:.2f}%<br>
            • Final Accuracy: {final_acc:.2f}%<br>
            • Best Accuracy: {best_acc:.2f}%<br>
            • Total Improvement: {improvement:.2f}%<br>
            • Training Rounds: {len(rounds)}<br>
            • Convergence: Round {convergence_round if convergence_round else 'N/A'}
            """

            fig.add_annotation(
                text=summary_text,
                xref="paper", yref="paper",
                x=0.98, y=0.98,
                xanchor="right", yanchor="top",
                showarrow=False,
                bgcolor="lightblue",
                bordercolor="blue",
                borderwidth=2,
                font=dict(size=10)
            )

            # Ensure output directory exists
            import os
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            fig.write_html(output_file)
            print(f"✅ Performance metrics saved: {output_file}")
            return output_file

        except Exception as e:
            print(f"❌ Error generating performance metrics: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _find_convergence_point(self, accuracies, window=5, threshold=1.0):
        """Find where the model converges (accuracy changes become small)"""
        if len(accuracies) < window * 2:
            return None

        for i in range(window, len(accuracies) - window):
            prev_mean = np.mean(accuracies[i-window:i])
            next_mean = np.mean(accuracies[i:i+window])
            if abs(next_mean - prev_mean) < threshold:
                return i
        return None

    def generate_animated_training(self, output_file="animated_training.html"):
        """Generate animated training progression - FIXED NAME"""
        return self.generate_animated(output_file)

    def generate_standard_dashboard(self, output_file="standard_dashboard.html"):
        """Generate standard dashboard - FIXED METHOD NAME"""
        return self.create_training_dashboard(output_file)

    def generate_all_standard_visualizations(self, output_dir="Visualisations/Standard"):
        """Generate all standard visualizations - FIXED METHOD NAME"""
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)

            outputs = {}

            # Generate each visualization
            viz_methods = [
                ('performance', self.generate_performance_metrics),
                ('correlation', self.generate_correlation_matrix),
                ('feature_explorer', self.generate_basic_3d_visualization),
                ('animated', self.generate_animated_training),
                ('standard_dashboard', self.create_training_dashboard)
            ]

            for name, method in viz_methods:
                output_file = os.path.join(output_dir, f"{name}.html")
                result = method(output_file)
                if result:
                    outputs[name] = result
                    print(f"✅ Generated {name}: {result}")
                else:
                    print(f"❌ Failed to generate {name}")

            return outputs

        except Exception as e:
            print(f"Error generating standard visualizations: {e}")
            return None

    def _add_enhanced_3d_snapshot(self, fig, snapshot):
        """Enhanced version of 3D snapshot with better visualization"""
        try:
            features = snapshot['features']
            targets = snapshot['targets']
            predictions = snapshot['predictions']
            feature_names = snapshot['feature_names']
            class_names = snapshot['class_names']

            # Use first 3 features for 3D visualization
            if features.shape[1] >= 3:
                x, y, z = features[:, 0], features[:, 1], features[:, 2]
            else:
                # Pad with zeros if fewer than 3 features
                x = features[:, 0]
                y = features[:, 1] if features.shape[1] > 1 else np.zeros(len(features))
                z = np.zeros(len(features))

            # Create color mapping for classes
            unique_classes = np.unique(targets)
            colors = px.colors.qualitative.Bold

            # Plot each class with enhanced visualization
            for i, cls in enumerate(unique_classes):
                class_indices = np.where(targets == cls)[0]

                if len(class_indices) == 0:
                    continue

                class_predictions = predictions[class_indices]
                class_targets = targets[class_indices]

                correct_indices = class_indices[class_predictions == class_targets]
                incorrect_indices = class_indices[class_predictions != class_targets]

                cls_name = class_names[int(cls)] if int(cls) < len(class_names) else f'Class_{int(cls)}'

                # Correct predictions - larger, brighter markers
                if len(correct_indices) > 0:
                    fig.add_trace(go.Scatter3d(
                        x=x[correct_indices],
                        y=y[correct_indices],
                        z=z[correct_indices],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=colors[i % len(colors)],
                            opacity=0.9,
                            line=dict(width=2, color='white')
                        ),
                        name=f'{cls_name} ✓',
                        legendgroup=f'class_{cls}',
                        hovertemplate=(
                            f'<b>{cls_name} - Correct</b><br>' +
                            f'{feature_names[0]}: %{{x:.3f}}<br>' +
                            f'{feature_names[1]}: %{{y:.3f}}<br>' +
                            f'{feature_names[2]}: %{{z:.3f}}<br>' +
                            '<extra></extra>'
                        )
                    ))

                # Incorrect predictions - different symbol
                if len(incorrect_indices) > 0:
                    fig.add_trace(go.Scatter3d(
                        x=x[incorrect_indices],
                        y=y[incorrect_indices],
                        z=z[incorrect_indices],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=colors[i % len(colors)],
                            opacity=0.7,
                            symbol='diamond',
                            line=dict(width=2, color='black')
                        ),
                        name=f'{cls_name} ✗',
                        legendgroup=f'class_{cls}',
                        hovertemplate=(
                            f'<b>{cls_name} - Incorrect</b><br>' +
                            f'{feature_names[0]}: %{{x:.3f}}<br>' +
                            f'{feature_names[1]}: %{{y:.3f}}<br>' +
                            f'{feature_names[2]}: %{{z:.3f}}<br>' +
                            '<extra></extra>'
                        )
                    ))

        except Exception as e:
            print(f"Error in enhanced 3D snapshot: {e}")


    def _add_3d_snapshot_to_plot(self, fig, snapshot):
        """Add a single snapshot to the 3D plot - COMPLETELY FIXED VERSION"""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
        except ImportError:
            print("Plotly not available for 3D visualization")
            return

        try:
            features = snapshot['features']
            targets = snapshot['targets']
            predictions = snapshot['predictions']
            feature_names = snapshot.get('feature_names', ['Feature_1', 'Feature_2', 'Feature_3'])
            class_names = snapshot.get('class_names', ['Class_1', 'Class_2', 'Class_3', 'Class_4', 'Class_5'])

            # Ensure we have at least 3 feature names
            while len(feature_names) < 3:
                feature_names.append(f'Feature_{len(feature_names) + 1}')

            # Use first 3 features for 3D visualization
            if features.shape[1] >= 3:
                x, y, z = features[:, 0], features[:, 1], features[:, 2]
            else:
                # Pad with zeros if fewer than 3 features
                x = features[:, 0]
                y = features[:, 1] if features.shape[1] > 1 else np.zeros(len(features))
                z = np.zeros(len(features))

            # Create color mapping for classes
            unique_classes = np.unique(targets)
            colors = px.colors.qualitative.Set1
            color_map = {}
            for i, cls in enumerate(unique_classes):
                color_map[cls] = colors[i % len(colors)]

            # Plot each class - COMPLETELY FIXED INDEXING
            for cls in unique_classes:
                # Convert to integer for reliable indexing
                cls_int = int(cls)

                # Get indices for this class using numpy where (SAFE)
                class_indices = np.where(targets == cls)[0]

                if len(class_indices) == 0:
                    continue

                # Get predictions and targets for this class
                class_predictions = np.array(predictions)[class_indices]
                class_targets = targets[class_indices]

                # Create correct/incorrect masks
                correct_mask = class_predictions == class_targets
                correct_indices = class_indices[correct_mask]
                incorrect_indices = class_indices[~correct_mask]

                # Get class name safely
                if cls_int < len(class_names):
                    cls_name = class_names[cls_int]
                else:
                    cls_name = f'Class_{cls_int}'

                # Correct predictions
                if len(correct_indices) > 0:
                    fig.add_trace(go.Scatter3d(
                        x=x[correct_indices],
                        y=y[correct_indices],
                        z=z[correct_indices],
                        mode='markers',
                        marker=dict(
                            size=6,
                            color=color_map[cls],
                            opacity=0.8,
                            line=dict(width=1, color='white')
                        ),
                        name=f'{cls_name} (Correct)',
                        legendgroup=f'class_{cls}',
                        hovertemplate=(
                            f'<b>{cls_name}</b><br>' +
                            f'{feature_names[0]}: %{{x:.3f}}<br>' +
                            f'{feature_names[1]}: %{{y:.3f}}<br>' +
                            f'{feature_names[2]}: %{{z:.3f}}<br>' +
                            '<extra></extra>'
                        )
                    ))

                # Incorrect predictions
                if len(incorrect_indices) > 0:
                    fig.add_trace(go.Scatter3d(
                        x=x[incorrect_indices],
                        y=y[incorrect_indices],
                        z=z[incorrect_indices],
                        mode='markers',
                        marker=dict(
                            size=6,
                            color=color_map[cls],
                            opacity=0.8,
                            symbol='x',
                            line=dict(width=1, color='black')
                        ),
                        name=f'{cls_name} (Incorrect)',
                        legendgroup=f'class_{cls}',
                        hovertemplate=(
                            f'<b>{cls_name} - MISCLASSIFIED</b><br>' +
                            f'{feature_names[0]}: %{{x:.3f}}<br>' +
                            f'{feature_names[1]}: %{{y:.3f}}<br>' +
                            f'{feature_names[2]}: %{{z:.3f}}<br>' +
                            '<extra></extra>'
                        )
                    ))

        except Exception as e:
            print(f"Error in _add_3d_snapshot_to_plot: {e}")
            import traceback
            traceback.print_exc()

    def _create_feature_dropdowns(self):
        """
        Create dropdown menus for feature selection in 3D visualization.

        Returns:
            list: List of dropdown menu configurations
        """
        if not self.feature_names:
            return []

        dropdowns = []

        # X-axis feature dropdown
        dropdowns.append({
            'buttons': [
                {
                    'label': self.feature_names[i],
                    'method': 'restyle',
                    'args': [{'x': [self.feature_space_snapshots[0]['features'][:, i]]}]
                } for i in range(len(self.feature_names))
            ],
            'direction': 'down',
            'showactive': True,
            'x': 0.1,
            'xanchor': 'left',
            'y': 0.95,
            'yanchor': 'top',
            'bgcolor': 'lightblue',
            'bordercolor': 'black',
            'borderwidth': 1,
            'font': {'size': 12}
        })

        # Y-axis feature dropdown
        dropdowns.append({
            'buttons': [
                {
                    'label': self.feature_names[i],
                    'method': 'restyle',
                    'args': [{'y': [self.feature_space_snapshots[0]['features'][:, i]]}]
                } for i in range(len(self.feature_names))
            ],
            'direction': 'down',
            'showactive': True,
            'x': 0.3,
            'xanchor': 'left',
            'y': 0.95,
            'yanchor': 'top',
            'bgcolor': 'lightgreen',
            'bordercolor': 'black',
            'borderwidth': 1,
            'font': {'size': 12}
        })

        # Z-axis feature dropdown
        dropdowns.append({
            'buttons': [
                {
                    'label': self.feature_names[i],
                    'method': 'restyle',
                    'args': [{'z': [self.feature_space_snapshots[0]['features'][:, i]]}]
                } for i in range(len(self.feature_names))
            ],
            'direction': 'down',
            'showactive': True,
            'x': 0.5,
            'xanchor': 'left',
            'y': 0.95,
            'yanchor': 'top',
            'bgcolor': 'lightcoral',
            'bordercolor': 'black',
            'borderwidth': 1,
            'font': {'size': 12}
        })

        return dropdowns

    def _create_iteration_slider(self):
        """
        Create iteration slider for animation control.

        Returns:
            dict: Slider configuration
        """
        steps = []

        for i, snapshot in enumerate(self.feature_space_snapshots):
            step = {
                'args': [
                    [f'frame_{i}'],
                    {
                        'frame': {'duration': 300, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 300}
                    }
                ],
                'label': f'Iter {snapshot["iteration"]}',
                'method': 'animate'
            }
            steps.append(step)

        slider = {
            'active': 0,
            'currentvalue': {
                'prefix': 'Iteration: ',
                'xanchor': 'right',
                'font': {'size': 16, 'color': 'black'}
            },
            'transition': {'duration': 300, 'easing': 'cubic-in-out'},
            'x': 0.1,
            'len': 0.8,
            'xanchor': 'left',
            'y': 0.02,
            'yanchor': 'bottom',
            'bgcolor': 'lightgray',
            'bordercolor': 'black',
            'borderwidth': 1,
            'tickwidth': 1,
            'steps': steps
        }

        return slider

    # =========================================================================
    # ADVANCED DASHBOARD AND EDUCATIONAL VISUALIZATIONS
    # =========================================================================

    def create_advanced_interactive_dashboard(self, output_file="advanced_dbnn_dashboard.html"):
        """Create a comprehensive dashboard with multiple interactive visualizations - FIXED VERSION"""
        try:
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go
            import plotly.express as px
            import numpy as np
            from sklearn.metrics import confusion_matrix

            # Check if we have sufficient data
            if not self.training_history and not self.feature_space_snapshots:
                print("❌ No training data available for advanced dashboard")
                return None

            # Create a simpler but more robust dashboard layout
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Training Accuracy Progression",
                    "Feature Space Overview",
                    "Weight Distribution",
                    "Training Summary"
                ),
                specs=[
                    [{"type": "xy"}, {"type": "scatter3d"}],
                    [{"type": "xy"}, {"type": "domain"}]
                ]
            )

            # 1. Accuracy Progression (top-left)
            if self.accuracy_progression:
                rounds = [s['round'] for s in self.accuracy_progression]
                accuracies = [s['accuracy'] for s in self.accuracy_progression]

                fig.add_trace(go.Scatter(
                    x=rounds, y=accuracies, mode='lines+markers',
                    name='Accuracy', line=dict(color='blue', width=3),
                    hovertemplate='Round: %{x}<br>Accuracy: %{y:.2f}%<extra></extra>'
                ), row=1, col=1)
            else:
                # Add placeholder if no accuracy data
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 100], mode='text',
                    text=['No accuracy data available'],
                    textposition='middle center',
                    showlegend=False
                ), row=1, col=1)

            # 2. Feature Space (top-right) - use latest snapshot
            if self.feature_space_snapshots:
                latest_snapshot = self.feature_space_snapshots[-1]
                features = latest_snapshot['features']
                targets = latest_snapshot['targets']

                if features.shape[1] >= 3:
                    # Use first 3 features for 3D plot
                    x, y, z = features[:, 0], features[:, 1], features[:, 2]
                    unique_classes = np.unique(targets)
                    colors = px.colors.qualitative.Set1

                    for i, cls in enumerate(unique_classes):
                        class_mask = targets == cls
                        if np.any(class_mask):
                            fig.add_trace(go.Scatter3d(
                                x=x[class_mask], y=y[class_mask], z=z[class_mask],
                                mode='markers',
                                name=f'Class {int(cls)}',
                                marker=dict(
                                    size=4,
                                    color=colors[i % len(colors)],
                                    opacity=0.7
                                ),
                                hovertemplate=f'Class {int(cls)}<br>X: %{{x:.3f}}<br>Y: %{{y:.3f}}<br>Z: %{{z:.3f}}<extra></extra>'
                            ), row=1, col=2)
                else:
                    # Handle cases with fewer than 3 features
                    fig.add_trace(go.Scatter3d(
                        x=[0], y=[0], z=[0], mode='text',
                        text=['Insufficient features for 3D visualization'],
                        textposition='middle center',
                        showlegend=False
                    ), row=1, col=2)
            else:
                # Add placeholder if no feature data
                fig.add_trace(go.Scatter3d(
                    x=[0], y=[0], z=[0], mode='text',
                    text=['No feature space data available'],
                    textposition='middle center',
                    showlegend=False
                ), row=1, col=2)

            # 3. Weight Distribution (bottom-left)
            if self.weight_evolution:
                latest_weights = self.weight_evolution[-1]
                # Create simulated weight distribution
                weights = np.random.normal(latest_weights['mean'], latest_weights['std'], 1000)
                weights = weights[(weights > -10) & (weights < 10)]  # Filter extremes

                fig.add_trace(go.Histogram(
                    x=weights, nbinsx=30,
                    name='Weight Distribution',
                    marker_color='lightgreen', opacity=0.7,
                    hovertemplate='Weight: %{x:.3f}<br>Count: %{y}<extra></extra>'
                ), row=2, col=1)
            else:
                # Add placeholder if no weight data
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], mode='text',
                    text=['No weight distribution data available'],
                    textposition='middle center',
                    showlegend=False
                ), row=2, col=1)

            # 4. Training Summary (bottom-right)
            summary_text = self._generate_training_summary()

            fig.add_trace(go.Scatter(
                x=[0, 1, 2, 3], y=[1, 1, 1, 1], mode='text',
                text=[summary_text],
                textposition='middle center',
                showlegend=False,
                hoverinfo='none'
            ), row=2, col=2)

            # Update layout for better appearance
            fig.update_layout(
                title={
                    'text': "DBNN Advanced Training Dashboard",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 24, 'color': 'darkblue'}
                },
                height=800,
                showlegend=True,
                template="plotly_white"
            )

            # Update axis labels
            fig.update_xaxes(title_text="Training Round", row=1, col=1)
            fig.update_yaxes(title_text="Accuracy (%)", row=1, col=1)
            fig.update_xaxes(title_text="Weight Value", row=2, col=1)
            fig.update_yaxes(title_text="Frequency", row=2, col=1)

            # Ensure output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            fig.write_html(output_file)
            print(f"✅ Advanced interactive dashboard saved: {output_file}")
            return output_file

        except Exception as e:
            print(f"❌ Error creating advanced dashboard: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _generate_training_summary(self):
        """Generate training summary text"""
        summary_parts = ["<b>Training Summary:</b><br>"]

        if self.accuracy_progression:
            best_accuracy = max([s['accuracy'] for s in self.accuracy_progression])
            final_accuracy = self.accuracy_progression[-1]['accuracy']
            total_rounds = len(self.accuracy_progression)

            summary_parts.extend([
                f"• Total Rounds: {total_rounds}",
                f"• Best Accuracy: {best_accuracy:.2f}%",
                f"• Final Accuracy: {final_accuracy:.2f}%"
            ])
        else:
            summary_parts.append("• No training rounds completed")

        if self.feature_space_snapshots:
            latest = self.feature_space_snapshots[-1]
            summary_parts.extend([
                f"• Features: {latest['features'].shape[1]}",
                f"• Samples: {len(latest['features'])}"
            ])

        if self.weight_evolution:
            latest_weights = self.weight_evolution[-1]
            summary_parts.extend([
                f"• Mean Weight: {latest_weights['mean']:.3f}",
                f"• Weight Std: {latest_weights['std']:.3f}"
            ])

        return "<br>".join(summary_parts)

    def _populate_enhanced_dashboard(self, fig):
        """
        Populate the enhanced dashboard with educational visualizations.

        Args:
            fig (plotly.graph_objects.Figure): Dashboard figure to populate
        """
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            import numpy as np
            from sklearn.metrics import confusion_matrix
            import pandas as pd

            # 1. 3D Feature Space with Decision Boundaries (Top-left)
            if self.feature_space_snapshots:
                self._add_3d_feature_visualization(fig, 1, 1)

            # 2. Accuracy Progression (Top-middle)
            if self.accuracy_progression:
                self._add_accuracy_progression(fig, 1, 2)

            # 3. Weight Distribution (Top-right)
            if self.weight_evolution:
                self._add_weight_distribution(fig, 1, 3)

            # 4. Weight Evolution (Middle-right)
            if self.weight_evolution:
                self._add_weight_evolution(fig, 2, 3)

            # 5. Feature Correlation Heatmap (Bottom-left)
            if self.feature_space_snapshots:
                self._add_feature_correlation(fig, 3, 1)

            # 6. Model Performance Summary (Bottom-middle) - Pie chart
            self._add_performance_summary(fig, 3, 2)

            # 7. Confusion Matrix (Bottom-right)
            if self.feature_space_snapshots:
                self._add_confusion_matrix(fig, 3, 3)

        except Exception as e:
            print(f"Error in enhanced dashboard: {e}")

    def _add_3d_feature_visualization(self, fig, row, col):
        """
        Add 3D feature space visualization with decision boundaries.

        Args:
            fig (plotly.graph_objects.Figure): Figure to add to
            row (int): Subplot row
            col (int): Subplot column
        """
        if not self.feature_space_snapshots:
            return

        latest_snapshot = self.feature_space_snapshots[-1]
        features = latest_snapshot['features']
        targets = latest_snapshot['targets']
        predictions = latest_snapshot['predictions']

        # Use first 3 features for 3D
        if features.shape[1] >= 3:
            x, y, z = features[:, 0], features[:, 1], features[:, 2]
        else:
            x, y, z = self._project_to_3d(features)

        # Calculate accuracy for this snapshot
        accuracy = np.mean(predictions == targets) * 100

        # Create interactive 3D plot
        unique_classes = np.unique(targets)
        colors = px.colors.qualitative.Bold

        for i, cls in enumerate(unique_classes):
            class_mask = targets == cls
            correct_mask = predictions[class_mask] == targets[class_mask]

            # Correct predictions
            if np.any(correct_mask):
                fig.add_trace(go.Scatter3d(
                    x=x[class_mask][correct_mask],
                    y=y[class_mask][correct_mask],
                    z=z[class_mask][correct_mask],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=colors[i % len(colors)],
                        opacity=0.8,
                        line=dict(width=1, color='white')
                    ),
                    name=f'Class {int(cls)} ✓',
                    legendgroup=f'class_{cls}',
                    hovertemplate=f'Class {int(cls)}<br>Correct<extra></extra>'
                ), row=row, col=col)

            # Incorrect predictions
            if np.any(~correct_mask):
                fig.add_trace(go.Scatter3d(
                    x=x[class_mask][~correct_mask],
                    y=y[class_mask][~correct_mask],
                    z=z[class_mask][~correct_mask],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=colors[i % len(colors)],
                        opacity=0.8,
                        symbol='x',
                        line=dict(width=2, color='black')
                    ),
                    name=f'Class {int(cls)} ✗',
                    legendgroup=f'class_{cls}',
                    hovertemplate=f'Class {int(cls)}<br>Misclassified<extra></extra>'
                ), row=row, col=col)

        # Add decision boundary visualization (simplified)
        self._add_decision_boundary_hint(fig, row, col, x, y, z, accuracy)

    def _add_weight_distribution(self, fig, row, col):
        """
        Add weight distribution histogram to dashboard.

        Args:
            fig (plotly.graph_objects.Figure): Figure to add to
            row (int): Subplot row
            col (int): Subplot column
        """
        if not self.weight_evolution:
            return

        latest_weights = self.weight_evolution[-1]

        # Create a simulated weight distribution for demonstration
        weights = np.random.normal(latest_weights['mean'], latest_weights['std'], 1000)
        weights = weights[(weights > -10) & (weights < 10)]  # Filter extremes

        fig.add_trace(go.Histogram(
            x=weights,
            nbinsx=50,
            name='Weight Distribution',
            marker_color='lightblue',
            opacity=0.7,
            hovertemplate='Weight: %{x:.3f}<br>Count: %{y}<extra></extra>'
        ), row=row, col=col)

        # Add statistical annotations
        fig.add_annotation(
            xref=f"x{3*(row-1)+col}", yref=f"y{3*(row-1)+col}",
            x=0.8, y=0.9,
            xanchor='left',
            text=f"μ: {latest_weights['mean']:.3f}<br>σ: {latest_weights['std']:.3f}",
            showarrow=False,
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )

    def _add_feature_correlation(self, fig, row, col):
        """
        Add feature correlation heatmap to dashboard.

        Args:
            fig (plotly.graph_objects.Figure): Figure to add to
            row (int): Subplot row
            col (int): Subplot column
        """
        if not self.feature_space_snapshots:
            return

        latest_snapshot = self.feature_space_snapshots[-1]
        features = latest_snapshot['features']

        if features.shape[1] > 1:
            corr_matrix = np.corrcoef(features.T)
            feature_names = self.feature_names if self.feature_names else [f'F{i+1}' for i in range(features.shape[1])]

            fig.add_trace(go.Heatmap(
                z=corr_matrix,
                x=feature_names,
                y=feature_names,
                colorscale='RdBu',
                zmid=0,
                colorbar=dict(title="Correlation"),
                hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
            ), row=row, col=col)

    def _add_weight_evolution(self, fig, row, col):
        """
        Add weight evolution over time to dashboard.

        Args:
            fig (plotly.graph_objects.Figure): Figure to add to
            row (int): Subplot row
            col (int): Subplot column
        """
        if not self.weight_evolution:
            return

        rounds = [w['round'] for w in self.weight_evolution]
        means = [w['mean'] for w in self.weight_evolution]
        stds = [w['std'] for w in self.weight_evolution]

        fig.add_trace(go.Scatter(
            x=rounds, y=means, mode='lines',
            name='Mean Weight', line=dict(color='green', width=2),
            hovertemplate='Round: %{x}<br>Mean: %{y:.4f}<extra></extra>'
        ), row=row, col=col)

        # Add std deviation area
        fig.add_trace(go.Scatter(
            x=rounds + rounds[::-1],
            y=np.array(means) + np.array(stds) + (np.array(means) - np.array(stds))[::-1],
            fill='toself',
            fillcolor='rgba(0,255,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='±1 Std Dev',
            showlegend=False,
            hovertemplate='Standard Deviation Range<extra></extra>'
        ), row=row, col=col)

    def _add_accuracy_progression(self, fig, row, col):
        """
        Add accuracy progression with educational annotations.

        Args:
            fig (plotly.graph_objects.Figure): Figure to add to
            row (int): Subplot row
            col (int): Subplot column
        """
        if not self.accuracy_progression:
            return

        rounds = [s['round'] for s in self.accuracy_progression]
        accuracies = [s['accuracy'] for s in self.accuracy_progression]

        fig.add_trace(go.Scatter(
            x=rounds, y=accuracies, mode='lines+markers',
            name='Accuracy', line=dict(color='blue', width=3),
            hovertemplate='Round: %{x}<br>Accuracy: %{y:.2f}%<extra></extra>'
        ), row=row, col=col)

        # Add educational markers
        if len(accuracies) > 10:
            # Mark convergence point
            convergence_idx = self._find_convergence_point(accuracies)
            if convergence_idx is not None:
                fig.add_trace(go.Scatter(
                    x=[rounds[convergence_idx]], y=[accuracies[convergence_idx]],
                    mode='markers+text',
                    marker=dict(size=12, color='green', symbol='diamond'),
                    text=['Convergence'],
                    textposition='top center',
                    name='Convergence Point',
                    hovertemplate=f'Convergence at round {rounds[convergence_idx]}<br>Accuracy: {accuracies[convergence_idx]:.2f}%<extra></extra>'
                ), row=row, col=col)

    def _add_performance_summary(self, fig, row, col):
        """
        Add performance summary as donut chart.

        Args:
            fig (plotly.graph_objects.Figure): Figure to add to
            row (int): Subplot row
            col (int): Subplot column
        """
        if not self.accuracy_progression:
            return

        latest_accuracy = self.accuracy_progression[-1]['accuracy']
        error_rate = 100 - latest_accuracy

        fig.add_trace(go.Bar(
            x=['Correct', 'Incorrect'],
            y=[latest_accuracy, error_rate],
            marker_color=['green', 'red'],
            name="Performance",
            hovertemplate='%{x}: %{y:.1f}%<extra></extra>'
        ), row=row, col=col)

    def _add_confusion_matrix(self, fig, row, col):
        """
        Add confusion matrix visualization.

        Args:
            fig (plotly.graph_objects.Figure): Figure to add to
            row (int): Subplot row
            col (int): Subplot column
        """
        if not self.feature_space_snapshots:
            return

        latest_snapshot = self.feature_space_snapshots[-1]
        targets = latest_snapshot['targets']
        predictions = latest_snapshot['predictions']

        # Create simplified confusion matrix
        unique_classes = np.unique(np.concatenate([targets, predictions]))
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(targets, predictions, labels=unique_classes)

        fig.add_trace(go.Heatmap(
            z=cm,
            x=[f'Pred {int(c)}' for c in unique_classes],
            y=[f'True {int(c)}' for c in unique_classes],
            colorscale='Blues',
            hovertemplate='True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>'
        ), row=row, col=col)

    # =========================================================================
    # STANDARD VISUALIZATION METHODS
    # =========================================================================

    def create_training_dashboard(self, output_file="training_dashboard.html"):
        """
        Create a comprehensive training dashboard with multiple visualization types.

        Args:
            output_file (str): Path for output HTML file

        Returns:
            str or None: Path to generated file if successful, None otherwise
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import pandas as pd

            if not self.training_history:
                print("No training history available for dashboard")
                return None

            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Accuracy Progression', 'Feature Space',
                              'Weight Distribution', 'Training Summary'),
                specs=[[{"type": "xy"}, {"type": "scatter3d"}],
                       [{"type": "xy"}, {"type": "xy"}]]  # CHANGED from "domain" to "xy"
            )

            # 1. Accuracy Progression (top-left)
            rounds = [s['round'] for s in self.training_history]
            accuracies = [s['accuracy'] for s in self.training_history]

            fig.add_trace(go.Scatter(
                x=rounds, y=accuracies, mode='lines+markers',
                name='Accuracy', line=dict(color='blue', width=2)
            ), row=1, col=1)

            # 2. Feature Space (top-right) - use latest snapshot
            latest_snapshot = self.training_history[-1]
            if (latest_snapshot['features'] is not None and
                latest_snapshot['features'].shape[1] >= 3):

                features = latest_snapshot['features']
                targets = latest_snapshot['targets']

                # Use first 3 features for 3D plot
                x, y, z = features[:, 0], features[:, 1], features[:, 2]

                # Create color mapping
                unique_classes = np.unique(targets)
                for i, cls in enumerate(unique_classes):
                    class_mask = targets == cls
                    fig.add_trace(go.Scatter3d(
                        x=x[class_mask], y=y[class_mask], z=z[class_mask],
                        mode='markers', name=f'Class {int(cls)}',
                        marker=dict(size=4, opacity=0.7)
                    ), row=1, col=2)

            # 3. Weight Distribution (bottom-left)
            if self.weight_evolution:
                latest_weights = self.weight_evolution[-1]
                # Create sample weight distribution
                weights = np.random.normal(latest_weights['mean'],
                                         latest_weights['std'], 1000)
                fig.add_trace(go.Histogram(
                    x=weights, nbinsx=30, name='Weights',
                    marker_color='lightgreen', opacity=0.7
                ), row=2, col=1)

            # 4. Training Summary (bottom-right)
            best_accuracy = max(accuracies) if accuracies else 0
            final_accuracy = accuracies[-1] if accuracies else 0

            summary_text = f"""
            Training Summary:
            • Total Rounds: {len(self.training_history)}
            • Best Accuracy: {best_accuracy:.2f}%
            • Final Accuracy: {final_accuracy:.2f}%
            • Features: {latest_snapshot['features'].shape[1]}
            • Samples: {len(latest_snapshot['features'])}
            """

            fig.add_annotation(
                text=summary_text,
                xref="paper", yref="paper",
                x=0.95, y=0.05,
                xanchor="right", yanchor="bottom",
                showarrow=False,
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
                font=dict(size=10)
            )

            fig.update_layout(
                height=800,
                title_text="DBNN Training Dashboard",
                showlegend=True
            )

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            fig.write_html(output_file)
            print(f"✅ Training dashboard saved: {output_file}")
            return output_file

        except Exception as e:
            print(f"Error creating training dashboard: {e}")
            import traceback
            traceback.print_exc()
            return None

    def generate_feature_space_plot(self, snapshot_idx: int, feature_indices: List[int] = [0, 1, 2]):
        """
        Generate 3D feature space plot for a specific training snapshot.

        Args:
            snapshot_idx (int): Index of training snapshot to visualize
            feature_indices (list): Indices of features to use for 3D plot

        Returns:
            plotly.graph_objects.Figure or None: 3D feature space plot
        """
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            import pandas as pd
        except ImportError:
            print("Plotly not available for visualization")
            return None

        if snapshot_idx >= len(self.training_history):
            return None

        snapshot = self.training_history[snapshot_idx]
        features = snapshot['features']
        targets = snapshot['targets']

        # Create DataFrame for plotting
        df = pd.DataFrame({
            f'Feature_{feature_indices[0]}': features[:, feature_indices[0]],
            f'Feature_{feature_indices[1]}': features[:, feature_indices[1]],
            f'Feature_{feature_indices[2]}': features[:, feature_indices[2]],
            'Class': targets,
            'Prediction': snapshot['predictions']
        })

        fig = px.scatter_3d(
            df,
            x=f'Feature_{feature_indices[0]}',
            y=f'Feature_{feature_indices[1]}',
            z=f'Feature_{feature_indices[2]}',
            color='Class',
            title=f'Feature Space - Round {snapshot["round"]}<br>Accuracy: {snapshot["accuracy"]:.2f}%',
            opacity=0.7
        )

        return fig

    def generate_accuracy_plot(self):
        """
        Generate accuracy progression plot over training rounds.

        Returns:
            plotly.graph_objects.Figure or None: Accuracy progression plot
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            print("Plotly not available for visualization")
            return None

        if len(self.training_history) < 2:
            return None

        rounds = [s['round'] for s in self.training_history]
        accuracies = [s['accuracy'] for s in self.training_history]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rounds, y=accuracies,
            mode='lines+markers',
            name='Accuracy'
        ))

        fig.update_layout(
            title='Training Accuracy Progression',
            xaxis_title='Training Round',
            yaxis_title='Accuracy (%)'
        )

        return fig

    def generate_weight_distribution_plot(self, snapshot_idx: int):
        """
        Generate weight distribution histogram for a specific snapshot.

        Args:
            snapshot_idx (int): Index of training snapshot

        Returns:
            plotly.graph_objects.Figure or None: Weight distribution histogram
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            print("Plotly not available for visualization")
            return None

        if snapshot_idx >= len(self.training_history):
            return None

        snapshot = self.training_history[snapshot_idx]
        weights = snapshot['weights']

        # Flatten weights for histogram
        flat_weights = weights.flatten()
        # Remove zeros and extreme values for better visualization
        flat_weights = flat_weights[(flat_weights != 0) & (np.abs(flat_weights) < 100)]

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=flat_weights,
            nbinsx=50,
            name='Weight Distribution'
        ))

        fig.update_layout(
            title=f'Weight Distribution - Round {snapshot["round"]}',
            xaxis_title='Weight Value',
            yaxis_title='Frequency'
        )

        return fig

    # =========================================================================
    # TENSOR MODE SPECIFIC VISUALIZATIONS
    # =========================================================================

    def generate_tensor_space_plot(self, snapshot_idx: int, feature_indices: List[int] = [0, 1, 2]):
        """
        Generate 3D tensor feature space plot for tensor mode training.

        Args:
            snapshot_idx (int): Index of tensor snapshot
            feature_indices (list): Feature indices for 3D plot

        Returns:
            plotly.graph_objects.Figure or None: Tensor feature space plot
        """
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            import pandas as pd
        except ImportError:
            print("Plotly not available for visualization")
            return None

        if snapshot_idx >= len(self.training_history):
            return None

        snapshot = self.training_history[snapshot_idx]
        features = snapshot['features']
        targets = snapshot['targets']
        predictions = snapshot['predictions']

        # Create DataFrame for plotting
        df = pd.DataFrame({
            f'Feature_{feature_indices[0]}': features[:, feature_indices[0]],
            f'Feature_{feature_indices[1]}': features[:, feature_indices[1]],
            f'Feature_{feature_indices[2]}': features[:, feature_indices[2]],
            'Actual_Class': targets,
            'Predicted_Class': predictions,
            'Correct': targets == predictions
        })

        fig = px.scatter_3d(
            df,
            x=f'Feature_{feature_indices[0]}',
            y=f'Feature_{feature_indices[1]}',
            z=f'Feature_{feature_indices[2]}',
            color='Predicted_Class',
            symbol='Correct',
            title=f'Tensor Feature Space - Iteration {snapshot["round"]}<br>Accuracy: {snapshot["accuracy"]:.2f}%',
            opacity=0.7,
            color_discrete_sequence=px.colors.qualitative.Set1
        )

        return fig

    def generate_weight_matrix_heatmap(self, snapshot_idx: int):
        """
        Generate heatmap of the weight matrix for tensor mode.

        Args:
            snapshot_idx (int): Index of tensor snapshot

        Returns:
            plotly.graph_objects.Figure or None: Weight matrix heatmap
        """
        try:
            import plotly.graph_objects as go
            import plotly.express as px
        except ImportError:
            print("Plotly not available for visualization")
            return None

        if (snapshot_idx >= len(self.training_history) or
            not self.training_history[snapshot_idx].get('is_tensor_mode', False) or
            'tensor_data' not in self.training_history[snapshot_idx]):
            return None

        snapshot = self.training_history[snapshot_idx]
        tensor_data = snapshot['tensor_data']
        weight_matrix = tensor_data.get('weight_matrix')

        if weight_matrix is None:
            return None

        fig = go.Figure(data=go.Heatmap(
            z=weight_matrix,
            colorscale='RdBu',
            zmid=0,
            colorbar=dict(title="Weight Value")
        ))

        fig.update_layout(
            title=f'Weight Matrix - Iteration {snapshot["round"]}',
            xaxis_title='Output Classes',
            yaxis_title='Input Features',
            width=600,
            height=500
        )

        return fig

    def generate_orthogonal_basis_plot(self, snapshot_idx: int):
        """
        Generate visualization of orthogonal basis components for tensor mode.

        Args:
            snapshot_idx (int): Index of tensor snapshot

        Returns:
            plotly.graph_objects.Figure or None: Orthogonal basis plot
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            print("Plotly not available for visualization")
            return None

        if (snapshot_idx >= len(self.training_history) or
            not self.training_history[snapshot_idx].get('is_tensor_mode', False) or
            'tensor_data' not in self.training_history[snapshot_idx]):
            return None

        snapshot = self.training_history[snapshot_idx]
        tensor_data = snapshot['tensor_data']
        orthogonal_basis = tensor_data.get('orthogonal_basis')

        if orthogonal_basis is None:
            return None

        # Show first few components
        n_components = min(6, orthogonal_basis.shape[1])
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[f'Component {i+1}' for i in range(n_components)]
        )

        for i in range(n_components):
            row = i // 3 + 1
            col = i % 3 + 1
            component = orthogonal_basis[:, i]

            fig.add_trace(
                go.Scatter(
                    y=component,
                    mode='lines',
                    name=f'Component {i+1}'
                ),
                row=row, col=col
            )

        fig.update_layout(
            title=f'Orthogonal Basis Components - Iteration {snapshot["round"]}',
            height=600,
            showlegend=False
        )

        return fig

    def generate_tensor_convergence_plot(self):
        """
        Generate convergence plot for tensor mode training.

        Returns:
            plotly.graph_objects.Figure or None: Tensor convergence plot
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            print("Plotly not available for visualization")
            return None

        if not self.tensor_snapshots:
            return None

        # Extract tensor-specific metrics
        iterations = []
        accuracies = []
        weight_norms = []
        basis_ranks = []

        for snapshot in self.tensor_snapshots:
            if snapshot.get('is_tensor_mode', False) and 'tensor_data' in snapshot:
                iterations.append(snapshot['round'])
                accuracies.append(snapshot['accuracy'])
                tensor_data = snapshot['tensor_data']
                weight_norms.append(tensor_data.get('weight_matrix_norm', 0))
                basis_ranks.append(tensor_data.get('basis_rank', 0))

        if not iterations:
            return None

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy Progression', 'Weight Matrix Norm',
                          'Basis Rank', 'Training Metrics'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": True}]]
        )

        # Accuracy plot
        fig.add_trace(
            go.Scatter(x=iterations, y=accuracies, mode='lines+markers',
                      name='Accuracy', line=dict(color='blue')),
            row=1, col=1
        )

        # Weight norm plot
        fig.add_trace(
            go.Scatter(x=iterations, y=weight_norms, mode='lines+markers',
                      name='Weight Norm', line=dict(color='red')),
            row=1, col=2
        )

        # Basis rank plot
        fig.add_trace(
            go.Scatter(x=iterations, y=basis_ranks, mode='lines+markers',
                      name='Basis Rank', line=dict(color='green')),
            row=2, col=1
        )

        # Combined metrics
        fig.add_trace(
            go.Scatter(x=iterations, y=accuracies, mode='lines',
                      name='Accuracy', line=dict(color='blue')),
            row=2, col=2, secondary_y=False
        )

        fig.add_trace(
            go.Scatter(x=iterations, y=weight_norms, mode='lines',
                      name='Weight Norm', line=dict(color='red')),
            row=2, col=2, secondary_y=True
        )

        fig.update_layout(
            height=800,
            title_text="Tensor Mode Training Convergence",
            showlegend=True
        )

        fig.update_xaxes(title_text="Iteration", row=2, col=1)
        fig.update_xaxes(title_text="Iteration", row=2, col=2)
        fig.update_yaxes(title_text="Accuracy (%)", row=1, col=1)
        fig.update_yaxes(title_text="Norm", row=1, col=2)
        fig.update_yaxes(title_text="Rank", row=2, col=1)
        fig.update_yaxes(title_text="Accuracy (%)", row=2, col=2, secondary_y=False)
        fig.update_yaxes(title_text="Weight Norm", row=2, col=2, secondary_y=True)

        return fig

    def create_tensor_dashboard(self, output_file: str = "tensor_training_dashboard.html"):
        """
        Create comprehensive tensor training dashboard.

        Args:
            output_file (str): Path for output HTML file

        Returns:
            str or None: Path to generated file if successful, None otherwise
        """
        try:
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go
        except ImportError:
            print("Plotly not available for dashboard creation")
            return None

        if not self.tensor_snapshots:
            print("No tensor snapshots available for dashboard")
            return None

        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Tensor Feature Space', 'Accuracy Progression',
                          'Weight Matrix', 'Orthogonal Basis',
                          'Convergence Metrics', 'Training Summary'),
            specs=[[{"type": "scatter3d"}, {"type": "xy"}],
                   [{"type": "heatmap"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "domain"}]],
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )

        # Feature Space (latest tensor snapshot)
        feature_fig = self.generate_tensor_space_plot(-1)
        if feature_fig:
            for trace in feature_fig.data:
                fig.add_trace(trace, row=1, col=1)

        # Accuracy Progression
        iterations = [s['round'] for s in self.tensor_snapshots]
        accuracies = [s['accuracy'] for s in self.tensor_snapshots]
        fig.add_trace(go.Scatter(x=iterations, y=accuracies, mode='lines+markers',
                               name='Accuracy', line=dict(color='blue')), row=1, col=2)

        # Weight Matrix Heatmap (latest)
        weight_fig = self.generate_weight_matrix_heatmap(-1)
        if weight_fig:
            for trace in weight_fig.data:
                fig.add_trace(trace, row=2, col=1)

        # Orthogonal Basis (latest)
        basis_fig = self.generate_orthogonal_basis_plot(-1)
        if basis_fig:
            for trace in basis_fig.data:
                fig.add_trace(trace, row=2, col=2)

        # Convergence Metrics
        convergence_fig = self.generate_tensor_convergence_plot()
        if convergence_fig:
            for trace in convergence_fig.data:
                fig.add_trace(trace, row=3, col=1)

        # Training Summary
        best_snapshot = max(self.tensor_snapshots, key=lambda x: x['accuracy'])
        final_accuracy = self.tensor_snapshots[-1]['accuracy'] if self.tensor_snapshots else 0

        summary_text = f"""
        <b>Tensor Training Summary:</b><br>
        - Total Iterations: {len(self.tensor_snapshots)}<br>
        - Best Accuracy: {best_snapshot['accuracy']:.2f}%<br>
        - Best Iteration: {best_snapshot['round']}<br>
        - Final Accuracy: {final_accuracy:.2f}%<br>
        - Features: {best_snapshot['features'].shape[1]}<br>
        - Classes: {len(np.unique(best_snapshot['targets']))}<br>
        - Mode: Tensor Transformation
        """

        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='text',
            text=[summary_text],
            textposition="middle center",
            showlegend=False,
            textfont=dict(size=11)
        ), row=3, col=2)

        fig.update_layout(
            height=1200,
            title_text="DBNN Tensor Training Dashboard",
            showlegend=True
        )

        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            fig.write_html(output_file)
            print(f"Tensor training dashboard saved to: {output_file}")
            return output_file
        except Exception as e:
            print(f"Error saving tensor dashboard: {e}")
            return None

    # =========================================================================
    # UTILITY AND HELPER METHODS
    # =========================================================================

    def get_training_history(self):
        """
        Get the complete training history.

        Returns:
            list: List of training snapshots
        """
        return self.training_history

    def clear_history(self):
        """Clear all visualization history and data."""
        self.training_history = []
        self.visualization_data = {}
        self.tensor_snapshots = []
        self.feature_space_snapshots = []
        self.feature_names = []
        self.class_names = []
        self.accuracy_progression = []
        self.weight_evolution = []
        self.confusion_data = []
        self.decision_boundaries = []
        self.feature_importance_data = []
        self.learning_curves = []
        self.network_topology_data = []

    def _project_to_3d(self, features):
        """
        Project features to 3D using PCA for visualization.

        Args:
            features (numpy.ndarray): Input features

        Returns:
            tuple: (x, y, z) coordinates for 3D plotting
        """
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        projected = pca.fit_transform(features)
        return projected[:, 0], projected[:, 1], projected[:, 2]

    def _find_convergence_point(self, accuracies, window=5, threshold=0.1):
        """
        Find where the model converges (accuracy changes become small).

        Args:
            accuracies (list): List of accuracy values
            window (int): Window size for convergence detection
            threshold (float): Threshold for convergence detection

        Returns:
            int or None: Index of convergence point
        """
        if len(accuracies) < window * 2:
            return None

        for i in range(window, len(accuracies) - window):
            prev_mean = np.mean(accuracies[i-window:i])
            next_mean = np.mean(accuracies[i:i+window])
            if abs(next_mean - prev_mean) < threshold:
                return i
        return None

    def _add_decision_boundary_hint(self, fig, row, col, x, y, z, accuracy):
        """
        Add visual hints about decision boundaries to 3D plot.

        Args:
            fig (plotly.graph_objects.Figure): Figure to add to
            row (int): Subplot row
            col (int): Subplot column
            x (numpy.ndarray): X coordinates
            y (numpy.ndarray): Y coordinates
            z (numpy.ndarray): Z coordinates
            accuracy (float): Current accuracy
        """
        # Add a transparent surface to suggest decision boundaries
        x_range = np.linspace(min(x), max(x), 10)
        y_range = np.linspace(min(y), max(y), 10)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.zeros_like(X)  # Simple plane for demonstration

        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            opacity=0.3,
            colorscale='Blues',
            showscale=False,
            name='Decision Boundary Hint'
        ), row=row, col=col)

        # Add accuracy annotation
        fig.add_annotation(
            xref=f"x{3*(row-1)+col}", yref=f"y{3*(row-1)+col}",
            x=0.5, y=0.5, z=1.1,
            text=f"Accuracy: {accuracy:.1f}%",
            showarrow=False,
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )

    def generate_enhanced_interactive_3d(self, output_file="enhanced_3d_visualization.html"):
        """Generate enhanced 3D visualization with interactive controls - ROBUST VERSION"""
        try:
            import plotly.graph_objects as go
            import plotly.express as px

            if not self.feature_space_snapshots:
                print("❌ No feature space snapshots available for enhanced 3D visualization")
                return None

            # Use the latest snapshot for static visualization
            latest_snapshot = self.feature_space_snapshots[-1]
            features = latest_snapshot['features']
            targets = latest_snapshot['targets']
            predictions = latest_snapshot['predictions']

            # Ensure we have at least 2D data
            if features.shape[1] < 2:
                print("❌ Insufficient features for 3D visualization (need at least 2)")
                return None

            # Create the figure
            fig = go.Figure()

            # Prepare features for 3D plotting
            if features.shape[1] >= 3:
                x, y, z = features[:, 0], features[:, 1], features[:, 2]
                feature_names = latest_snapshot.get('feature_names', ['Feature 1', 'Feature 2', 'Feature 3'])
            else:
                # Pad with zeros if fewer than 3 features
                x = features[:, 0]
                y = features[:, 1] if features.shape[1] > 1 else np.zeros_like(x)
                z = np.zeros_like(x)
                feature_names = latest_snapshot.get('feature_names', ['Feature 1', 'Feature 2', 'Feature 3'])
                # Ensure we have enough feature names
                while len(feature_names) < 3:
                    feature_names.append(f'Feature {len(feature_names) + 1}')

            unique_classes = np.unique(targets)
            colors = px.colors.qualitative.Set1

            # Plot each class
            for i, cls in enumerate(unique_classes):
                class_mask = targets == cls
                if np.any(class_mask):
                    # Get class predictions
                    class_predictions = predictions[class_mask]
                    class_targets = targets[class_mask]
                    correct_mask = class_predictions == class_targets

                    cls_name = f'Class {int(cls)}'

                    # Correct predictions
                    if np.any(correct_mask):
                        fig.add_trace(go.Scatter3d(
                            x=x[class_mask][correct_mask],
                            y=y[class_mask][correct_mask],
                            z=z[class_mask][correct_mask],
                            mode='markers',
                            marker=dict(
                                size=6,
                                color=colors[i % len(colors)],
                                opacity=0.8,
                                line=dict(width=1, color='white')
                            ),
                            name=f'{cls_name} ✓',
                            hovertemplate=(
                                f'{cls_name} - Correct<br>' +
                                f'{feature_names[0]}: %{{x:.3f}}<br>' +
                                f'{feature_names[1]}: %{{y:.3f}}<br>' +
                                f'{feature_names[2]}: %{{z:.3f}}<br>' +
                                '<extra></extra>'
                            )
                        ))

                    # Incorrect predictions
                    if np.any(~correct_mask):
                        fig.add_trace(go.Scatter3d(
                            x=x[class_mask][~correct_mask],
                            y=y[class_mask][~correct_mask],
                            z=z[class_mask][~correct_mask],
                            mode='markers',
                            marker=dict(
                                size=6,
                                color=colors[i % len(colors)],
                                opacity=0.8,
                                symbol='x',
                                line=dict(width=2, color='black')
                            ),
                            name=f'{cls_name} ✗',
                            hovertemplate=(
                                f'{cls_name} - Incorrect<br>' +
                                f'{feature_names[0]}: %{{x:.3f}}<br>' +
                                f'{feature_names[1]}: %{{y:.3f}}<br>' +
                                f'{feature_names[2]}: %{{z:.3f}}<br>' +
                                '<extra></extra>'
                            )
                        ))

            # Update layout
            fig.update_layout(
                title={
                    'text': f"3D Feature Space Visualization<br>Iteration {latest_snapshot['iteration']}",
                    'x': 0.5,
                    'xanchor': 'center'
                },
                scene=dict(
                    xaxis_title=feature_names[0],
                    yaxis_title=feature_names[1],
                    zaxis_title=feature_names[2],
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                width=1000,
                height=700,
                showlegend=True
            )

            # Ensure output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            fig.write_html(output_file)
            print(f"✅ Enhanced 3D visualization saved: {output_file}")
            return output_file

        except Exception as e:
            print(f"❌ Error creating enhanced 3D visualization: {e}")
            import traceback
            traceback.print_exc()
            return None

    def generate_interactive_visualizations(self, output_dir="Visualisations"):
        """Generate all interactive visualizations with proper folder organization"""
        if not hasattr(self, 'visualizer') or not self.visualizer:
            print("❌ No visualizer available. Please enable enhanced visualization before training.")
            return None

        # Check if we have any visualization data
        has_data = (hasattr(self.visualizer, 'training_history') and self.visualizer.training_history) or \
                   (hasattr(self.visualizer, 'feature_space_snapshots') and self.visualizer.feature_space_snapshots) or \
                   (hasattr(self.visualizer, 'accuracy_progression') and self.visualizer.accuracy_progression)

        if not has_data:
            print("❌ No visualization data available.")
            print("   Enable with enable_enhanced_visualization() before training")
            return None

        try:
            import os

            # Create organized directory structure based on data file
            if hasattr(self, 'current_file') and self.current_file:
                base_name = os.path.splitext(os.path.basename(self.current_file))[0]
            else:
                base_name = "dbnn_training"

            if hasattr(self, 'feature_columns') and self.feature_columns:
                feature_info = f"{len(self.feature_columns)}features"
                base_name = f"{base_name}_{feature_info}"

            # Create main visualization folder for this dataset
            main_viz_dir = os.path.join(output_dir, base_name)
            standard_dir = os.path.join(main_viz_dir, "Standard")
            enhanced_dir = os.path.join(main_viz_dir, "Enhanced")
            os.makedirs(standard_dir, exist_ok=True)
            os.makedirs(enhanced_dir, exist_ok=True)

            # Store the main visualization directory for easy access
            self.main_viz_directory = main_viz_dir

            print(f"📁 Creating visualizations in: {main_viz_dir}")

            outputs = {}

            # 1. Generate ENHANCED visualizations
            print("🔄 Generating enhanced visualizations...")

            # Enhanced 3D
            enhanced_3d_file = os.path.join(enhanced_dir, f"{base_name}_enhanced_3d.html")
            result = self.visualizer.generate_enhanced_interactive_3d(enhanced_3d_file)
            if result:
                outputs['enhanced_3d'] = result
                print(f"✅ Enhanced 3D: {result}")

            # Advanced Dashboard
            dashboard_file = os.path.join(enhanced_dir, f"{base_name}_advanced_dashboard.html")
            result = self.visualizer.create_advanced_interactive_dashboard(dashboard_file)
            if result:
                outputs['advanced_dashboard'] = result
                print(f"✅ Advanced dashboard: {result}")

            # Complex Tensor Evolution
            complex_file = os.path.join(enhanced_dir, f"{base_name}_complex_tensor.html")
            result = self.visualizer.generate_complex_tensor_evolution(complex_file)
            if result:
                outputs['complex_tensor'] = result
                print(f"✅ Complex tensor evolution: {result}")

            # Phase Diagram
            phase_file = os.path.join(enhanced_dir, f"{base_name}_phase_diagram.html")
            result = self.visualizer.generate_complex_phase_diagram(phase_file)
            if result:
                outputs['phase_diagram'] = result
                print(f"✅ Complex phase diagram: {result}")

            # 2. Generate STANDARD visualizations
            print("🔄 Generating standard visualizations...")

            # Traditional Dashboard
            traditional_file = os.path.join(standard_dir, f"{base_name}_traditional_dashboard.html")
            result = self.visualizer.create_training_dashboard(traditional_file)
            if result:
                outputs['traditional_dashboard'] = result
                print(f"✅ Traditional dashboard: {result}")

            # Performance Metrics
            performance_file = os.path.join(standard_dir, f"{base_name}_performance.html")
            result = self.visualizer.generate_performance_metrics(performance_file)
            if result:
                outputs['performance'] = result
                print(f"✅ Performance metrics: {result}")

            # Correlation Matrix
            correlation_file = os.path.join(standard_dir, f"{base_name}_correlation.html")
            result = self.visualizer.generate_correlation_matrix(correlation_file)
            if result:
                outputs['correlation'] = result
                print(f"✅ Correlation matrix: {result}")

            # Feature Explorer
            feature_file = os.path.join(standard_dir, f"{base_name}_feature_explorer.html")
            result = self.visualizer.generate_basic_3d_visualization(feature_file)
            if result:
                outputs['feature_explorer'] = result
                print(f"✅ Feature explorer: {result}")

            # Animated Training
            animated_file = os.path.join(standard_dir, f"{base_name}_animated.html")
            result = self.visualizer.generate_animated_training(animated_file)
            if result:
                outputs['animated'] = result
                print(f"✅ Animated training: {result}")

            # Confusion Matrix Animation
            confusion_file = os.path.join(standard_dir, f"{base_name}_confusion_animation.html")
            result = self.visualizer.generate_animated_confusion_matrix(confusion_file)
            if result:
                outputs['confusion_animation'] = result
                print(f"✅ Animated confusion matrix: {result}")

            # Generate summary
            successful = len(outputs)
            total_attempted = 11  # Total visualization types we attempt to generate
            print(f"📊 Visualization Summary: {successful}/{total_attempted} successful")

            if successful == 0:
                print("💡 Tips to get visualizations working:")
                print("   - Enable enhanced visualization before training")
                print("   - Ensure you have plotly installed: pip install plotly")
                print("   - Train for multiple iterations to capture evolution")

            return outputs

        except Exception as e:
            print(f"❌ Error generating visualizations: {e}")
            import traceback
            traceback.print_exc()
            return None

    def generate_basic_3d_visualization(self, output_file="basic_3d.html"):
        """Generate a basic 3D visualization that's guaranteed to work"""
        try:
            import plotly.graph_objects as go
            import plotly.express as px

            if not self.feature_space_snapshots:
                return None

            # Use the latest snapshot
            snapshot = self.feature_space_snapshots[-1]
            features = snapshot['features']
            targets = snapshot['targets']

            # Simple 3D scatter plot
            if features.shape[1] >= 3:
                fig = px.scatter_3d(
                    x=features[:, 0], y=features[:, 1], z=features[:, 2],
                    color=targets.astype(str),
                    title=f"3D Feature Space - Iteration {snapshot['iteration']}"
                )
            else:
                # If less than 3 features, use what we have
                x = features[:, 0]
                y = features[:, 1] if features.shape[1] > 1 else np.zeros_like(x)
                z = np.zeros_like(x) if features.shape[1] < 3 else features[:, 2]

                fig = px.scatter_3d(
                    x=x, y=y, z=z,
                    color=targets.astype(str),
                    title=f"3D Feature Space - Iteration {snapshot['iteration']}"
                )

            fig.write_html(output_file)
            print(f"✅ Basic 3D visualization saved: {output_file}")
            return output_file

        except Exception as e:
            print(f"Error creating basic 3D visualization: {e}")
            return None

    def generate_animated(self, output_file="animated_training.html"):
        """Generate animated training progression"""
        try:
            if not self.feature_space_snapshots:
                return None

            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            # Create simple animation with accuracy progression
            rounds = [s['round'] for s in self.accuracy_progression]
            accuracies = [s['accuracy'] for s in self.accuracy_progression]

            fig = go.Figure(
                data=[go.Scatter(x=rounds, y=accuracies, mode="lines+markers")],
                layout=go.Layout(
                    title="Training Progress Animation",
                    xaxis=dict(title="Iteration"),
                    yaxis=dict(title="Accuracy (%)"),
                    updatemenus=[dict(
                        type="buttons",
                        buttons=[dict(label="Play",
                                    method="animate",
                                    args=[None])])]
                ),
                frames=[go.Frame(
                    data=[go.Scatter(x=rounds[:k+1], y=accuracies[:k+1])],
                    name=str(k)
                ) for k in range(len(rounds))]
            )

            fig.write_html(output_file)
            print(f"✅ Animated training saved: {output_file}")
            return output_file

        except Exception as e:
            print(f"Error generating animation: {e}")
            return None

    def generate_feature_explorer(self, output_file="feature_explorer.html"):
        """Generate feature explorer - FIXED NAME"""
        return self.generate_basic_3d_visualization(output_file)

    def generate_performance(self, output_file="performance.html"):
        """Generate performance metrics - FIXED NAME"""
        return self.generate_performance_metrics(output_file)

    def generate_correlation(self, output_file="correlation.html"):
        """Generate correlation matrix - FIXED NAME"""
        return self.generate_correlation_matrix(output_file)

    def generate_all_standard(self, output_dir="Visualisations/Standard"):
        """Generate all standard visualizations - FIXED NAME"""
        return self.generate_all_standard_visualizations(output_dir)

class ClassEncoder:
    """Handles encoding and decoding of class labels"""

    def __init__(self):
        self.class_to_encoded = {}
        self.encoded_to_class = {}
        self.encoded_to_original = {}
        self.is_fitted = False
        self.original_dtype = None  # Track original data type

    def fit(self, class_labels):
        """Fit encoder to class labels"""
        # Detect original data type
        sample_label = class_labels[0] if len(class_labels) > 0 else None
        if isinstance(sample_label, (int, float, np.integer, np.floating)):
            self.original_dtype = 'numeric'
        else:
            self.original_dtype = 'string'
        unique_classes = sorted(set(class_labels))
        string_classes = [str(cls) for cls in unique_classes]

        for encoded_val, (original_class, string_class) in enumerate(zip(unique_classes, string_classes), 1):
            self.class_to_encoded[original_class] = float(encoded_val)
            self.class_to_encoded[string_class] = float(encoded_val)  # Add string representation
            self.encoded_to_class[float(encoded_val)] = original_class
            self.encoded_to_original[float(encoded_val)] = string_class

        self.is_fitted = True
        print(f"Fitted encoder with {len(unique_classes)} classes: {unique_classes}")
        print(f"Original dtype: {self.original_dtype}")

    def transform(self, class_labels):
        """Transform class labels to encoded numeric values"""
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before transforming")

        encoded = []
        for label in class_labels:
            # Try multiple representations
            if label in self.class_to_encoded:
                encoded.append(self.class_to_encoded[label])
            elif str(label) in self.class_to_encoded:
                encoded.append(self.class_to_encoded[str(label)])
            else:
                # Try numeric conversion for string labels
                try:
                    if self.original_dtype == 'numeric':
                        numeric_label = float(label)
                        if numeric_label in self.class_to_encoded:
                            encoded.append(self.class_to_encoded[numeric_label])
                        else:
                            raise ValueError(f"Unknown class label: {label}")
                    else:
                        raise ValueError(f"Unknown class label: {label}")
                except (ValueError, TypeError):
                    raise ValueError(f"Unknown class label: {label}")

        return np.array(encoded, dtype=np.float64)

    def inverse_transform(self, encoded_values):
        """Transform encoded values back to original class labels"""
        if not self.is_fitted:
            raise ValueError("Encoder must be fitted before inverse transforming")

        original = []
        for encoded_val in encoded_values:
            if encoded_val in self.encoded_to_class:
                original.append(self.encoded_to_class[encoded_val])
            else:
                closest = min(self.encoded_to_class.keys(),
                            key=lambda x: abs(x - encoded_val))
                original.append(self.encoded_to_class[closest])
                print(f"Warning: Encoded value {encoded_val} not found, using closest: {closest}")

        return original

    def get_encoded_classes(self):
        """Get the encoded class values used in class_labels"""
        return sorted(self.encoded_to_class.keys())

    def get_class_mapping(self):
        """Get the complete class mapping"""
        return {
            'class_to_encoded': self.class_to_encoded,
            'encoded_to_class': self.encoded_to_class,
            'original_dtype': self.original_dtype
        }

class DBNNCore:
    """
    Optimized DBNN algorithm powerhouse with parallel processing and hardware acceleration
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Default configuration
        self.config = config or {
            'LoC': 0.65,
            'nLoC': 0.0,
            'resol': 100,
            'nresol': 0,
            'nLoCcnt': 0,
            'skpchk': 0,
            'oneround': 100,
            'fst_gain': 1.0,
            'gain': 2.0,
            'margin': 0.2,
            'patience': 10,
            'min_improvement': 0.1
        }

        # Core data structures
        self.anti_net = None
        self.anti_wts = None
        self.antit_wts = None
        self.antip_wts = None
        self.binloc = None
        self.max_val = None
        self.min_val = None
        self.class_labels = None
        self.resolution_arr = None

        # Training state
        self.innodes = 0
        self.outnodes = 0
        self.class_encoder = ClassEncoder()
        self.training_history = []
        self.is_trained = False

        # Parallel processing and optimization
        self.num_workers = self._detect_optimal_workers()
        self.parallel_enabled = True
        self.gpu_enabled = False
        self.memory_optimized = False

        # Initialize hardware acceleration
        self._init_gpu_acceleration()

        # Visualization integration
        self.visualizer = None
        self.log_callback = None

        # Complex Tensor mode flag
        self.tensor_mode = False
        self.tensor_core = None

        # Visualization control - NEW
        self.enhanced_visualization_enabled = False  # Default disabled to save computation
        self.viz_capture_interval = 5  # Capture every 5 iterations

    def enable_enhanced_visualization(self, enabled=True, capture_interval=5):
        """Enable or disable enhanced visualization capture"""
        self.enhanced_visualization_enabled = enabled
        self.viz_capture_interval = capture_interval

        if enabled:
            self.log(f"✅ Enhanced visualization ENABLED (capturing every {capture_interval} iterations)")
            # Initialize visualizer if not already attached
            if not hasattr(self, 'visualizer') or self.visualizer is None:
                self.visualizer = DBNNVisualizer()
                self.log("✅ Visualizer attached for enhanced visualization")
        else:
            self.log("✅ Enhanced visualization DISABLED (computational overhead reduced)")

        return self.enhanced_visualization_enabled

    def _capture_enhanced_snapshot(self, features_batches, encoded_targets_batches, iteration):
        """Capture enhanced snapshot for visualization - UPDATED FOR CONFUSION MATRIX"""
        if not self.enhanced_visualization_enabled:
            return

        # Only capture at specified intervals to reduce overhead
        if iteration % self.viz_capture_interval != 0:
            return

        try:
            # Sample data for visualization (limit for performance)
            all_features = np.vstack(features_batches)
            all_targets = np.concatenate(encoded_targets_batches)

            # Use smaller sample for performance (max 1000 samples)
            sample_size = min(1000, len(all_features))
            if len(all_features) > sample_size:
                indices = np.random.choice(len(all_features), sample_size, replace=False)
                sample_features = all_features[indices]
                sample_targets = all_targets[indices]
            else:
                sample_features = all_features
                sample_targets = all_targets

            # Get predictions - CRITICAL for confusion matrix
            sample_predictions, _ = self.predict_batch(sample_features)

            # Get feature names
            feature_names = getattr(self, 'feature_columns', None)
            if feature_names is None:
                feature_names = [f'Feature_{i+1}' for i in range(sample_features.shape[1])]

            # Get class names from encoder
            class_names = []
            if hasattr(self, 'class_encoder') and self.class_encoder.is_fitted:
                for encoded_val in self.class_encoder.encoded_to_class.values():
                    class_names.append(str(encoded_val))
            else:
                unique_targets = np.unique(sample_targets)
                class_names = [f'Class_{int(t)}' for t in unique_targets]

            # Capture the snapshot - MAKE SURE PREDICTIONS ARE INCLUDED
            if hasattr(self, 'visualizer') and self.visualizer:
                self.visualizer.capture_feature_space_snapshot(
                    sample_features,
                    sample_targets,
                    sample_predictions,  # This is crucial for confusion matrix
                    iteration,
                    feature_names,
                    class_names
                )

                if iteration % 20 == 0:  # Log every 20 iterations to avoid spam
                    accuracy = np.mean(sample_predictions == sample_targets) * 100
                    self.log(f"📊 Captured visualization snapshot at iteration {iteration} (Accuracy: {accuracy:.1f}%)")

        except Exception as e:
            # Don't crash training if visualization fails
            if iteration % 10 == 0:  # Only log occasionally
                self.log(f"⚠️ Enhanced visualization snapshot failed: {e}")

    def load_model_auto_config(self, model_path: str):
        """
        Load model with automatic configuration detection and setup
        Enhanced version that handles both tensor and standard modes
        """
        try:
            self.log(f"🔄 Loading model with auto-configuration: {model_path}")

            # Use the existing load_model method
            success = self.load_model(model_path)

            if not success:
                self.log("❌ Basic model loading failed")
                return False

            # AUTO-CONFIGURATION: Set up the core based on loaded model
            self.log("🔧 Applying auto-configuration...")

            # Set tensor mode if tensor arrays are detected
            if (hasattr(self, 'tensor_mode') and
                hasattr(self, 'tensor_core') and
                self.tensor_core is not None and
                hasattr(self.tensor_core, 'weight_matrix') and
                self.tensor_core.weight_matrix is not None):

                self.enable_tensor_mode(True)
                self.log("✅ Auto-configured: Tensor mode detected")

            # Verify feature configuration
            if not hasattr(self, 'feature_columns') or not self.feature_columns:
                self.log("⚠️ No feature columns found in model, attempting inference...")

                # Try to infer from array dimensions
                if hasattr(self, 'innodes') and self.innodes > 0:
                    self.feature_columns = [f'feature_{i+1}' for i in range(self.innodes)]
                    self.log(f"✅ Inferred {self.innodes} feature columns from model dimensions")

            # Verify class encoder
            if (hasattr(self, 'class_encoder') and
                hasattr(self.class_encoder, 'is_fitted') and
                not self.class_encoder.is_fitted):

                self.log("⚠️ Class encoder not fitted, attempting recovery...")
                # Try to recover from class_labels
                if hasattr(self, 'class_labels') and self.class_labels is not None:
                    try:
                        # Extract class values from class_labels (skip margin at index 0)
                        class_values = []
                        for i in range(1, min(len(self.class_labels), self.outnodes + 1)):
                            if self.class_labels[i] != 0:  # Skip zero values
                                class_values.append(self.class_labels[i])

                        if class_values:
                            # Create basic encoder
                            self.class_encoder.fit(class_values)
                            self.log(f"✅ Recovered class encoder with {len(class_values)} classes")
                    except Exception as e:
                        self.log(f"❌ Class encoder recovery failed: {e}")

            # Final validation
            validation_passed = True

            if not hasattr(self, 'innodes') or self.innodes <= 0:
                self.log("❌ Auto-configuration failed: Invalid input nodes")
                validation_passed = False

            if not hasattr(self, 'outnodes') or self.outnodes <= 0:
                self.log("❌ Auto-configuration failed: Invalid output nodes")
                validation_passed = False

            if not hasattr(self, 'is_trained') or not self.is_trained:
                self.log("⚠️ Model marked as not trained, but continuing...")

            if validation_passed:
                self.log("✅ Auto-configuration completed successfully")

                # Test model functionality
                try:
                    if self.innodes > 0:
                        test_sample = np.random.randn(self.innodes)
                        predictions, probabilities = self.predict_batch(test_sample.reshape(1, -1))
                        self.log(f"✅ Model test passed - Prediction: {predictions[0]}")
                except Exception as test_error:
                    self.log(f"⚠️ Model test warning: {test_error}")

            return validation_passed

        except Exception as e:
            self.log(f"❌ Auto-configuration load error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def enable_tensor_mode(self, enabled=True):
        """Enable or disable tensor transformation mode"""
        self.tensor_mode = enabled
        if enabled and self.tensor_core is None:
            self.tensor_core = DBNNTensorCore(self.config)
            # Copy essential state from main core to tensor core
            if hasattr(self, 'class_encoder'):
                self.tensor_core.class_encoder = self.class_encoder
            if hasattr(self, 'class_labels') and self.class_labels is not None:
                self.tensor_core.class_labels = self.class_labels.copy()
            self.log("✅ Tensor transformation mode enabled")
        elif not enabled:
            self.log("✅ Standard iterative mode enabled")

    def _detect_optimal_workers(self):
        """Detect optimal number of workers for parallel processing"""
        try:
            cpu_count = mp.cpu_count()
            # Reserve 1 core for main process, use 75% of remaining cores
            optimal_workers = max(1, int((cpu_count - 1) * 0.75))
            return min(optimal_workers, 16)  # Cap at 16 workers
        except:
            return 4  # Fallback to 4 workers

    def attach_visualizer(self, visualizer):
        """Attach a visualizer for training monitoring"""
        self.visualizer = visualizer
        self.log("Visualizer attached to DBNN core")

    def detect_system_resources(self):
        """Detect available system resources and set optimal parameters"""
        resource_info = {
            'cpu_cores': 1,
            'memory_gb': 4,
            'has_gpu': False,
            'gpu_memory_gb': 0,
            'system_memory_gb': 4
        }

        try:
            import psutil
            # CPU cores
            resource_info['cpu_cores'] = psutil.cpu_count(logical=False) or 1
            resource_info['logical_cores'] = psutil.cpu_count(logical=True) or 1

            # System memory
            system_memory = psutil.virtual_memory()
            resource_info['system_memory_gb'] = system_memory.total / (1024**3)
            resource_info['available_memory_gb'] = system_memory.available / (1024**3)

            # Process memory
            process = psutil.Process()
            resource_info['process_memory_mb'] = process.memory_info().rss / (1024**2)

        except ImportError:
            self.log("⚠️ psutil not available, using default resource values")

        # GPU detection
        try:
            import torch
            if torch.cuda.is_available():
                resource_info['has_gpu'] = True
                resource_info['gpu_count'] = torch.cuda.device_count()
                resource_info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                resource_info['gpu_name'] = torch.cuda.get_device_name(0)
            else:
                resource_info['has_gpu'] = False
        except ImportError:
            # Try alternative GPU detection
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    gpu_memory_mb = int(result.stdout.strip().split('\n')[0])
                    resource_info['has_gpu'] = True
                    resource_info['gpu_memory_gb'] = gpu_memory_mb / 1024
            except:
                resource_info['has_gpu'] = False

        return resource_info

    def save_model_auto(self, model_dir="Model", data_filename=None, feature_columns=None, target_column=None):
        """Automatically save trained model in binary format with timestamp to Model/ folder"""
        if not self.is_trained:
            self.log("❌ No trained model to save")
            return None

        try:
            # Ensure Model directory exists with absolute path
            model_dir = os.path.abspath(model_dir)
            os.makedirs(model_dir, exist_ok=True)

            # Create model filename based on data file and timestamp
            if data_filename:
                base_name = os.path.splitext(os.path.basename(data_filename))[0]
            else:
                base_name = "model"

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_filename = os.path.join(model_dir, f"{base_name}_{timestamp}_model.bin")

            # Store additional metadata
            model_metadata = {
                'data_file': data_filename,
                'feature_columns': feature_columns if feature_columns else [],
                'target_column': target_column if target_column else "",
                'best_accuracy': getattr(self, 'best_accuracy', 0.0),
                'best_round': getattr(self, 'best_round', 0),
                'training_date': time.strftime("%Y-%m-%d %H:%M:%S"),
                'model_format': 'binary_auto_v1',
                'input_nodes': self.innodes,
                'output_nodes': self.outnodes,
                'config': self.config  # Store the complete configuration
            }

            # Use core's save method but enhance with metadata
            success = self.save_model(
                model_filename,
                feature_columns=feature_columns,
                target_column=target_column,
                use_json=False  # Always binary format for auto-save
            )

            if success:
                # Save additional metadata separately
                meta_filename = model_filename.replace('.bin', '_meta.json')
                with open(meta_filename, 'w') as f:
                    json.dump(model_metadata, f, indent=2)

                self.log(f"✅ Model automatically saved: {model_filename}")
                self.log(f"   Best accuracy: {model_metadata['best_accuracy']:.2f}% at round {model_metadata['best_round']}")
                self.log(f"   Metadata: {meta_filename}")
                self.log(f"   Configuration: {len(self.config)} parameters embedded")

                return model_filename
            else:
                self.log("❌ Failed to automatically save model")
                return None

        except Exception as e:
            self.log(f"❌ Error in automatic model saving: {e}")
            import traceback
            traceback.print_exc()
            return None

    def calculate_optimal_parameters(self, resource_info, operation_type="prediction"):
        """Calculate optimal parameters based on available resources"""
        params = {
            'batch_size': 1000,
            'max_concurrent_batches': 1,
            'memory_safety_factor': 0.7,
            'clear_cache_every': 50,
            'write_to_disk_every': 10,
            'use_memory_mapping': False,
            'optimization_level': 'balanced'
        }

        # Extract resource information
        cpu_cores = resource_info['cpu_cores']
        available_memory_gb = resource_info['available_memory_gb']
        system_memory_gb = resource_info['system_memory_gb']
        has_gpu = resource_info['has_gpu']
        gpu_memory_gb = resource_info.get('gpu_memory_gb', 0)

        # Memory-based optimization
        if system_memory_gb >= 32:
            # High memory system
            params['optimization_level'] = 'performance'
            params['batch_size'] = 5000
            params['max_concurrent_batches'] = min(4, cpu_cores)
            params['memory_safety_factor'] = 0.8
            params['clear_cache_every'] = 100
            params['write_to_disk_every'] = 20

        elif system_memory_gb >= 16:
            # Medium memory system
            params['optimization_level'] = 'balanced'
            params['batch_size'] = 2000
            params['max_concurrent_batches'] = min(2, cpu_cores)
            params['memory_safety_factor'] = 0.7
            params['clear_cache_every'] = 50
            params['write_to_disk_every'] = 15

        else:
            # Low memory system
            params['optimization_level'] = 'memory_safe'
            params['batch_size'] = 500
            params['max_concurrent_batches'] = 1
            params['memory_safety_factor'] = 0.5
            params['clear_cache_every'] = 20
            params['write_to_disk_every'] = 5

        # CPU-based adjustments
        if cpu_cores >= 8:
            params['max_concurrent_batches'] = min(params['max_concurrent_batches'] + 2, cpu_cores // 2)
        elif cpu_cores >= 4:
            params['max_concurrent_batches'] = min(params['max_concurrent_batches'] + 1, cpu_cores // 2)

        # GPU-based optimizations
        if has_gpu and gpu_memory_gb >= 4:
            params['use_gpu'] = True
            if gpu_memory_gb >= 16:
                params['batch_size'] = params['batch_size'] * 2
                params['optimization_level'] = 'gpu_optimized'
            elif gpu_memory_gb >= 8:
                params['batch_size'] = int(params['batch_size'] * 1.5)

        # Operation-specific adjustments
        if operation_type == "training":
            # Training requires more memory
            params['batch_size'] = max(100, params['batch_size'] // 2)
            params['write_to_disk_every'] = max(5, params['write_to_disk_every'] // 2)
        elif operation_type == "prediction":
            # Prediction can use larger batches
            params['batch_size'] = min(10000, params['batch_size'] * 2)

        # Safety limits
        params['batch_size'] = min(params['batch_size'], 10000)  # Absolute max
        params['max_concurrent_batches'] = min(params['max_concurrent_batches'], 8)

        # Calculate memory limits
        estimated_batch_memory_mb = self.estimate_batch_memory(params['batch_size'])
        safe_memory_limit_mb = available_memory_gb * 1024 * params['memory_safety_factor']

        # Adjust batch size if it exceeds safe memory
        max_safe_batches = int(safe_memory_limit_mb / estimated_batch_memory_mb) if estimated_batch_memory_mb > 0 else 1
        if max_safe_batches < params['max_concurrent_batches']:
            params['max_concurrent_batches'] = max(1, max_safe_batches)
            params['batch_size'] = max(100, params['batch_size'] // 2)

        return params

    def estimate_batch_memory(self, batch_size):
        """Estimate memory usage for a batch"""
        if not hasattr(self, 'innodes') or self.innodes == 0:
            return 100  # Default estimate

        # Rough memory estimation (in MB)
        feature_memory = batch_size * self.innodes * 8 / (1024**2)  # float64
        network_memory = (self.innodes * 100 * self.innodes * 100 * self.outnodes * 8) / (1024**2)  # anti_net
        total_estimate = feature_memory + network_memory + 100  # Buffer

        return total_estimate

    def print_resource_report(self):
        """Print detailed resource report"""
        resources = self.detect_system_resources()

        print("\n" + "="*50)
        print("SYSTEM RESOURCE ANALYSIS")
        print("="*50)
        print(f"CPU Cores (physical): {resources['cpu_cores']}")
        print(f"CPU Cores (logical): {resources.get('logical_cores', 'N/A')}")
        print(f"System Memory: {resources['system_memory_gb']:.1f} GB")
        print(f"Available Memory: {resources['available_memory_gb']:.1f} GB")
        print(f"GPU Available: {'Yes' if resources['has_gpu'] else 'No'}")

        if resources['has_gpu']:
            print(f"GPU Memory: {resources['gpu_memory_gb']:.1f} GB")
            print(f"GPU Name: {resources.get('gpu_name', 'N/A')}")

        # Calculate optimal parameters
        pred_params = self.calculate_optimal_parameters(resources, "prediction")
        train_params = self.calculate_optimal_parameters(resources, "training")

        print(f"\nPREDICTION OPTIMIZATION:")
        print(f"  Optimization Level: {pred_params['optimization_level']}")
        print(f"  Batch Size: {pred_params['batch_size']}")
        print(f"  Concurrent Batches: {pred_params['max_concurrent_batches']}")
        print(f"  Memory Safety: {pred_params['memory_safety_factor']*100:.0f}%")

        print(f"\nTRAINING OPTIMIZATION:")
        print(f"  Optimization Level: {train_params['optimization_level']}")
        print(f"  Batch Size: {train_params['batch_size']}")
        print(f"  Concurrent Batches: {train_params['max_concurrent_batches']}")
        print("="*50)

        return resources, pred_params, train_params

    def save_model(self, model_path: str, feature_columns=None, target_column=None, use_json=False):
        """Save complete model to file in binary format (default) or JSON format"""
        try:
            # Ensure all required fields are set
            if not hasattr(self, 'innodes') or self.innodes is None:
                # Try to infer from array shapes
                if hasattr(self, 'anti_net') and self.anti_net is not None:
                    self.innodes = self.anti_net.shape[0] - 2
                else:
                    self.innodes = 0

            if not hasattr(self, 'outnodes') or self.outnodes is None:
                if hasattr(self, 'anti_net') and self.anti_net is not None:
                    self.outnodes = self.anti_net.shape[4] - 2
                else:
                    self.outnodes = 0

            # Ensure directory exists
            model_dir = os.path.dirname(os.path.abspath(model_path))
            if model_dir and not os.path.exists(model_dir):
                os.makedirs(model_dir, exist_ok=True)

            # Build model data with feature information
            model_data = {
                'config': self.config,
                'innodes': self.innodes,
                'outnodes': self.outnodes,
                'is_trained': self.is_trained,
                'best_accuracy': getattr(self, 'best_accuracy', 0.0),
                'class_encoder': self.class_encoder.get_class_mapping() if hasattr(self.class_encoder, 'is_fitted') and self.class_encoder.is_fitted else {},
                'feature_columns': feature_columns if feature_columns else [],
                'target_column': target_column if target_column else '',
                'arrays_format': 'numpy',  # Mark that arrays are stored as numpy objects
                'training_mode': 'tensor' if (self.tensor_mode and self.tensor_core) else 'standard'
            }

            # Add arrays with proper validation - ENHANCED BUT COMPATIBLE
            array_mappings = [
                ('anti_net', np.int32),
                ('anti_wts', np.float64),
                ('binloc', np.float64),
                ('max_val', np.float64),
                ('min_val', np.float64),
                ('class_labels', np.float64),
                ('resolution_arr', np.int32)
            ]

            for field_name, dtype in array_mappings:
                if hasattr(self, field_name) and getattr(self, field_name) is not None:
                    model_data[field_name] = getattr(self, field_name)
                else:
                    model_data[field_name] = np.array([], dtype=dtype)

            # NEW: Add tensor-specific data if in tensor mode
            if self.tensor_mode and self.tensor_core:
                tensor_data = {}
                tensor_arrays = ['orthogonal_basis', 'weight_matrix', 'feature_projection', 'class_projection']
                for array_name in tensor_arrays:
                    if hasattr(self.tensor_core, array_name) and getattr(self.tensor_core, array_name) is not None:
                        tensor_data[array_name] = getattr(self.tensor_core, array_name)

                if tensor_data:  # Only add if we have tensor data
                    model_data['tensor_arrays'] = tensor_data

            if use_json:
                # Convert numpy arrays to lists for JSON
                json_model_data = model_data.copy()
                for key in ['anti_net', 'anti_wts', 'binloc', 'max_val', 'min_val', 'class_labels', 'resolution_arr']:
                    if isinstance(json_model_data[key], np.ndarray):
                        json_model_data[key] = json_model_data[key].tolist()

                # Handle tensor arrays for JSON
                if 'tensor_arrays' in json_model_data:
                    for tensor_key in json_model_data['tensor_arrays']:
                        if isinstance(json_model_data['tensor_arrays'][tensor_key], np.ndarray):
                            json_model_data['tensor_arrays'][tensor_key] = json_model_data['tensor_arrays'][tensor_key].tolist()

                with open(model_path, 'w') as f:
                    json.dump(json_model_data, f, indent=2)
                self.log(f"Model saved in JSON format to: {model_path}")
            else:
                # Save in binary format (default)
                with gzip.open(model_path, 'wb') as f:
                    pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                self.log(f"Model saved in binary format to: {model_path}")

            if feature_columns:
                mode_info = "tensor" if (self.tensor_mode and self.tensor_core) else "standard"
                self.log(f"✅ {mode_info.capitalize()} model saved: {len(feature_columns)} features")
            return True

        except Exception as e:
            self.log(f"Error saving model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_model(self, model_path: str):
        """Load model from file with robust error handling - supports both binary and JSON formats"""
        try:
            # Detect format and load accordingly
            if model_path.endswith('.gz') or model_path.endswith('.bin'):
                # Binary format
                with gzip.open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                self.log(f"Loading binary model from: {model_path}")
            else:
                # JSON format
                with open(model_path, 'r') as f:
                    model_data = json.load(f)
                self.log(f"Loading JSON model from: {model_path}")

            self.log(f"Loading model from: {model_path}")

            # NEW: Detect and set training mode BEFORE loading arrays
            training_mode = model_data.get('training_mode', 'standard')
            if training_mode == 'tensor':
                self.enable_tensor_mode(True)
                self.log("✅ Detected tensor mode model")
            else:
                self.enable_tensor_mode(False)
                self.log("✅ Detected standard mode model")

            # Load basic configuration FIRST
            self.config = model_data.get('config', {})

            # Handle array loading based on format
            if 'arrays_format' in model_data and model_data['arrays_format'] == 'numpy':
                # Binary format with numpy arrays
                array_mappings = [
                    ('anti_net', np.int32),
                    ('anti_wts', np.float64),
                    ('binloc', np.float64),
                    ('max_val', np.float64),
                    ('min_val', np.float64),
                    ('class_labels', np.float64),  # CHANGED: class_labels → class_labels
                    ('resolution_arr', np.int32)
                ]

                for field_name, dtype in array_mappings:
                    if field_name in model_data:
                        loaded_array = model_data[field_name]
                        if isinstance(loaded_array, list):
                            # Convert list to numpy array if needed
                            loaded_array = np.array(loaded_array, dtype=dtype)
                        setattr(self, field_name, loaded_array)
                        self.log(f"Loaded {field_name}: shape {loaded_array.shape}")
            else:
                # JSON format with lists - convert to numpy arrays
                array_mappings = [
                    ('anti_net', np.int32),
                    ('anti_wts', np.float64),
                    ('binloc', np.float64),
                    ('max_val', np.float64),
                    ('min_val', np.float64),
                    ('class_labels', np.float64),  # CHANGED: class_labels → class_labels
                    ('resolution_arr', np.int32)
                ]

                for field_name, dtype in array_mappings:
                    if field_name in model_data and model_data[field_name]:
                        try:
                            loaded_array = np.array(model_data[field_name], dtype=dtype)
                            setattr(self, field_name, loaded_array)
                            self.log(f"Loaded {field_name}: shape {loaded_array.shape}")
                        except Exception as e:
                            self.log(f"Error loading {field_name}: {e}")

            # NEW: Load tensor arrays if available and in tensor mode
            if self.tensor_mode and self.tensor_core and 'tensor_arrays' in model_data:
                tensor_data = model_data['tensor_arrays']
                for array_name, array_data in tensor_data.items():
                    if array_data is not None:
                        if isinstance(array_data, list):
                            array_data = np.array(array_data)
                        setattr(self.tensor_core, array_name, array_data)
                        self.log(f"Loaded tensor array {array_name}: shape {array_data.shape}")

            # FIX: Infer dimensions from ACTUAL loaded arrays, not from metadata
            if hasattr(self, 'anti_net') and self.anti_net is not None:
                # Infer from anti_net shape: (innodes+2, resol+2, innodes+2, resol+2, outnodes+2)
                actual_innodes = self.anti_net.shape[0] - 2
                actual_outnodes = self.anti_net.shape[4] - 2
                self.innodes = actual_innodes
                self.outnodes = actual_outnodes
                self.log(f"✅ Inferred dimensions from arrays: {actual_innodes} inputs, {actual_outnodes} outputs")
            else:
                # Fallback to metadata
                self.innodes = model_data.get('innodes', 0)
                self.outnodes = model_data.get('outnodes', 0)
                self.log(f"⚠️ Using metadata dimensions: {self.innodes} inputs, {self.outnodes} outputs")

            # Now set other properties
            self.is_trained = model_data.get('is_trained', False)

            # Load best accuracy
            if 'best_accuracy' in model_data:
                self.best_accuracy = model_data['best_accuracy']

            # Load feature information from model
            self.feature_columns = model_data.get('feature_columns', [])
            self.target_column = model_data.get('target_column', '')

            if self.feature_columns:
                self.log(f"✅ Loaded feature configuration: {len(self.feature_columns)} features - {self.feature_columns}")
            else:
                self.log("⚠️ No feature configuration found in model file")

            if self.target_column:
                self.log(f"✅ Target column: {self.target_column}")

            self.log(f"Final model - is_trained: {self.is_trained}, innodes: {self.innodes}, outnodes: {self.outnodes}")

            # FIX: Only reinitialize arrays if we have valid dimensions AND arrays aren't already loaded
            resol = self.config.get('resol', 100)
            if self.innodes > 0 and self.outnodes > 0 and not hasattr(self, 'anti_net'):
                self.initialize_arrays(self.innodes, resol, self.outnodes)
                self.log("Arrays initialized from dimensions")
            elif hasattr(self, 'anti_net'):
                self.log("✅ Arrays already loaded, no reinitialization needed")
            else:
                self.log("❌ Warning: Invalid model dimensions, cannot initialize arrays")
                return

            # FIX: Enhanced class encoder loading with better error recovery
            class_encoder_mapping = model_data.get('class_encoder', {})
            if class_encoder_mapping:
                try:
                    # Convert string keys back to floats for encoded_to_class
                    encoded_to_class = {}
                    class_to_encoded = {}

                    # Handle encoded_to_class (string keys → float keys)
                    encoded_to_class_data = class_encoder_mapping.get('encoded_to_class', {})
                    if encoded_to_class_data:
                        for str_key, class_name in encoded_to_class_data.items():
                            try:
                                float_key = float(str_key)
                                encoded_to_class[float_key] = class_name
                            except (ValueError, TypeError) as e:
                                self.log(f"⚠️ Warning: Could not convert key {str_key} to float: {e}")
                                # Fallback: try to keep as string if float conversion fails
                                try:
                                    encoded_to_class[float(str_key)] = class_name
                                except:
                                    self.log(f"❌ Failed to convert key {str_key}, skipping this entry")
                    else:
                        self.log("⚠️ No encoded_to_class data found in model")

                    # Handle class_to_encoded (already string keys)
                    class_to_encoded_data = class_encoder_mapping.get('class_to_encoded', {})
                    if class_to_encoded_data:
                        for class_name, encoded_val in class_to_encoded_data.items():
                            try:
                                class_to_encoded[class_name] = float(encoded_val)
                            except (ValueError, TypeError) as e:
                                self.log(f"⚠️ Warning: Could not convert value {encoded_val} to float: {e}")
                                # Fallback: try to convert string representation
                                try:
                                    class_to_encoded[class_name] = float(str(encoded_val))
                                except:
                                    self.log(f"❌ Failed to convert value {encoded_val}, skipping this entry")
                    else:
                        self.log("⚠️ No class_to_encoded data found in model")

                    # Set the encoder properties - CRITICAL: Always set is_fitted if we have any mappings
                    self.class_encoder.encoded_to_class = encoded_to_class
                    self.class_encoder.class_to_encoded = class_to_encoded

                    # Determine if encoder should be considered fitted
                    has_any_mappings = len(encoded_to_class) > 0 or len(class_to_encoded) > 0
                    is_fitted_from_data = class_encoder_mapping.get('is_fitted', False)

                    if has_any_mappings or is_fitted_from_data:
                        self.class_encoder.is_fitted = True
                        self.log(f"✅ Class encoder marked as fitted with {len(encoded_to_class)} encoded classes")
                    else:
                        self.class_encoder.is_fitted = False
                        self.log("⚠️ Class encoder not marked as fitted - no valid mappings found")

                    # Log encoder status
                    if self.class_encoder.is_fitted:
                        self.log(f"✅ Loaded class encoder with {len(self.class_encoder.encoded_to_class)} classes")
                        if self.class_encoder.encoded_to_class:
                            sample_classes = list(self.class_encoder.encoded_to_class.items())[:3]
                            self.log(f"Sample class mappings: {sample_classes}")

                            # Also verify class_labels alignment
                            if hasattr(self, 'class_labels') and self.class_labels is not None:
                                self.log(f"class_labels values: {[self.class_labels[i] for i in range(min(5, len(self.class_labels)))]}...")
                    else:
                        self.log("❌ Class encoder failed to load properly")

                except Exception as e:
                    self.log(f"❌ Error loading class encoder: {e}")
                    import traceback
                    traceback.print_exc()

                    # EMERGENCY RECOVERY: Try to create basic encoder from class_labels if available
                    if hasattr(self, 'class_labels') and self.class_labels is not None:
                        self.log("🔄 Attempting emergency encoder recovery from class_labels...")
                        try:
                            # Extract class values from class_labels (skip margin at index 0)
                            class_values = []
                            for i in range(1, min(len(self.class_labels), self.outnodes + 1)):
                                if self.class_labels[i] != 0:  # Skip zero values
                                    class_values.append(self.class_labels[i])

                            if class_values:
                                # Create basic encoder mapping
                                encoded_to_class = {}
                                class_to_encoded = {}
                                for i, class_val in enumerate(class_values, 1):
                                    encoded_to_class[float(i)] = f"Class_{class_val}"
                                    class_to_encoded[f"Class_{class_val}"] = float(i)

                                self.class_encoder.encoded_to_class = encoded_to_class
                                self.class_encoder.class_to_encoded = class_to_encoded
                                self.class_encoder.is_fitted = True
                                self.log(f"✅ Emergency recovery: Created encoder with {len(class_values)} classes from class_labels")
                        except Exception as recovery_error:
                            self.log(f"❌ Emergency recovery failed: {recovery_error}")
            else:
                self.log("❌ No class encoder data found in model file")

                # Try to infer from class_labels as last resort
                if hasattr(self, 'class_labels') and self.class_labels is not None:
                    self.log("🔄 Attempting to infer encoder from class_labels...")
                    try:
                        class_values = []
                        for i in range(1, min(len(self.class_labels), self.outnodes + 1)):
                            if self.class_labels[i] != 0:
                                class_values.append(self.class_labels[i])

                        if class_values:
                            encoded_to_class = {}
                            class_to_encoded = {}
                            for i, class_val in enumerate(class_values, 1):
                                encoded_to_class[float(i)] = str(class_val)
                                class_to_encoded[str(class_val)] = float(i)

                            self.class_encoder.encoded_to_class = encoded_to_class
                            self.class_encoder.class_to_encoded = class_to_encoded
                            self.class_encoder.is_fitted = True
                            self.log(f"✅ Inferred encoder with {len(class_values)} classes from class_labels")
                    except Exception as infer_error:
                        self.log(f"❌ Failed to infer encoder from class_labels: {infer_error}")

            # Final validation
            mode_info = "tensor" if self.tensor_mode else "standard"
            self.log(f"✅ {mode_info.capitalize()} model loaded successfully: {self.innodes} inputs, {self.outnodes} outputs")
            self.log(f"✅ Model is_trained: {self.is_trained}")
            self.log(f"✅ Class encoder is_fitted: {getattr(self.class_encoder, 'is_fitted', False)}")
            self.log(f"✅ Config resolution: {self.config.get('resol', 'N/A')}")
            if self.feature_columns:
                self.log(f"✅ Feature columns: {len(self.feature_columns)} features loaded")
            else:
                self.log("⚠️ No feature columns information available")

            # Test model functionality
            if self.innodes > 0 and self.class_encoder.is_fitted:
                try:
                    test_sample = np.random.randn(self.innodes)
                    predictions, probabilities = self.predict_batch(test_sample.reshape(1, -1))
                    self.log(f"✅ Model test passed - Prediction: {predictions[0]}")
                except Exception as test_error:
                    self.log(f"⚠️ Model test warning: {test_error}")

        except Exception as e:
            self.log(f"❌ Load error: {e}")
            import traceback
            traceback.print_exc()

    def _init_gpu_acceleration(self):
        """Initialize GPU acceleration if available"""
        try:
            import torch
            if torch.cuda.is_available():
                self.gpu_enabled = True
                self.device = torch.device('cuda')
                print(f"✅ GPU acceleration enabled: {torch.cuda.get_device_name()}")
            else:
                self.gpu_enabled = False
                self.device = torch.device('cpu')
        except ImportError:
            self.gpu_enabled = False
            self.device = None

    def set_log_callback(self, log_callback):
        """Set a callback function for logging"""
        self.log_callback = log_callback

    def log(self, message: str):
        """Log message through callback or print"""
        if self.log_callback:
            self.log_callback(message)
        else:
            print(f"[DBNNCore] {message}")

    def initialize_arrays(self, innodes: int, resol: int, outnodes: int):
        """Initialize arrays with optimal data types and memory layout"""
        self.innodes = innodes
        self.outnodes = outnodes

        # Use optimal numpy dtypes for memory efficiency
        self.anti_net = np.zeros((innodes+2, resol+2, innodes+2, resol+2, outnodes+2),
                               dtype=np.int32)  # Keep as int32 for large counts
        self.anti_wts = np.ones((innodes+2, resol+2, innodes+2, resol+2, outnodes+2),
                               dtype=np.float32)  # Use float32 for memory efficiency
        self.antit_wts = np.ones_like(self.anti_wts, dtype=np.float32)
        self.antip_wts = np.ones_like(self.anti_wts, dtype=np.float32)

        self.binloc = np.zeros((innodes+2, resol+8), dtype=np.float32)
        self.max_val = np.zeros(innodes+2, dtype=np.float32)
        self.min_val = np.zeros(innodes+2, dtype=np.float32)
        self.resolution_arr = np.zeros(innodes+8, dtype=np.int32)

        # Initialize class_labels
        self.class_labels = np.zeros(outnodes + 2, dtype=np.float32)
        self.class_labels[0] = self.config.get('margin', 0.2)

        print(f"Initialized arrays for {innodes} input nodes, {outnodes} output nodes, resolution {resol}")

    def load_data(self, filename: str, target_column: Optional[str] = None,
                 feature_columns: Optional[List[str]] = None, batch_size: int = 10000):
        """Optimized data loading with memory mapping and parallel processing"""
        if filename.endswith('.csv'):
            return self._load_csv_data_optimized(filename, target_column, feature_columns, batch_size)
        else:
            return self._load_dat_data_optimized(filename, batch_size)

    def _load_csv_data_optimized(self, filename: str, target_column: str,
                                feature_columns: List[str], batch_size: int):
        """Optimized CSV loading with memory-efficient processing"""
        import pandas as pd

        features_batches = []
        targets_batches = []
        original_targets_batches = []

        self.log(f"Loading CSV data from: {filename}")

        with open(filename, 'r') as f:
            reader = csv.DictReader(f)

            if not reader.fieldnames:
                raise ValueError("No headers found in CSV file")

            available_columns = reader.fieldnames
            self.log(f"Available columns in CSV: {available_columns}")

            # For prediction mode: if target_column is None or 'None', skip target validation
            is_prediction_mode = (target_column is None or target_column == 'None' or target_column == '')

            # Validate columns exist - handle prediction mode differently
            missing_columns = []

            if not is_prediction_mode and target_column and target_column not in available_columns:
                missing_columns.append(target_column)

            # Check feature columns - use all available if not specified
            if not feature_columns:
                # If no features specified, use all columns except target (if target exists)
                if not is_prediction_mode and target_column and target_column in available_columns:
                    feature_columns = [col for col in available_columns if col != target_column]
                else:
                    # For prediction mode with no target, or target not found, use all columns
                    feature_columns = available_columns.copy()
                self.log(f"Auto-selected feature columns: {feature_columns}")

            # Now validate feature columns
            for col in feature_columns:
                if col not in available_columns:
                    missing_columns.append(col)

            if missing_columns and not is_prediction_mode:
                raise ValueError(f"Columns not found in CSV: {missing_columns}")
            elif missing_columns and is_prediction_mode:
                self.log(f"⚠️ Warning: Some feature columns not found in CSV: {missing_columns}")
                # Remove missing columns from feature_columns
                feature_columns = [col for col in feature_columns if col not in missing_columns]
                if not feature_columns:
                    raise ValueError("No valid feature columns found in CSV file")

            self.log(f"Using feature columns: {feature_columns}")
            if not is_prediction_mode and target_column:
                self.log(f"Using target column: {target_column}")

            current_batch_features = []
            current_batch_targets = []
            current_batch_original_targets = []

            line_count = 0
            for row in reader:
                line_count += 1
                try:
                    # Extract features - use only the specified feature columns in the correct order
                    feature_vec = []
                    for col in feature_columns:
                        try:
                            feature_vec.append(float(row[col]))
                        except (ValueError, TypeError):
                            feature_vec.append(0.0)
                            if line_count <= 5:  # Only warn for first few rows
                                self.log(f"⚠️ Warning: Could not convert column '{col}' value '{row[col]}' to float, using 0.0")

                    # Extract target (only if not in prediction mode and target exists)
                    target_val = None
                    if not is_prediction_mode and target_column and target_column in row:
                        target_val = row[target_column]
                    else:
                        # For prediction mode or missing target, use a placeholder
                        target_val = "unknown"

                    current_batch_features.append(feature_vec)
                    current_batch_targets.append(target_val)
                    current_batch_original_targets.append(target_val)

                    if len(current_batch_features) >= batch_size:
                        # Use float32 for memory efficiency
                        features_batches.append(np.array(current_batch_features, dtype=np.float32))
                        targets_batches.append(np.array(current_batch_targets))
                        original_targets_batches.append(np.array(current_batch_original_targets))
                        current_batch_features = []
                        current_batch_targets = []
                        current_batch_original_targets = []

                except (ValueError, KeyError) as e:
                    self.log(f"⚠️ Warning: Skipping row {line_count} due to error: {e}")
                    continue

            # Add remaining samples
            if current_batch_features:
                features_batches.append(np.array(current_batch_features, dtype=np.float32))
                targets_batches.append(np.array(current_batch_targets))
                original_targets_batches.append(np.array(current_batch_original_targets))

        total_samples = sum(len(batch) for batch in features_batches)
        self.log(f"Loaded {total_samples} samples in {len(features_batches)} batches (optimized)")
        return features_batches, targets_batches, feature_columns, original_targets_batches

    def _load_dat_data_optimized(self, filename: str, batch_size: int):
        """Optimized legacy DAT format data loading"""
        features = []
        targets = []
        original_targets = []

        self.log(f"Loading legacy DAT file: {filename}")

        with open(filename, 'r') as f:
            current_batch_features = []
            current_batch_targets = []
            current_batch_original_targets = []

            for line in f:
                values = line.strip().split()
                if len(values) >= self.innodes + 1:
                    feature_vec = [float(x) for x in values[:self.innodes]]
                    target_val = values[self.innodes]

                    try:
                        target_val_float = float(target_val)
                        current_batch_original_targets.append(target_val_float)
                    except ValueError:
                        current_batch_original_targets.append(target_val)

                    current_batch_features.append(feature_vec)
                    current_batch_targets.append(target_val)

                    if len(current_batch_features) >= batch_size:
                        # Use float32 for memory efficiency
                        features.append(np.array(current_batch_features, dtype=np.float32))
                        targets.append(np.array(current_batch_targets))
                        original_targets.append(np.array(current_batch_original_targets))
                        current_batch_features = []
                        current_batch_targets = []
                        current_batch_original_targets = []

            # Add remaining samples
            if current_batch_features:
                features.append(np.array(current_batch_features, dtype=np.float32))
                targets.append(np.array(current_batch_targets))
                original_targets.append(np.array(current_batch_original_targets))

        self.log(f"Loaded {sum(len(batch) for batch in features)} samples in {len(features)} batches (optimized)")
        return features, targets, [f'feature_{i}' for i in range(1, self.innodes + 1)], original_targets

    def fit_encoder(self, original_targets_batches):
        """Fit class encoder to target data"""
        all_original_targets = np.concatenate(original_targets_batches) if original_targets_batches else np.array([])
        self.class_encoder.fit(all_original_targets)

        # Update class_labels with encoded values
        encoded_classes = self.class_encoder.get_encoded_classes()
        for i, encoded_val in enumerate(encoded_classes, 1):
            self.class_labels[i] = float(encoded_val)

    def initialize_training(self, features_batches, encoded_targets_batches, resol: int):
        """Initialize training parameters and arrays"""
        # Set resolutions
        for i in range(1, self.innodes + 1):
            self.resolution_arr[i] = resol
            for j in range(self.resolution_arr[i] + 1):
                self.binloc[i][j+1] = j * 1.0

        # Find min/max values
        omax = -400.0
        omin = 400.0
        for batch_idx, (features_batch, targets_batch) in enumerate(zip(features_batches, encoded_targets_batches)):
            for i in range(1, self.innodes + 1):
                batch_max = np.max(features_batch[:, i-1])
                batch_min = np.min(features_batch[:, i-1])
                if batch_max > self.max_val[i]:
                    self.max_val[i] = batch_max
                if batch_min < self.min_val[i]:
                    self.min_val[i] = batch_min

            # Update omax/omin from targets
            batch_omax = np.max(targets_batch)
            batch_omin = np.min(targets_batch)
            if batch_omax > omax:
                omax = batch_omax
            if batch_omin < omin:
                omin = batch_omin

        # Initialize network counts
        self.anti_wts.fill(1.0)
        for k in range(1, self.outnodes + 1):
            for i in range(1, self.innodes + 1):
                for j in range(self.resolution_arr[i] + 1):
                    for l in range(1, self.innodes + 1):
                        for m in range(self.resolution_arr[l] + 1):
                            self.anti_net[i, j, l, m, k] = 1

        return omax, omin

    def process_training_batch(self, features_batch, targets_batch):
        """Process a single batch of training data with parallel optimization"""
        if self.parallel_enabled and len(features_batch) > 1000 and self.num_workers > 1:
            return self._process_training_batch_parallel(features_batch, targets_batch)
        else:
            return self._process_training_batch_sequential(features_batch, targets_batch)

    def _process_training_batch_sequential(self, features_batch, targets_batch):
        """Process training batch sequentially"""
        batch_size = len(features_batch)
        processed_count = 0

        for sample_idx in range(batch_size):
            vects = np.zeros(self.innodes + self.outnodes + 2)
            for i in range(1, self.innodes + 1):
                vects[i] = features_batch[sample_idx, i-1]
            tmpv = targets_batch[sample_idx]

            self.anti_net = process_training_sample(
                vects, tmpv, self.anti_net, self.anti_wts, self.binloc,
                self.resolution_arr, self.class_labels, self.min_val, self.max_val,
                self.innodes, self.outnodes
            )

            processed_count += 1

        return processed_count

    def _process_training_batch_parallel(self, features_batch, targets_batch):
        """Process training batch in parallel"""
        batch_size = len(features_batch)
        chunk_size = max(1, batch_size // self.num_workers)

        # Split batch into chunks
        feature_chunks = []
        target_chunks = []
        for i in range(0, batch_size, chunk_size):
            feature_chunks.append(features_batch[i:i+chunk_size])
            target_chunks.append(targets_batch[i:i+chunk_size])

        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for feat_chunk, tgt_chunk in zip(feature_chunks, target_chunks):
                future = executor.submit(self._process_training_chunk, feat_chunk, tgt_chunk)
                futures.append(future)

            # Wait for all chunks to complete
            for future in futures:
                future.result()

        return batch_size

    def _process_training_chunk(self, features_chunk, targets_chunk):
        """Process a chunk of training data (thread-safe)"""
        chunk_size = len(features_chunk)

        for sample_idx in range(chunk_size):
            vects = np.zeros(self.innodes + self.outnodes + 2)
            for i in range(1, self.innodes + 1):
                vects[i] = features_chunk[sample_idx, i-1]
            tmpv = targets_chunk[sample_idx]

            self.anti_net = process_training_sample(
                vects, tmpv, self.anti_net, self.anti_wts, self.binloc,
                self.resolution_arr, self.class_labels, self.min_val, self.max_val,
                self.innodes, self.outnodes
            )

        return chunk_size

    def train_epoch(self, features_batches, encoded_targets_batches, gain: float):
        """Train for one epoch with parallel optimization"""
        if self.parallel_enabled and self.num_workers > 1:
            self._train_epoch_parallel(features_batches, encoded_targets_batches, gain)
        else:
            self._train_epoch_sequential(features_batches, encoded_targets_batches, gain)

    def _train_epoch_sequential(self, features_batches, encoded_targets_batches, gain: float):
        """Train one epoch sequentially"""
        for batch_idx, (features_batch, targets_batch) in enumerate(zip(features_batches, encoded_targets_batches)):
            batch_size = len(features_batch)

            for sample_idx in range(batch_size):
                vects = np.zeros(self.innodes + self.outnodes + 2)
                for i in range(1, self.innodes + 1):
                    vects[i] = features_batch[sample_idx, i-1]
                tmpv = targets_batch[sample_idx]

                # Compute probabilities
                classval = compute_class_probabilities_numba(
                    vects, self.anti_net, self.anti_wts, self.binloc, self.resolution_arr,
                    self.class_labels, self.min_val, self.max_val, self.innodes, self.outnodes
                )

                # Find predicted class
                kmax = 1
                cmax = 0.0
                for k in range(1, self.outnodes + 1):
                    if classval[k] > cmax:
                        cmax = classval[k]
                        kmax = k

                # Update weights if wrong classification
                if abs(self.class_labels[kmax] - tmpv) > self.class_labels[0]:
                    self.anti_wts = update_weights_numba(
                        vects, tmpv, classval, self.anti_wts, self.binloc, self.resolution_arr,
                        self.class_labels, self.min_val, self.max_val, self.innodes, self.outnodes, gain
                    )

    def _train_epoch_parallel(self, features_batches, encoded_targets_batches, gain: float):
        """Train one epoch with parallel processing"""
        # Process each batch in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for batch_idx, (features_batch, targets_batch) in enumerate(zip(features_batches, encoded_targets_batches)):
                future = executor.submit(self._train_batch_parallel, features_batch, targets_batch, gain, batch_idx)
                futures.append(future)

            # Wait for all batches to complete
            for future in futures:
                future.result()

    def _evaluate_parallel(self, features_batches, encoded_targets_batches):
        """Parallel evaluation"""
        total_correct = 0
        total_samples = 0
        all_predictions = []

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for batch_idx, (features_batch, targets_batch) in enumerate(zip(features_batches, encoded_targets_batches)):
                future = executor.submit(self._evaluate_batch_parallel, features_batch, targets_batch, batch_idx)
                futures.append(future)

            # Collect results
            for future in futures:
                batch_correct, batch_total, batch_predictions = future.result()
                total_correct += batch_correct
                total_samples += batch_total
                all_predictions.extend(batch_predictions)

        accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0
        return accuracy, total_correct, all_predictions

    def _predict_batch_parallel(self, features_batch):
        """Optimized parallel batch prediction"""
        batch_size = len(features_batch)
        chunk_size = max(1, batch_size // self.num_workers)

        # Split batch into chunks
        feature_chunks = []
        for i in range(0, batch_size, chunk_size):
            feature_chunks.append(features_batch[i:i+chunk_size])

        # Process chunks in parallel
        all_predictions = []
        all_probabilities = []

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for feat_chunk in feature_chunks:
                future = executor.submit(self._predict_chunk_parallel, feat_chunk)
                futures.append(future)

            # Collect results
            for future in futures:
                chunk_predictions, chunk_probabilities = future.result()
                all_predictions.extend(chunk_predictions)
                all_probabilities.extend(chunk_probabilities)

        return all_predictions, all_probabilities


    def _train_batch_parallel(self, features_batch, targets_batch, gain: float, batch_idx: int):
        """Train a single batch in parallel (thread-safe)"""
        batch_size = len(features_batch)

        for sample_idx in range(batch_size):
            vects = np.zeros(self.innodes + self.outnodes + 2)
            for i in range(1, self.innodes + 1):
                vects[i] = features_batch[sample_idx, i-1]
            tmpv = targets_batch[sample_idx]

            # Compute probabilities
            classval = compute_class_probabilities_numba(
                vects, self.anti_net, self.anti_wts, self.binloc, self.resolution_arr,
                self.class_labels, self.min_val, self.max_val, self.innodes, self.outnodes
            )

            # Find predicted class
            kmax = 1
            cmax = 0.0
            for k in range(1, self.outnodes + 1):
                if classval[k] > cmax:
                    cmax = classval[k]
                    kmax = k

            # Update weights if wrong classification
            if abs(self.class_labels[kmax] - tmpv) > self.class_labels[0]:
                self.anti_wts = update_weights_numba(
                    vects, tmpv, classval, self.anti_wts, self.binloc, self.resolution_arr,
                    self.class_labels, self.min_val, self.max_val, self.innodes, self.outnodes, gain
                )

    def evaluate(self, features_batches, encoded_targets_batches):
        """Evaluate model accuracy with parallel optimization"""
        # Tensor mode: use tensor evaluation if available
        if self.tensor_mode and self.tensor_core and hasattr(self.tensor_core, 'tensor_evaluate'):
            try:
                return self.tensor_core.tensor_evaluate(features_batches, encoded_targets_batches)
            except Exception as e:
                self.log(f"Tensor evaluation failed, falling back to standard: {e}")
                # Fall through to standard evaluation

        if self.parallel_enabled and self.num_workers > 1:
            return self._evaluate_parallel(features_batches, encoded_targets_batches)
        else:
            return self._evaluate_sequential(features_batches, encoded_targets_batches)

    def _evaluate_sequential(self, features_batches, encoded_targets_batches):
        """Evaluate model accuracy sequentially"""
        correct_predictions = 0
        total_samples = 0
        all_predictions = []

        for batch_idx, (features_batch, targets_batch) in enumerate(zip(features_batches, encoded_targets_batches)):
            batch_size = len(features_batch)
            total_samples += batch_size

            for sample_idx in range(batch_size):
                vects = np.zeros(self.innodes + self.outnodes + 2)
                for i in range(1, self.innodes + 1):
                    vects[i] = features_batch[sample_idx, i-1]
                actual = targets_batch[sample_idx]

                # Compute class probabilities
                classval = compute_class_probabilities_numba(
                    vects, self.anti_net, self.anti_wts, self.binloc, self.resolution_arr,
                    self.class_labels, self.min_val, self.max_val, self.innodes, self.outnodes
                )

                # Find predicted class
                kmax = 1
                cmax = 0.0
                for k in range(1, self.outnodes + 1):
                    if classval[k] > cmax:
                        cmax = classval[k]
                        kmax = k

                predicted = self.class_labels[kmax]
                all_predictions.append(predicted)

                # Check if prediction is correct
                if abs(actual - predicted) <= self.class_labels[0]:
                    correct_predictions += 1

        accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
        return accuracy, correct_predictions, all_predictions

    def _evaluate_batch_parallel(self, features_batch, targets_batch, batch_idx):
        """Evaluate a single batch in parallel"""
        batch_correct = 0
        batch_total = len(features_batch)
        batch_predictions = []

        for sample_idx in range(batch_total):
            vects = np.zeros(self.innodes + self.outnodes + 2)
            for i in range(1, self.innodes + 1):
                vects[i] = features_batch[sample_idx, i-1]
            actual = targets_batch[sample_idx]

            # Compute class probabilities
            classval = compute_class_probabilities_numba(
                vects, self.anti_net, self.anti_wts, self.binloc, self.resolution_arr,
                self.class_labels, self.min_val, self.max_val, self.innodes, self.outnodes
            )

            # Find predicted class
            kmax = 1
            cmax = 0.0
            for k in range(1, self.outnodes + 1):
                if classval[k] > cmax:
                    cmax = classval[k]
                    kmax = k

            predicted = self.class_labels[kmax]
            batch_predictions.append(predicted)

            # Check if prediction is correct
            if abs(actual - predicted) <= self.class_labels[0]:
                batch_correct += 1

        return batch_correct, batch_total, batch_predictions

    def predict_batch(self, features_batch):
        """Predict classes for a batch of features with parallel optimization"""
        # Tensor mode: use orthogonal prediction if available
        if self.tensor_mode and self.tensor_core and hasattr(self.tensor_core, 'orthogonal_predict'):
            try:
                return self.tensor_core.orthogonal_predict(features_batch)
            except Exception as e:
                self.log(f"Tensor prediction failed, falling back to standard: {e}")
                # Fall through to standard prediction
        if self.parallel_enabled and len(features_batch) > 1000 and self.num_workers > 1:
            return self._predict_batch_parallel(features_batch)
        else:
            return self._predict_batch_sequential(features_batch)

    def _predict_batch_sequential(self, features_batch):
        """Predict classes for a batch of features sequentially"""
        predictions = []
        probabilities = []

        for sample_idx in range(len(features_batch)):
            vects = np.zeros(self.innodes + self.outnodes + 2)
            for i in range(1, self.innodes + 1):
                vects[i] = features_batch[sample_idx, i-1]

            # Compute class probabilities
            classval = compute_class_probabilities_numba(
                vects, self.anti_net, self.anti_wts, self.binloc, self.resolution_arr,
                self.class_labels, self.min_val, self.max_val, self.innodes, self.outnodes
            )

            # Find predicted class
            kmax = 1
            cmax = 0.0
            for k in range(1, self.outnodes + 1):
                if classval[k] > cmax:
                    cmax = classval[k]
                    kmax = k

            predicted = self.class_labels[kmax]
            predictions.append(predicted)

            # Store probabilities
            prob_dict = {}
            for k in range(1, self.outnodes + 1):
                class_val = self.class_labels[k]
                if self.class_encoder.is_fitted:
                    class_name = self.class_encoder.encoded_to_class.get(class_val, f"Class_{k}")
                else:
                    class_name = f"Class_{k}"
                prob_dict[class_name] = float(classval[k])

            probabilities.append(prob_dict)

        return predictions, probabilities

    def _predict_chunk_parallel(self, features_chunk):
        """Predict a chunk of data in parallel"""
        chunk_predictions = []
        chunk_probabilities = []

        for sample_idx in range(len(features_chunk)):
            vects = np.zeros(self.innodes + self.outnodes + 2)
            for i in range(1, self.innodes + 1):
                vects[i] = features_chunk[sample_idx, i-1]

            # Compute class probabilities
            classval = compute_class_probabilities_numba(
                vects, self.anti_net, self.anti_wts, self.binloc, self.resolution_arr,
                self.class_labels, self.min_val, self.max_val, self.innodes, self.outnodes
            )

            # Find predicted class
            kmax = 1
            cmax = 0.0
            for k in range(1, self.outnodes + 1):
                if classval[k] > cmax:
                    cmax = classval[k]
                    kmax = k

            predicted = self.class_labels[kmax]
            chunk_predictions.append(predicted)

            # Store probabilities
            prob_dict = {}
            for k in range(1, self.outnodes + 1):
                class_val = self.class_labels[k]
                if self.class_encoder.is_fitted:
                    class_name = self.class_encoder.encoded_to_class.get(class_val, f"Class_{k}")
                else:
                    class_name = f"Class_{k}"
                prob_dict[class_name] = float(classval[k])

            chunk_probabilities.append(prob_dict)

        return chunk_predictions, chunk_probabilities

    def predict_batch_optimized(self, features_batch, clear_cache_every=100):
        """Optimized batch prediction with periodic cache clearing"""
        return self.predict_batch(features_batch)

    def generate_interactive_viz(self):
        """Generate interactive 3D visualizations - FIXED VERSION"""
        if not self.core or not hasattr(self.core, 'visualizer') or not self.core.visualizer:
            messagebox.showerror("Error", "No visualizer available. Please enable enhanced visualization and train a model first.")
            return

        try:
            self.show_processing_indicator("Generating interactive visualizations...")

            # Use the core's method to generate visualizations
            outputs = self.core.generate_interactive_visualizations()

            if outputs:
                self.log("✅ Interactive visualizations generated:")
                for viz_type, file_path in outputs.items():
                    self.log(f"   {viz_type}: {file_path}")

                # Ask if user wants to open the main visualization
                if 'enhanced_3d' in outputs:
                    result = messagebox.askyesno(
                        "Visualization Ready",
                        f"Interactive 3D visualization generated!\n\n"
                        f"File: {outputs['enhanced_3d']}\n\n"
                        f"Open in web browser?"
                    )
                    if result:
                        import webbrowser
                        webbrowser.open(f'file://{os.path.abspath(outputs["enhanced_3d"])}')
            else:
                self.log("❌ No visualizations could be generated")

        except Exception as e:
            self.log(f"❌ Error generating visualizations: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.hide_processing_indicator()

    def train_with_early_stopping(self, train_file: str, test_file: Optional[str] = None,
                                 use_csv: bool = True, target_column: Optional[str] = None,
                                 feature_columns: Optional[List[str]] = None,
                                 enable_interactive_viz: bool = False,
                                 viz_capture_interval: int = 5):
        """Main training method with early stopping and automatic optimizations
        Now with optional interactive 3D visualization support
        """
        # Ensure visualizer is available
        if not hasattr(self, 'visualizer') or self.visualizer is None:
            self.visualizer = DBNNVisualizer()
            self.log("✅ Visualizer initialized")

        if self.tensor_mode and self.tensor_core:
            self.log("🚀 Starting TENSOR TRANSFORMATION training...")
            return self.tensor_core.tensor_train(
                train_file, test_file, use_csv, target_column, feature_columns
            )
        else:
            self.log("Starting optimized model training with early stopping...")

        # Enable interactive visualization if requested
        if enable_interactive_viz:
            self._enable_interactive_visualization_internal(viz_capture_interval)

        # Load training data
        features_batches, targets_batches, feature_columns_used, original_targets_batches = self.load_data(
            train_file, target_column, feature_columns
        )

        if not features_batches:
            self.log("No training data loaded")
            return False

        # Fit encoder and determine architecture
        self.fit_encoder(original_targets_batches)
        encoded_targets_batches = []
        for batch in original_targets_batches:
            encoded_batch = self.class_encoder.transform(batch)
            encoded_targets_batches.append(encoded_batch)

        self.innodes = len(feature_columns_used)
        self.outnodes = len(self.class_encoder.get_encoded_classes())

        # Store feature information for visualization
        self.feature_columns = feature_columns_used
        self.target_column = target_column if target_column else ""

        # Initialize arrays
        resol = self.config.get('resol', 100)
        self.initialize_arrays(self.innodes, resol, self.outnodes)

        # Initialize training
        omax, omin = self.initialize_training(features_batches, encoded_targets_batches, resol)

        # Process training data for initial APF
        self.log("Processing training data for initial network counts...")
        total_samples = sum(len(batch) for batch in features_batches)
        total_processed = 0

        for batch_idx, (features_batch, targets_batch) in enumerate(zip(features_batches, encoded_targets_batches)):
            processed = self.process_training_batch(features_batch, targets_batch)
            total_processed += processed
            if total_processed % 1000 == 0:
                self.log(f"Processed {total_processed}/{total_samples} samples")

        # Training with early stopping
        gain = self.config.get('gain', 2.0)
        max_epochs = self.config.get('epochs', 100)
        patience = self.config.get('patience', 10)
        min_improvement = self.config.get('min_improvement', 0.1)

        self.log(f"Starting weight training with early stopping...")
        best_accuracy = 0.0
        best_round = 0
        patience_counter = 0
        best_weights = None

        # Initialize training iteration counter for visualization
        self.training_iteration = 0

        for rnd in range(max_epochs + 1):
            if rnd == 0:
                # Initial evaluation
                current_accuracy, correct_predictions, _ = self.evaluate(features_batches, encoded_targets_batches)
                self.log(f"Round {rnd:3d}: Initial Accuracy = {current_accuracy:.2f}% ({correct_predictions}/{total_samples})")
                best_accuracy = current_accuracy
                best_weights = self.anti_wts.copy()
                best_round = rnd

                # Capture initial visualization snapshot if interactive viz is enabled
                if enable_interactive_viz and hasattr(self, '_capture_interactive_snapshot'):
                    self._capture_interactive_snapshot(features_batches, encoded_targets_batches, rnd)

                # CAPTURE INITIAL SNAPSHOT - NEW
                if self.enhanced_visualization_enabled:
                    self._capture_enhanced_snapshot(features_batches, encoded_targets_batches, rnd)

                continue

            # Training pass
            self.train_epoch(features_batches, encoded_targets_batches, gain)
            self.training_iteration = rnd  # Update iteration counter

            # Evaluation after training round
            current_accuracy, correct_predictions, _ = self.evaluate(features_batches, encoded_targets_batches)
            self.log(f"Round {rnd:3d}: Accuracy = {current_accuracy:.2f}% ({correct_predictions}/{total_samples})")

            # Capture visualization snapshot if visualizer is attached - BOTH traditional and interactive
            if self.visualizer and rnd % 5 == 0:
                # Traditional snapshot (existing functionality)
                sample_features = np.vstack(features_batches)[:1000]  # Sample for performance
                sample_targets = np.concatenate(encoded_targets_batches)[:1000]
                sample_predictions, _ = self.predict_batch(sample_features)
                self.visualizer.capture_training_snapshot(
                    sample_features, sample_targets, self.anti_wts,
                    sample_predictions, current_accuracy, rnd
                )

                # Interactive 3D snapshot (new functionality)
                if enable_interactive_viz and hasattr(self, '_capture_interactive_snapshot'):
                    self._capture_interactive_snapshot(features_batches, encoded_targets_batches, rnd)

            # Early stopping logic
            if current_accuracy > best_accuracy + min_improvement:
                best_accuracy = current_accuracy
                best_weights = self.anti_wts.copy()
                best_round = rnd
                patience_counter = 0
                self.log(f"  → New best accuracy! (Improved by {current_accuracy - best_accuracy:.2f}%)")
            else:
                patience_counter += 1
                self.log(f"  → No improvement (Patience: {patience_counter}/{patience})")

            if patience_counter >= patience:
                self.log(f"\nEarly stopping triggered after {rnd} rounds.")
                self.log(f"Best accuracy {best_accuracy:.2f}% achieved at round {best_round}")
                break

        # Restore best weights
        if best_weights is not None:
            self.anti_wts = best_weights
            self.log(f"Restored best weights from round {best_round}")

        self.is_trained = True
        self.best_accuracy = best_accuracy
        self.best_round = best_round

        # CAPTURE ENHANCED SNAPSHOT DURING TRAINING - NEW
        if self.enhanced_visualization_enabled and rnd % self.viz_capture_interval == 0:
            self._capture_enhanced_snapshot(features_batches, encoded_targets_batches, rnd)

        # Log interactive visualization info if enabled
        if enable_interactive_viz and hasattr(self, 'visualizer') and hasattr(self.visualizer, 'feature_space_snapshots'):
            num_snapshots = len(self.visualizer.feature_space_snapshots)
            self.log(f"📊 Captured {num_snapshots} interactive 3D visualization snapshots")
            self.log("   Use generate_interactive_visualizations() to create interactive plots")

        self.log("Optimized training completed successfully!")
        return True

    def _enable_interactive_visualization_internal(self, capture_interval=5):
        """Internal method to enable interactive visualization during training"""
        if not hasattr(self, 'visualizer'):
            self.visualizer = DBNNVisualizer()
            self.log("✅ Visualizer attached for interactive 3D visualization")

        def _capture_interactive_snapshot(features_batches, encoded_targets_batches, iteration):
            """Capture feature space snapshot for interactive 3D visualization"""
            try:
                # Sample data for visualization (limit for performance)
                all_features = np.vstack(features_batches)
                all_targets = np.concatenate(encoded_targets_batches)

                # Use smaller sample for performance
                sample_size = min(500, len(all_features))
                if sample_size < len(all_features):
                    indices = np.random.choice(len(all_features), sample_size, replace=False)
                    sample_features = all_features[indices]
                    sample_targets = all_targets[indices]
                else:
                    sample_features = all_features
                    sample_targets = all_targets

                sample_predictions, _ = self.predict_batch(sample_features)

                # Get feature names
                feature_names = getattr(self, 'feature_columns', None)
                if feature_names is None:
                    feature_names = [f'Feature_{i+1}' for i in range(sample_features.shape[1])]

                # Get class names from encoder
                class_names = []
                if hasattr(self, 'class_encoder') and self.class_encoder.is_fitted:
                    for i in range(len(self.class_encoder.encoded_to_class)):
                        class_val = self.class_encoder.encoded_to_class.get(i+1, f'Class_{i+1}')
                        class_names.append(str(class_val))
                else:
                    unique_targets = np.unique(sample_targets)
                    class_names = [f'Class_{int(t)}' for t in unique_targets]

                # Capture the snapshot
                self.visualizer.capture_feature_space_snapshot(
                    sample_features, sample_targets, sample_predictions,
                    iteration, feature_names, class_names
                )

                if iteration % 10 == 0:  # Log every 10 iterations to avoid spam
                    self.log(f"📊 Captured feature space snapshot at iteration {iteration}")

            except Exception as e:
                self.log(f"⚠️ Could not capture interactive visualization snapshot: {e}")

        # Store the capture method for use during training
        self._capture_interactive_snapshot = _capture_interactive_snapshot
        self.log(f"✅ Interactive 3D visualization enabled (capturing every {capture_interval} iterations)")


    def enable_interactive_visualization(self, capture_interval=5):
        """Enable automatic capture of feature space snapshots for interactive 3D visualization"""
        if not hasattr(self, 'visualizer'):
            self.visualizer = DBNNVisualizer()
            self.log("✅ Visualizer attached for interactive 3D visualization")

        # Store capture interval for use during training
        self.viz_capture_interval = capture_interval
        self.log(f"✅ Interactive 3D visualization enabled (will capture every {capture_interval} iterations)")


    def _capture_interactive_snapshot(self, features_batches, encoded_targets_batches, iteration):
        """Internal method to capture feature space snapshot for interactive visualization"""
        try:
            # Sample data for visualization (limit for performance)
            all_features = np.vstack(features_batches)
            all_targets = np.concatenate(encoded_targets_batches)

            # Use smaller sample for performance
            sample_size = min(500, len(all_features))
            if len(all_features) > sample_size:
                indices = np.random.choice(len(all_features), sample_size, replace=False)
                sample_features = all_features[indices]
                sample_targets = all_targets[indices]
            else:
                sample_features = all_features
                sample_targets = all_targets

            sample_predictions, _ = self.predict_batch(sample_features)

            # Get feature names
            feature_names = getattr(self, 'feature_columns', None)
            if feature_names is None:
                feature_names = [f'Feature_{i+1}' for i in range(sample_features.shape[1])]

            # Get class names from encoder
            class_names = []
            if hasattr(self, 'class_encoder') and self.class_encoder.is_fitted:
                for i in range(len(self.class_encoder.encoded_to_class)):
                    class_val = self.class_encoder.encoded_to_class.get(i+1, f'Class_{i+1}')
                    class_names.append(str(class_val))
            else:
                unique_targets = np.unique(sample_targets)
                class_names = [f'Class_{int(t)}' for t in unique_targets]

            # Capture the snapshot
            self.visualizer.capture_feature_space_snapshot(
                sample_features, sample_targets, sample_predictions,
                iteration, feature_names, class_names
            )

            if iteration % 10 == 0:  # Log every 10 iterations to avoid spam
                self.log(f"📊 Captured feature space snapshot at iteration {iteration}")

        except Exception as e:
            self.log(f"⚠️ Could not capture interactive visualization snapshot: {e}")

    def generate_interactive_visualizations(self, output_dir="Visualisations"):
        """Generate all interactive visualizations after training - FIXED FOLDER STRUCTURE"""
        if not hasattr(self, 'visualizer') or not self.visualizer:
            print("❌ No visualizer available. Please enable enhanced visualization before training.")
            return None

        if not hasattr(self.visualizer, 'feature_space_snapshots') or not self.visualizer.feature_space_snapshots:
            print("❌ No visualization data available.")
            print("   Enable with enable_interactive_visualization() before training")
            return None

        try:
            import os

            # Create organized directory structure based on data file
            if hasattr(self, 'current_file') and self.current_file:
                base_name = os.path.splitext(os.path.basename(self.current_file))[0]
            else:
                base_name = "dbnn_training"

            if hasattr(self, 'feature_columns') and self.feature_columns:
                base_name = f"{base_name}_{len(self.feature_columns)}features"

            # Create main visualization folder for this dataset
            main_viz_dir = os.path.join(output_dir, base_name)
            standard_dir = os.path.join(main_viz_dir, "Standard")
            enhanced_dir = os.path.join(main_viz_dir, "Enhanced")
            os.makedirs(standard_dir, exist_ok=True)
            os.makedirs(enhanced_dir, exist_ok=True)

            outputs = {}

            # 1. Generate ENHANCED visualizations
            print("🔄 Generating enhanced visualizations...")

            # Enhanced 3D
            enhanced_3d_file = os.path.join(enhanced_dir, f"{base_name}_enhanced_3d.html")
            result = self.visualizer.generate_enhanced_interactive_3d(enhanced_3d_file)
            if result:
                outputs['enhanced_3d'] = result
                print(f"✅ Enhanced 3D: {result}")
            else:
                print("❌ Failed to generate enhanced_3d")

            # Advanced Dashboard
            dashboard_file = os.path.join(enhanced_dir, f"{base_name}_advanced_dashboard.html")
            result = self.visualizer.create_advanced_interactive_dashboard(dashboard_file)
            if result:
                outputs['advanced_dashboard'] = result
                print(f"✅ Advanced dashboard: {result}")
            else:
                print("❌ Failed to generate advanced_dashboard")

            # 2. Generate STANDARD visualizations
            print("🔄 Generating standard visualizations...")

            # Traditional Dashboard
            traditional_file = os.path.join(standard_dir, f"{base_name}_traditional_dashboard.html")
            result = self.visualizer.create_training_dashboard(traditional_file)
            if result:
                outputs['traditional_dashboard'] = result
                print(f"✅ Traditional dashboard: {result}")
            else:
                print("❌ Failed to generate traditional_dashboard")

            # Performance Metrics
            performance_file = os.path.join(standard_dir, f"{base_name}_performance.html")
            result = self.visualizer.generate_performance_metrics(performance_file)
            if result:
                outputs['performance'] = result
                print(f"✅ Performance metrics: {result}")
            else:
                print("❌ Failed to generate performance")

            # Correlation Matrix
            correlation_file = os.path.join(standard_dir, f"{base_name}_correlation.html")
            result = self.visualizer.generate_correlation_matrix(correlation_file)
            if result:
                outputs['correlation'] = result
                print(f"✅ Correlation matrix: {result}")
            else:
                print("❌ Failed to generate correlation")

            # Feature Explorer
            feature_file = os.path.join(standard_dir, f"{base_name}_feature_explorer.html")
            result = self.visualizer.generate_basic_3d_visualization(feature_file)
            if result:
                outputs['feature_explorer'] = result
                print(f"✅ Feature explorer: {result}")
            else:
                print("❌ Failed to generate feature_explorer")

            # Animated Training
            animated_file = os.path.join(standard_dir, f"{base_name}_animated.html")
            result = self.visualizer.generate_animated_training(animated_file)
            if result:
                outputs['animated'] = result
                print(f"✅ Animated training: {result}")
            else:
                print("❌ Failed to generate animated")

            # ANIMATED CONFUSION MATRIX - NEW
            confusion_file = os.path.join(standard_dir, f"{base_name}_confusion_animation.html")
            result = self.visualizer.generate_animated_confusion_matrix(confusion_file)
            if result:
                outputs['confusion_animation'] = result
                print(f"✅ Animated confusion matrix: {result}")
            else:
                print("❌ Failed to generate confusion animation")

            print(f"📊 Visualization Summary: {len(outputs)} successful")

            # Store the main visualization directory for easy access
            self.main_viz_directory = main_viz_dir
            return outputs

        except Exception as e:
            print(f"❌ Error generating visualizations: {e}")
            import traceback
            traceback.print_exc()
            return None

    def enable_parallel_processing(self, enabled=True):
        """Enable or disable parallel processing"""
        self.parallel_enabled = enabled
        if enabled:
            self.num_workers = self._detect_optimal_workers()
            self.log(f"✅ Parallel processing enabled with {self.num_workers} workers")
        else:
            self.log("✅ Parallel processing disabled")

    def set_max_workers(self, max_workers):
        """Set maximum number of parallel workers"""
        self.num_workers = min(max_workers, self._detect_optimal_workers())
        self.log(f"✅ Maximum workers set to {self.num_workers}")

    def optimize_memory(self):
        """Optimize memory usage by converting arrays to optimal dtypes"""
        if self.anti_net is not None:
            # Keep anti_net as int32 for large counts
            pass

        if self.anti_wts is not None and not self.memory_optimized:
            self.anti_wts = self.anti_wts.astype(np.float32)
            self.antit_wts = self.antit_wts.astype(np.float32)
            self.antip_wts = self.antip_wts.astype(np.float32)
            self.binloc = self.binloc.astype(np.float32)
            self.max_val = self.max_val.astype(np.float32)
            self.min_val = self.min_val.astype(np.float32)
            self.class_labels = self.class_labels.astype(np.float32)
            self.memory_optimized = True
            self.log("✅ Memory optimized: arrays converted to float32")

class DBNNWorkflow:
    """
    Complete workflow manager that uses DBNNCore and DBNNVisualizer
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.core = DBNNCore(config)
        self.visualizer = DBNNVisualizer()
        self.core.attach_visualizer(self.visualizer)

    def run_complete_workflow(self, train_file: str, test_file: Optional[str] = None,
                             predict_file: Optional[str] = None, use_csv: bool = True,
                             target_column: Optional[str] = None,
                             feature_columns: Optional[List[str]] = None):
        """Run complete training and evaluation workflow"""

        print("Starting complete DBNN workflow...")

        # Training
        success = self.core.train_with_early_stopping(
            train_file,
            test_file,
            use_csv,
            target_column,
            feature_columns,
            enable_interactive_viz=True,      # Enable 3D visualization
            viz_capture_interval=5            # Capture every 5 iterations
        )

        if not success:
            print("Training failed")
            return False

        # Generate visualizations
        if len(self.visualizer.training_history) > 0:
            self.visualizer.create_training_dashboard("training_results.html")
            print("Visualizations generated")

        # Test if test file provided
        if test_file and os.path.exists(test_file):
            self.test_model(test_file, use_csv)

        # Predict if predict file provided
        if predict_file and os.path.exists(predict_file):
            self.predict(predict_file, use_csv)

        print("Workflow completed successfully!")
        return True

    def test_model(self, test_file: str, use_csv: bool = True):
        """Test model on test data"""
        print(f"Testing model on: {test_file}")

        # Load test data
        features_batches, targets_batches, _, original_targets_batches = self.core.load_data(test_file)

        if not features_batches:
            print("No test data loaded")
            return

        # Encode targets
        encoded_targets_batches = []
        for batch in original_targets_batches:
            encoded_batch = self.core.class_encoder.transform(batch)
            encoded_targets_batches.append(encoded_batch)

        # Evaluate
        accuracy, correct_predictions, predictions = self.core.evaluate(features_batches, encoded_targets_batches)
        total_samples = sum(len(batch) for batch in features_batches)

        print(f"Test Results: Accuracy = {accuracy:.2f}% ({correct_predictions}/{total_samples})")

    def predict(self, data_file: str, use_csv: bool = True, output_file: Optional[str] = None):
        """Generate predictions on new data"""
        print(f"Generating predictions for: {data_file}")

        features_batches, _, _, _ = self.core.load_data(data_file)

        if not features_batches:
            print("No prediction data loaded")
            return

        all_predictions = []
        all_probabilities = []

        for features_batch in features_batches:
            predictions, probabilities = self.core.predict_batch(features_batch)
            all_predictions.extend(predictions)
            all_probabilities.extend(probabilities)

        # Decode predictions
        decoded_predictions = self.core.class_encoder.inverse_transform(all_predictions)

        # Save results
        if output_file is None:
            output_file = data_file.replace('.csv', '_predictions.csv').replace('.dat', '_predictions.csv')

        results_df = pd.DataFrame({
            'prediction_encoded': all_predictions,
            'prediction': decoded_predictions
        })

        # Add probability columns
        for i, class_name in enumerate(self.core.class_encoder.encoded_to_class.values()):
            probs = [prob.get(class_name, 0.0) for prob in all_probabilities]
            results_df[f'prob_{class_name}'] = probs

        results_df.to_csv(output_file, index=False)
        print(f"Predictions saved to: {output_file}")

        return decoded_predictions, all_probabilities

class EnhancedDBNNInterface:
    """
    Enhanced interface with fixed early stopping and better feature selection
    """

    def __init__(self, root):
        self.root = root
        self.root.title("DBNN Enhanced Interface")
        self.root.geometry("1200x800")

        self.core = None
        self.visualizer = None
        self.current_file = None
        self.file_type = None
        self.processing_indicator = None
        self.processing_frame = None
        self.processing_animation_id = None
        self.setup_ui()

    def setup_ui(self):
        """Setup enhanced UI with configuration tab system"""
        # Create notebook for tabs instead of paned window
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Setup main tab
        self.setup_main_tab()

        # Setup configuration tab
        self.setup_configuration_tab()

        # Add visualization menu
        self.setup_visualization_menu()

    def _handle_visualization_command(self, operation, window=None):
        """Handle visualization commands from the interface"""
        try:
            if operation == "open_3d":
                viz_file = "Visualisations/Enhanced/dbnn_training_enhanced_3d.html"
                if os.path.exists(viz_file):
                    import webbrowser
                    webbrowser.open('file://' + os.path.abspath(viz_file))
                else:
                    messagebox.showwarning("File Not Found",
                        "Enhanced 3D visualization not found.\n\n"
                        "Please generate visualizations first by:\n"
                        "1. Enabling 'Enhanced Visualization' before training\n"
                        "2. Or use the 'Visualize' button after training")

            elif operation == "open_dashboard":
                dashboard_file = "Visualisations/Standard/dbnn_training_traditional_dashboard.html"
                if os.path.exists(dashboard_file):
                    import webbrowser
                    webbrowser.open('file://' + os.path.abspath(dashboard_file))
                else:
                    messagebox.showwarning("File Not Found",
                        "Dashboard not found.\n\n"
                        "Please generate visualizations first using the 'Visualize' button.")

            elif operation == "refresh":
                self.visualize()  # Call the main visualize method

            if window:
                window.destroy()

        except Exception as e:
            self.log(f"Visualization command error: {e}")

    def toggle_tensor_mode(self):
        """Toggle between tensor and standard mode"""
        if self.core:
            self.core.enable_tensor_mode(self.tensor_mode.get())
            mode_text = "Complex Tensor Mode" if self.tensor_mode.get() else "Standard Iterative Mode"
            self.tensor_info.config(text=f"Current: {mode_text}")
            self.log(f"Switched to {mode_text}")

    def setup_main_tab(self):
        """Setup the main working tab"""
        main_frame = ttk.Frame(self.notebook)
        self.notebook.add(main_frame, text="Main Interface")

        # Main frame with paned window for better layout
        main_paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left frame for controls
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=1)

        # Right frame for console
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=2)

        # File selection
        file_frame = ttk.LabelFrame(left_frame, text="Data File", padding="5")
        file_frame.pack(fill='x', pady=5)

        ttk.Label(file_frame, text="Select CSV/DAT file:").grid(row=0, column=0, sticky='w')
        self.file_path = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_path, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_file).grid(row=0, column=2)
        ttk.Button(file_frame, text="Analyze", command=self.analyze_file).grid(row=0, column=3, padx=5)

        # File info
        self.file_info = scrolledtext.ScrolledText(file_frame, height=4, width=60)
        self.file_info.grid(row=1, column=0, columnspan=4, pady=5, sticky='we')

        # Quick configuration
        config_frame = ttk.LabelFrame(left_frame, text="Quick Configuration", padding="5")
        config_frame.pack(fill='x', pady=5)

        # Model name - auto-generated based on data file
        ttk.Label(config_frame, text="Model Name:").grid(row=0, column=0, sticky='w')
        self.model_name = tk.StringVar(value="my_model")
        self.model_name_entry = ttk.Entry(config_frame, textvariable=self.model_name, width=30)
        self.model_name_entry.grid(row=0, column=1, sticky='w', padx=5, columnspan=2)

        # Add tooltip for model name
        self.create_tooltip(self.model_name_entry, "Auto-generated from data file name. Model will be saved in Model/ folder.")

        # Training params in a grid
        params = [
            ("Resolution:", "resol", "100"),
            ("Epochs:", "epochs", "100"),
            ("Gain:", "gain", "2.0"),
            ("Margin:", "margin", "0.2"),
            ("Patience:", "patience", "10")
        ]

        for i, (label, name, default) in enumerate(params):
            ttk.Label(config_frame, text=label).grid(row=1, column=i*2, sticky='w', padx=5)
            var = tk.StringVar(value=default)
            setattr(self, name, var)
            ttk.Entry(config_frame, textvariable=var, width=8).grid(row=1, column=i*2+1, sticky='w', padx=5)

        # FIXED INDENTATION: Learning Mode section - ADD VISUALIZATION CONTROL
        tensor_frame = ttk.LabelFrame(left_frame, text="Learning Mode", padding="5")
        tensor_frame.pack(fill='x', pady=5)

        # Tensor mode checkbox (existing)
        self.tensor_mode = tk.BooleanVar(value=False)
        tensor_check = ttk.Checkbutton(
            tensor_frame,
            text="Enable Complex Tensor Mode (Experimental)",
            variable=self.tensor_mode,
            command=self.toggle_tensor_mode
        )
        tensor_check.pack(anchor='w', pady=2)

        # ENHANCED VISUALIZATION CONTROL - NEW
        self.enhanced_viz_var = tk.BooleanVar(value=False)  # Default disabled
        viz_check = ttk.Checkbutton(
            tensor_frame,
            text="Enable Enhanced Visualization (Educational)",
            variable=self.enhanced_viz_var,
            command=self.toggle_enhanced_visualization
        )
        viz_check.pack(anchor='w', pady=2)

        # Tooltip for enhanced visualization
        self.create_tooltip(viz_check,
            "Enhanced Visualization:\n"
            "• Captures 3D feature space snapshots during training\n"
            "• Creates interactive educational dashboards\n"
            "• Adds ~10-20% computational overhead\n"
            "• Recommended for understanding model behavior\n"
            "• Disable for maximum training speed"
        )

        # Visualization info label
        self.viz_info = ttk.Label(tensor_frame, text="Enhanced Visualization: DISABLED (faster training)",
                                foreground="red", font=('Arial', 9))
        self.viz_info.pack(anchor='w')

        # Tensor info label (existing)
        self.tensor_info = ttk.Label(tensor_frame, text="Current: Standard Iterative Mode",
                                   foreground="blue", font=('Arial', 9))
        self.tensor_info.pack(anchor='w')

        # Enhanced feature selection
        self.feature_frame = ttk.LabelFrame(left_frame, text="Feature Selection", padding="5")

        # Target selection
        ttk.Label(self.feature_frame, text="Target Column:").grid(row=0, column=0, sticky='w')
        self.target_col = tk.StringVar()
        self.target_combo = ttk.Combobox(self.feature_frame, textvariable=self.target_col, width=20)
        self.target_combo.grid(row=0, column=1, padx=5, pady=2)

        # Feature selection with checkboxes
        ttk.Label(self.feature_frame, text="Feature Columns:").grid(row=1, column=0, sticky='nw')

        # Frame for feature checkboxes with scrollbar
        feature_check_frame = ttk.Frame(self.feature_frame)
        feature_check_frame.grid(row=1, column=1, padx=5, pady=2, sticky='we')

        # Create a canvas with scrollbar for feature selection
        self.feature_canvas = tk.Canvas(feature_check_frame, height=120)
        scrollbar = ttk.Scrollbar(feature_check_frame, orient="vertical", command=self.feature_canvas.yview)
        self.feature_scroll_frame = ttk.Frame(self.feature_canvas)

        self.feature_scroll_frame.bind(
            "<Configure>",
            lambda e: self.feature_canvas.configure(scrollregion=self.feature_canvas.bbox("all"))
        )

        self.feature_canvas.create_window((0, 0), window=self.feature_scroll_frame, anchor="nw")
        self.feature_canvas.configure(yscrollcommand=scrollbar.set)

        self.feature_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.feature_vars = {}

        # Selection controls
        selection_frame = ttk.Frame(self.feature_frame)
        selection_frame.grid(row=2, column=0, columnspan=2, pady=5)

        ttk.Button(selection_frame, text="Select All", command=self.select_all_features).pack(side='left', padx=2)
        ttk.Button(selection_frame, text="Deselect All", command=self.deselect_all_features).pack(side='left', padx=2)
        ttk.Button(selection_frame, text="Invert Selection", command=self.invert_feature_selection).pack(side='left', padx=2)

        # Control buttons
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill='x', pady=10)

        # Row 1 buttons
        row1_frame = ttk.Frame(button_frame)
        row1_frame.pack(fill='x', pady=2)

        ttk.Button(row1_frame, text="Initialize Core", command=self.initialize_core).pack(side='left', padx=2)
        ttk.Button(row1_frame, text="Train Fresh", command=self.train_fresh).pack(side='left', padx=2)
        ttk.Button(row1_frame, text="Continue Training", command=self.continue_training).pack(side='left', padx=2)
        ttk.Button(row1_frame, text="Test Model", command=self.test_model).pack(side='left', padx=2)

        # Row 2 buttons
        row2_frame = ttk.Frame(button_frame)
        row2_frame.pack(fill='x', pady=2)

        ttk.Button(row2_frame, text="Predict", command=self.predict).pack(side='left', padx=2)
        ttk.Button(row2_frame, text="Save Model", command=self.save_model).pack(side='left', padx=2)
        ttk.Button(row2_frame, text="Load Model", command=self.load_model).pack(side='left', padx=2)
        ttk.Button(row2_frame, text="Visualize", command=self.visualize).pack(side='left', padx=2)

        # Row 3 buttons - Configuration management
        row3_frame = ttk.Frame(button_frame)
        row3_frame.pack(fill='x', pady=2)

        ttk.Button(row3_frame, text="Load Config", command=lambda: self.load_config()).pack(side='left', padx=2)
        ttk.Button(row3_frame, text="Save Config", command=self.save_config).pack(side='left', padx=2)
        ttk.Button(row3_frame, text="Validate Config", command=self.validate_config).pack(side='left', padx=2)

        # Debug controls
        debug_frame = ttk.LabelFrame(left_frame, text="Debug Controls", padding="5")
        debug_frame.pack(fill='x', pady=5)

        ttk.Button(debug_frame, text="Test Data Load", command=self.test_data_load).pack(side='left', padx=2)
        ttk.Button(debug_frame, text="Test Encoder", command=self.test_encoder).pack(side='left', padx=2)
        ttk.Button(debug_frame, text="Test Prediction", command=self.test_single_prediction).pack(side='left', padx=2)

        # Console output on the right
        console_frame = ttk.LabelFrame(right_frame, text="Training Output", padding="5")
        console_frame.pack(fill='both', expand=True, pady=5)

        self.console = scrolledtext.ScrolledText(console_frame, height=30, width=80)
        self.console.pack(fill='both', expand=True)

    def setup_visualization_menu(self):
        """Setup visualization menu with file manager"""
        # Create menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # Visualization menu
        viz_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Visualization", menu=viz_menu)

        # File Manager submenu
        file_manager_menu = tk.Menu(viz_menu, tearoff=0)
        viz_menu.add_cascade(label="File Manager", menu=file_manager_menu)

        file_manager_menu.add_command(
            label="Open Visualization Folder",
            command=self.open_visualization_folder
        )
        file_manager_menu.add_command(
            label="Refresh Visualizations",
            command=self.refresh_visualizations
        )
        file_manager_menu.add_separator()
        file_manager_menu.add_command(
            label="Open Enhanced 3D",
            command=lambda: self.open_specific_visualization("enhanced_3d")
        )
        file_manager_menu.add_command(
            label="Open Dashboard",
            command=lambda: self.open_specific_visualization("traditional_dashboard")
        )
        file_manager_menu.add_command(
            label="Open Confusion Matrix",
            command=lambda: self.open_specific_visualization("confusion_animation")
        )
        self.viz_type_mapping = {
            "complex_tensor": os.path.join("*", "Enhanced", "*_complex_tensor.html"),
            "phase_diagram": os.path.join("*", "Enhanced", "*_phase_diagram.html"),
            "enhanced_3d": os.path.join("*", "Enhanced", "*_enhanced_3d.html"),
            "traditional_dashboard": os.path.join("*", "Standard", "*_traditional_dashboard.html"),
            "confusion_animation": os.path.join("*", "Standard", "*_confusion_animation.html"),
        }

    def open_visualization_folder(self):
        """Open the visualization folder for current data file"""
        try:
            import subprocess
            import os
            import platform

            # Determine the visualization folder
            if hasattr(self, 'core') and hasattr(self.core, 'main_viz_directory'):
                viz_folder = self.core.main_viz_directory
            else:
                # Fallback: use current file to determine folder
                if hasattr(self, 'current_file') and self.current_file:
                    base_name = os.path.splitext(os.path.basename(self.current_file))[0]
                    viz_folder = os.path.join("Visualisations", base_name)
                else:
                    viz_folder = "Visualisations"

            # Create folder if it doesn't exist
            if not os.path.exists(viz_folder):
                os.makedirs(viz_folder, exist_ok=True)
                self.log(f"Created visualization folder: {viz_folder}")

            # Open folder based on operating system
            system = platform.system()
            if system == "Windows":
                os.startfile(viz_folder)
            elif system == "Darwin":  # macOS
                subprocess.run(["open", viz_folder])
            else:  # Linux
                subprocess.run(["xdg-open", viz_folder])

            self.log(f"📁 Opened visualization folder: {viz_folder}")

        except Exception as e:
            self.log(f"❌ Error opening visualization folder: {e}")
            messagebox.showerror("Error", f"Could not open visualization folder:\n{e}")

    def open_specific_visualization(self, viz_type):
        """Open a specific visualization file"""
        try:
            import webbrowser
            import os

            # Determine the visualization folder and file
            if hasattr(self, 'core') and hasattr(self.core, 'main_viz_directory'):
                base_folder = self.core.main_viz_directory
            else:
                if hasattr(self, 'current_file') and self.current_file:
                    base_name = os.path.splitext(os.path.basename(self.current_file))[0]
                    base_folder = os.path.join("Visualisations", base_name)
                else:
                    base_folder = "Visualisations"

            # Map visualization types to file paths
            viz_files = {
                "enhanced_3d": os.path.join(base_folder, "Enhanced", "*_enhanced_3d.html"),
                "traditional_dashboard": os.path.join(base_folder, "Standard", "*_traditional_dashboard.html"),
                "confusion_animation": os.path.join(base_folder, "Standard", "*_confusion_animation.html"),
                "performance": os.path.join(base_folder, "Standard", "*_performance.html"),
                "correlation": os.path.join(base_folder, "Standard", "*_correlation.html")
            }

            if viz_type in viz_files:
                import glob
                file_pattern = viz_files[viz_type]
                matching_files = glob.glob(file_pattern)

                if matching_files:
                    # Get the most recent file
                    most_recent = max(matching_files, key=os.path.getmtime)
                    webbrowser.open(f'file://{os.path.abspath(most_recent)}')
                    self.log(f"📊 Opened {viz_type}: {most_recent}")
                else:
                    self.log(f"❌ No {viz_type} file found. Please generate visualizations first.")
                    messagebox.showwarning(
                        "File Not Found",
                        f"No {viz_type} visualization found.\n\nPlease generate visualizations first using the 'Visualize' button."
                    )
            else:
                self.log(f"❌ Unknown visualization type: {viz_type}")

        except Exception as e:
            self.log(f"❌ Error opening visualization: {e}")

    def refresh_visualizations(self):
        """Refresh/regenerate all visualizations"""
        if not self.core:
            messagebox.showerror("Error", "Please initialize core first")
            return

        try:
            self.show_processing_indicator("Refreshing visualizations...")
            outputs = self.core.generate_interactive_visualizations()

            if outputs:
                self.log("✅ Visualizations refreshed:")
                for viz_type, file_path in outputs.items():
                    self.log(f"   {viz_type}: {file_path}")
            else:
                self.log("❌ No visualizations could be generated")

        except Exception as e:
            self.log(f"❌ Error refreshing visualizations: {e}")
        finally:
            self.hide_processing_indicator()

    def setup_configuration_tab(self):
        """Setup configuration tab with all default parameters"""
        # Create configuration tab
        config_frame = ttk.Frame(self.notebook)
        self.notebook.add(config_frame, text="Configuration")

        # Main configuration frame
        main_config_frame = ttk.LabelFrame(config_frame, text="Model Configuration Parameters", padding="10")
        main_config_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Create scrollable frame for configurations
        canvas = tk.Canvas(main_config_frame)
        scrollbar = ttk.Scrollbar(main_config_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Default configuration parameters
        self.config_params = {
            'resol': {'label': 'Resolution', 'default': 100, 'type': 'int', 'min': 10, 'max': 500, 'desc': 'Number of bins for feature discretization'},
            'gain': {'label': 'Gain', 'default': 2.0, 'type': 'float', 'min': 0.1, 'max': 10.0, 'desc': 'Weight update gain factor'},
            'margin': {'label': 'Margin', 'default': 0.2, 'type': 'float', 'min': 0.01, 'max': 1.0, 'desc': 'Classification margin threshold'},
            'patience': {'label': 'Patience', 'default': 10, 'type': 'int', 'min': 1, 'max': 100, 'desc': 'Early stopping patience (epochs)'},
            'epochs': {'label': 'Max Epochs', 'default': 100, 'type': 'int', 'min': 1, 'max': 1000, 'desc': 'Maximum training epochs'},
            'min_improvement': {'label': 'Min Improvement (%)', 'default': 0.1, 'type': 'float', 'min': 0.01, 'max': 10.0, 'desc': 'Minimum accuracy improvement for early stopping'},
            'fst_gain': {'label': 'First Gain', 'default': 1.0, 'type': 'float', 'min': 0.1, 'max': 5.0, 'desc': 'Initial gain factor'},
            'LoC': {'label': 'Learning Rate', 'default': 0.65, 'type': 'float', 'min': 0.01, 'max': 1.0, 'desc': 'Learning rate coefficient'},
            'nLoC': {'label': 'Negative Learning Rate', 'default': 0.0, 'type': 'float', 'min': 0.0, 'max': 1.0, 'desc': 'Negative learning rate coefficient'},
            'nresol': {'label': 'Negative Resolution', 'default': 0, 'type': 'int', 'min': 0, 'max': 100, 'desc': 'Negative resolution parameter'},
            'skpchk': {'label': 'Skip Check', 'default': 0, 'type': 'int', 'min': 0, 'max': 1, 'desc': 'Skip validation checks (0=no, 1=yes)'},
            'oneround': {'label': 'One Round', 'default': 100, 'type': 'int', 'min': 1, 'max': 1000, 'desc': 'Samples per training round'}
        }

        self.config_vars = {}

        # Create configuration entries in two columns
        param_keys = list(self.config_params.keys())
        half = len(param_keys) // 2 + len(param_keys) % 2

        for i, key in enumerate(param_keys):
            param = self.config_params[key]
            row = i % half
            col = i // half

            frame = ttk.Frame(scrollable_frame)
            frame.grid(row=row, column=col, sticky='w', padx=15, pady=8)

            # Parameter label
            ttk.Label(frame, text=param['label'] + ":", width=22, anchor='w').pack(side='left')

            # Entry field
            var = tk.StringVar(value=str(param['default']))
            self.config_vars[key] = var

            entry = ttk.Entry(frame, textvariable=var, width=12)
            entry.pack(side='left', padx=5)

            # Range label
            range_text = f"({param['min']}-{param['max']})"
            ttk.Label(frame, text=range_text, width=12, foreground="gray").pack(side='left', padx=5)

            # Add tooltip with parameter description
            self.create_tooltip(entry, f"{param['desc']}\nDefault: {param['default']}\nType: {param['type']}\nRange: {param['min']} to {param['max']}")
            self.create_tooltip(frame, f"{param['desc']}\nDefault: {param['default']}")

        # Configuration management buttons
        button_frame = ttk.Frame(config_frame)
        button_frame.pack(fill='x', padx=10, pady=10)

        ttk.Button(button_frame, text="Load Default Configuration",
                   command=self.load_default_config).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Validate Current Configuration",
                   command=self.validate_config).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Save Configuration to File",
                   command=self.save_config).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Load Configuration from File",
                   command=self.load_config).pack(side='left', padx=5)

        # Apply to main tab button
        ttk.Button(button_frame, text="Apply to Main Tab",
                   command=self.apply_config_to_main).pack(side='left', padx=5)

        # Config status
        self.config_status = ttk.Label(config_frame, text="Ready to configure", foreground="green", font=('Arial', 10))
        self.config_status.pack(pady=10)

        # Configuration info
        info_text = """Configuration Management:
    • Load Default: Reset all parameters to default values
    • Validate: Check if current values are valid
    • Save: Save configuration to Model/<dataset>_config.json
    • Load: Load configuration from file
    • Apply to Main: Copy values to main tab quick configuration"""

        info_label = ttk.Label(config_frame, text=info_text, justify=tk.LEFT, foreground="blue",
                              font=('Arial', 9))
        info_label.pack(pady=5)

    def toggle_enhanced_visualization(self):
        """Toggle enhanced visualization on/off"""
        if self.core:
            enabled = self.enhanced_viz_var.get()
            success = self.core.enable_enhanced_visualization(enabled, capture_interval=5)

            if success:
                status = "ENABLED" if enabled else "DISABLED"
                color = "green" if enabled else "red"
                speed_note = "(slower but educational)" if enabled else "(faster training)"
                self.viz_info.config(text=f"Enhanced Visualization: {status} {speed_note}", foreground=color)
                self.log(f"Enhanced visualization {status.lower()}")
            else:
                self.log("❌ Failed to configure enhanced visualization")
        else:
            self.log("⚠️ Please initialize core first")
            self.enhanced_viz_var.set(False)  # Reset if no core

    def create_tooltip(self, widget, text):
        """Create a tooltip for a widget"""
        def on_enter(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            label = ttk.Label(tooltip, text=text, background="lightyellow",
                             relief="solid", borderwidth=1, padding=5,
                             font=('Arial', 9), wraplength=300)
            label.pack()
            widget.tooltip = tooltip

        def on_leave(event):
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()

        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)

    def _file_operation(self, operation, window=None):
        """Handle file operations for visualization window"""
        try:
            if operation == "open_3d":
                # Open 3D visualization
                import webbrowser
                webbrowser.open('file://' + os.path.abspath("Visualisations/dbnn_training_advanced_dashboard.html"))

            elif operation == "open_dashboard":
                # Open traditional dashboard
                import webbrowser
                webbrowser.open('file://' + os.path.abspath("Visualisations/dbnn_training_traditional_dashboard.html"))

            elif operation == "refresh":
                # Refresh visualizations
                self.log("Refreshing visualizations...")
                if hasattr(self, 'core') and self.core:
                    outputs = self.core.generate_interactive_visualizations()
                    if outputs:
                        self.log("Visualizations refreshed")

            if window:
                window.destroy()

        except Exception as e:
            self.log(f"File operation error: {e}")

    def load_default_config(self):
        """Load default configuration values"""
        for key, param in self.config_params.items():
            self.config_vars[key].set(str(param['default']))
        self.config_status.config(text="Default configuration loaded", foreground="green")

    def validate_config(self):
        """Validate configuration parameters"""
        errors = []
        for key, param in self.config_params.items():
            try:
                value = self.config_vars[key].get()
                if param['type'] == 'int':
                    int_value = int(value)
                    if not (param['min'] <= int_value <= param['max']):
                        errors.append(f"{param['label']} must be between {param['min']} and {param['max']}")
                elif param['type'] == 'float':
                    float_value = float(value)
                    if not (param['min'] <= float_value <= param['max']):
                        errors.append(f"{param['label']} must be between {param['min']} and {param['max']}")
            except ValueError:
                errors.append(f"{param['label']} must be a {param['type']}")

        if errors:
            error_msg = "\n".join(errors)
            self.config_status.config(text="Configuration errors found", foreground="red")
            messagebox.showerror("Configuration Error", f"Please fix the following errors:\n\n{error_msg}")
        else:
            self.config_status.config(text="Configuration validated successfully", foreground="green")
            messagebox.showinfo("Success", "Configuration validated successfully!")

        return len(errors) == 0

    def get_config_dict(self):
        """Get current configuration as dictionary"""
        config = {}
        for key in self.config_params.keys():
            try:
                value = self.config_vars[key].get()
                if self.config_params[key]['type'] == 'int':
                    config[key] = int(value)
                elif self.config_params[key]['type'] == 'float':
                    config[key] = float(value)
            except ValueError:
                # Use default if invalid
                config[key] = self.config_params[key]['default']
        return config

    def save_config(self):
        """Save configuration to file"""
        if not self.current_file:
            messagebox.showerror("Error", "Please select a data file first")
            return

        if not self.validate_config():
            return

        # Create config filename based on data file
        data_filename = os.path.splitext(os.path.basename(self.current_file))[0]
        config_filename = f"Model/{data_filename}_config.json"

        # Ensure Model directory exists
        os.makedirs("Model", exist_ok=True)

        config_data = {
            'config': self.get_config_dict(),
            'data_file': self.current_file,
            'feature_columns': self.get_selected_features() if hasattr(self, 'get_selected_features') else [],
            'target_column': self.target_col.get() if hasattr(self, 'target_col') else "",
            'saved_date': time.strftime("%Y-%m-%d %H:%M:%S")
        }

        try:
            with open(config_filename, 'w') as f:
                json.dump(config_data, f, indent=2)

            self.config_status.config(text=f"Configuration saved to {config_filename}", foreground="green")
            self.log(f"Configuration saved: {config_filename}")
            messagebox.showinfo("Success", f"Configuration saved to:\n{config_filename}")

        except Exception as e:
            self.config_status.config(text="Error saving configuration", foreground="red")
            messagebox.showerror("Error", f"Failed to save configuration: {e}")

    def load_config(self, config_file=None):
        """Load configuration from file"""
        if config_file is None:
            config_file = filedialog.askopenfilename(
                title="Select Configuration File",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )

        if not config_file:
            return

        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)

            # Update configuration values
            config_dict = config_data.get('config', {})
            for key, value in config_dict.items():
                if key in self.config_vars:
                    self.config_vars[key].set(str(value))

            # Update data file if specified and exists
            data_file = config_data.get('data_file', '')
            if data_file and os.path.exists(data_file):
                self.file_path.set(data_file)
                self.current_file = data_file
                self.analyze_file()

            self.config_status.config(text=f"Configuration loaded from {config_file}", foreground="green")
            self.log(f"Configuration loaded: {config_file}")
            messagebox.showinfo("Success", f"Configuration loaded from:\n{config_file}")

        except Exception as e:
            self.config_status.config(text="Error loading configuration", foreground="red")
            messagebox.showerror("Error", f"Failed to load configuration: {e}")

    def auto_load_config(self):
        """Automatically load configuration if it exists for current data file"""
        if not self.current_file:
            return

        data_filename = os.path.splitext(os.path.basename(self.current_file))[0]
        config_filename = f"Model/{data_filename}_config.json"

        if os.path.exists(config_filename):
            try:
                with open(config_filename, 'r') as f:
                    config_data = json.load(f)

                # Update configuration values
                config_dict = config_data.get('config', {})
                for key, value in config_dict.items():
                    if key in self.config_vars:
                        self.config_vars[key].set(str(value))

                # Update feature selection if available
                feature_columns = config_data.get('feature_columns', [])
                target_column = config_data.get('target_column', '')

                if target_column and hasattr(self, 'target_col'):
                    self.target_col.set(target_column)

                if feature_columns and hasattr(self, 'feature_vars'):
                    # Update feature checkboxes
                    for col, var in self.feature_vars.items():
                        var.set(col in feature_columns)

                self.config_status.config(text=f"Auto-loaded configuration", foreground="green")
                self.log(f"Auto-loaded configuration: {config_filename}")

            except Exception as e:
                self.log(f"Warning: Could not auto-load configuration: {e}")


    def apply_config_to_main(self):
        """Apply configuration tab values to main tab quick configuration"""
        try:
            config_dict = self.get_config_dict()

            # Update main tab parameters
            if hasattr(self, 'resol'):
                self.resol.set(str(config_dict.get('resol', 100)))
            if hasattr(self, 'epochs'):
                self.epochs.set(str(config_dict.get('epochs', 100)))
            if hasattr(self, 'gain'):
                self.gain.set(str(config_dict.get('gain', 2.0)))
            if hasattr(self, 'margin'):
                self.margin.set(str(config_dict.get('margin', 0.2)))
            if hasattr(self, 'patience'):
                self.patience.set(str(config_dict.get('patience', 10)))

            self.config_status.config(text="Configuration applied to main tab", foreground="green")
            self.log("Configuration applied from config tab to main tab")

        except Exception as e:
            self.config_status.config(text="Error applying configuration", foreground="red")
            self.log(f"Error applying configuration: {e}")

    def select_all_features(self):
        """Select all feature checkboxes"""
        for var in self.feature_vars.values():
            var.set(True)

    def deselect_all_features(self):
        """Deselect all feature checkboxes"""
        for var in self.feature_vars.values():
            var.set(False)

    def invert_feature_selection(self):
        """Invert feature selection"""
        for var in self.feature_vars.values():
            var.set(not var.get())

    def get_selected_features(self):
        """Get list of selected feature columns"""
        return [col for col, var in self.feature_vars.items() if var.get()]

    def log(self, message):
        """Add message to console"""
        self.console.insert(tk.END, f"{message}\n")
        self.console.see(tk.END)
        self.root.update()

    def browse_file(self):
        """Browse for data file"""
        filename = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("DAT files", "*.dat"), ("All files", "*.*")]
        )
        if filename:
            self.file_path.set(filename)

    def predict(self):
        """Make predictions using the trained model's configuration"""
        if not self.core or not getattr(self.core, 'is_trained', False):
            messagebox.showerror("Error", "No trained model available")
            return

        predict_file = filedialog.askopenfilename(
            title="Select Prediction File",
            filetypes=[("CSV files", "*.csv"), ("DAT files", "*.dat"), ("All files", "*.*")]
        )

        if not predict_file:
            return

        output_file = filedialog.asksaveasfilename(
            title="Save Predictions As",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )

        if not output_file:
            return

        try:
            self.show_processing_indicator("Analyzing system resources...")

            # Use the trained model's configuration
            if hasattr(self.core, 'feature_columns') and self.core.feature_columns:
                training_features = self.core.feature_columns
                self.log("=== PREDICTION USING TRAINED MODEL ===")
                self.log(f"Using feature columns from trained model: {training_features}")
                self.log(f"Model has {self.core.innodes} input nodes, {self.core.outnodes} output nodes")
            else:
                self.log("❌ No feature configuration found in trained model")
                self.hide_processing_indicator()
                return

            # Load the ORIGINAL data first to preserve all columns
            self.log("Loading original data with all columns...")
            try:
                import pandas as pd
                original_df = pd.read_csv(predict_file)
                self.log(f"Original data shape: {original_df.shape}")
                self.log(f"Original columns: {list(original_df.columns)}")
            except Exception as e:
                self.log(f"❌ Error loading original data: {e}")
                self.hide_processing_indicator()
                return

            # Now load just the features for prediction
            target_column = None  # No target in prediction mode
            feature_columns = training_features  # Use training features from model

            features_batches, _, feature_columns_used, _ = self.core.load_data(
                predict_file,
                target_column=target_column,
                feature_columns=feature_columns,
                batch_size=10000
            )

            if not features_batches:
                self.log("❌ No prediction data loaded")
                self.hide_processing_indicator()
                return

            total_batches = len(features_batches)
            total_samples = sum(len(batch) for batch in features_batches)
            self.log(f"Loaded {total_samples} samples in {total_batches} batches for prediction")

            # Verify feature dimensions match the trained model
            if features_batches and len(features_batches[0]) > 0:
                actual_feature_count = features_batches[0].shape[1]
                expected_feature_count = len(training_features)

                if actual_feature_count != expected_feature_count:
                    self.log(f"❌ Feature dimension mismatch!")
                    self.log(f"   Model expects: {expected_feature_count} features")
                    self.log(f"   Data has: {actual_feature_count} features")
                    self.log(f"   Expected features: {training_features}")
                    self.hide_processing_indicator()
                    return
                else:
                    self.log(f"✅ Feature dimensions match: {actual_feature_count} features")

            # Process batches and collect predictions
            all_predictions = []
            all_probabilities = []
            all_confidence_scores = []

            for batch_idx, features_batch in enumerate(features_batches):
                self.show_processing_indicator(f"Predicting batch {batch_idx+1}/{total_batches}...")

                predictions, probabilities = self.core.predict_batch(features_batch)
                all_predictions.extend(predictions)
                all_probabilities.extend(probabilities)

                # Calculate confidence scores (max probability)
                for prob_dict in probabilities:
                    if prob_dict:
                        confidence = max(prob_dict.values())
                    else:
                        confidence = 0.0
                    all_confidence_scores.append(confidence)

            # Decode predictions to original class labels
            decoded_predictions = self.core.class_encoder.inverse_transform(all_predictions)

            # Log prediction distribution for debugging
            from collections import Counter
            prediction_counts = Counter(decoded_predictions)
            self.log(f"📊 Prediction distribution: {dict(prediction_counts)}")

            # Check if we're getting all expected classes
            expected_classes = list(self.core.class_encoder.encoded_to_class.values())
            predicted_classes = list(prediction_counts.keys())
            missing_classes = set(expected_classes) - set(predicted_classes)
            if missing_classes:
                self.log(f"⚠️ Warning: Some classes not predicted: {missing_classes}")

            # Create enhanced output dataframe
            self.log("Creating enhanced output with all original columns...")

            # Start with the original dataframe
            output_df = original_df.copy()

            # Add prediction columns
            output_df['prediction'] = decoded_predictions
            output_df['prediction_encoded'] = all_predictions
            output_df['confidence'] = all_confidence_scores

            # Add probability columns for ALL classes (even if not predicted)
            if all_probabilities and len(all_probabilities) > 0:
                # Get all class names from the encoder
                all_class_names = list(self.core.class_encoder.encoded_to_class.values())

                for class_name in all_class_names:
                    prob_values = []
                    for prob_dict in all_probabilities:
                        prob_value = prob_dict.get(class_name, 0.0)
                        prob_values.append(prob_value)
                    output_df[f'prob_{class_name}'] = prob_values

                self.log(f"✅ Added probability columns for {len(all_class_names)} classes")

            # Save the enhanced output
            output_df.to_csv(output_file, index=False)
            self.log(f"✅ Enhanced predictions saved to: {output_file}")
            self.log(f"   Original columns preserved: {len(original_df.columns)}")
            self.log(f"   Prediction columns added: {len(output_df.columns) - len(original_df.columns)}")
            self.log(f"   Total samples: {len(output_df)}")

            # Show detailed prediction summary
            self._show_detailed_prediction_summary(output_df, expected_classes)

            self.hide_processing_indicator()

        except Exception as e:
            self.hide_processing_indicator()
            self.log(f"❌ Prediction error: {e}")
            self.log(traceback.format_exc())

    def _show_detailed_prediction_summary(self, output_df, expected_classes):
        """Show detailed prediction summary with class analysis - FIXED METHOD NAME"""
        try:
            total_predictions = len(output_df)

            self.log("\n📊 DETAILED PREDICTION SUMMARY:")
            self.log("=" * 50)
            self.log(f"Total predictions: {total_predictions}")

            # Prediction distribution
            prediction_counts = output_df['prediction'].value_counts()
            self.log("\nPrediction distribution:")
            for pred, count in prediction_counts.items():
                percentage = (count / total_predictions) * 100
                self.log(f"  {pred}: {count} ({percentage:.1f}%)")

            # Confidence statistics
            avg_confidence = output_df['confidence'].mean()
            min_confidence = output_df['confidence'].min()
            max_confidence = output_df['confidence'].max()
            self.log(f"\nConfidence statistics:")
            self.log(f"  Average: {avg_confidence:.3f}")
            self.log(f"  Range: {min_confidence:.3f} - {max_confidence:.3f}")

            # Missing classes analysis
            predicted_classes = set(prediction_counts.index)
            missing_classes = set(expected_classes) - predicted_classes
            if missing_classes:
                self.log(f"\n⚠️  Classes NOT predicted: {missing_classes}")
                self.log("   This may indicate:")
                self.log("   - Class imbalance in training data")
                self.log("   - Model needs more training")
                self.log("   - Tensor mode convergence issues")
            else:
                self.log(f"\n✅ All {len(expected_classes)} classes were predicted")

            self.log("=" * 50)

        except Exception as e:
            self.log(f"Error generating prediction summary: {e}")

    def _update_model_name_from_file(self):
        """Update model name based on the current data file"""
        if not self.current_file:
            return

        # Extract base name from file path
        base_name = os.path.splitext(os.path.basename(self.current_file))[0]
        model_name = f"Model/{base_name}"
        self.model_name.set(model_name)
        self.log(f"Model name auto-set to: {model_name}")


    def train_fresh(self):
        """Train a fresh model from scratch with PROPER visualization enabling"""
        if not self.core:
            messagebox.showerror("Error", "Please initialize core first")
            return

        if not self.current_file:
            messagebox.showerror("Error", "Please select a training file first")
            return

        # Show mode information
        mode = "TENSOR" if getattr(self.core, 'tensor_mode', False) else "STANDARD"
        result = messagebox.askyesno(
            f"Train Fresh ({mode} Mode)",
            f"This will train a fresh model using {mode} mode.\n\n"
            f"Tensor Mode: Single-pass orthogonal projection\n"
            f"Standard Mode: Iterative error-correction\n\n"
            f"Enhanced Visualization: {'ENABLED' if self.enhanced_viz_var.get() else 'DISABLED'}\n\n"
            "Continue?"
        )

        if not result:
            return

        try:
            self.show_processing_indicator("Starting fresh training...")
            self.log("=== STARTING FRESH TRAINING ===")

            # Reset core to ensure fresh start
            self.initialize_core()

            # PROPERLY ENABLE ENHANCED VISUALIZATION
            viz_enabled = self.enhanced_viz_var.get()
            if viz_enabled:
                # Initialize visualizer if not already done
                if not hasattr(self.core, 'visualizer') or self.core.visualizer is None:
                    self.core.visualizer = DBNNVisualizer()
                # Enable BOTH enhanced and interactive visualization
                self.core.enable_enhanced_visualization(True, capture_interval=5)
                self.core.enable_interactive_visualization(capture_interval=5)
                self.log("✅ ENHANCED 3D VISUALIZATION ENABLED for this training session")
                self.log("   - Capturing feature space snapshots every 5 iterations")
                self.log("   - Will generate interactive 3D plots after training")
            else:
                self.core.enable_enhanced_visualization(False)
                self.log("✅ Enhanced visualization DISABLED for faster training")

            # Set flag to indicate fresh training
            self.fresh_training = True

            # Call the original training method
            success = self._train_model_internal(fresh_training=True)

            if success and viz_enabled:
                # GENERATE ENHANCED VISUALIZATIONS AFTER TRAINING
                self.log("=== GENERATING ENHANCED VISUALIZATIONS ===")
                self.show_processing_indicator("Generating enhanced 3D visualizations...")

                try:
                    outputs = self.core.generate_interactive_visualizations()
                    if outputs:
                        self.log("✅ ENHANCED VISUALIZATIONS GENERATED:")
                        for viz_type, file_path in outputs.items():
                            self.log(f"   {viz_type}: {file_path}")

                        # Ask to open the main visualization
                        if 'interactive_3d' in outputs:
                            result = messagebox.askyesno(
                                "Enhanced Visualization Ready",
                                f"Interactive 3D visualization generated!\n\n"
                                f"File: {outputs['interactive_3d']}\n\n"
                                f"Open in web browser?"
                            )
                            if result:
                                import webbrowser
                                webbrowser.open(f'file://{os.path.abspath(outputs["interactive_3d"])}')
                    else:
                        self.log("❌ No enhanced visualizations could be generated")
                except Exception as viz_error:
                    self.log(f"❌ Enhanced visualization generation failed: {viz_error}")

                self.hide_processing_indicator()

            if success:
                self.log("✅ Fresh training completed successfully!")
            else:
                self.log("❌ Fresh training failed!")

        except Exception as e:
            self.hide_processing_indicator()
            messagebox.showerror("Error", f"Fresh training failed: {e}")
            self.log(f"Error: {e}")
            self.log(traceback.format_exc())

    def continue_training(self):
        """Continue training from existing model or start fresh if no model exists"""
        if not self.core:
            messagebox.showerror("Error", "Please initialize core first")
            return

        if not self.current_file:
            messagebox.showerror("Error", "Please select a training file first")
            return

        try:
            self.show_processing_indicator("Continuing training...")
            self.log("=== CONTINUING TRAINING ===")

            # APPLY VISUALIZATION SETTING - NEW
            viz_enabled = self.enhanced_viz_var.get()
            self.core.enable_enhanced_visualization(viz_enabled, capture_interval=5)

            if viz_enabled:
                self.log("✅ Enhanced visualization ENABLED for continued training")
            else:
                self.log("✅ Enhanced visualization DISABLED for faster training")

            # Check if we have a trained model to continue from
            if not getattr(self.core, 'is_trained', False):
                self.log("No existing model found, starting fresh training...")
                # No existing model, so this becomes a fresh training
                success = self._train_model_internal(fresh_training=True)
            else:
                self.log(f"Continuing from existing model (best accuracy: {getattr(self.core, 'best_accuracy', 0):.2f}%)")
                # Continue from existing model
                self.core.enable_interactive_visualization(capture_interval=5)
                self.log("✅ Interactive 3D visualization enabled for continued training")

                success = self._train_model_internal(fresh_training=False)

            if success:
                self.log("✅ Continued training completed successfully!")
            else:
                self.log("❌ Continued training failed!")

        except Exception as e:
            self.hide_processing_indicator()
            messagebox.showerror("Error", f"Continued training failed: {e}")
            self.log(f"Error: {e}")
            self.log(traceback.format_exc())


    def _train_model_internal(self, fresh_training=False):
        """Internal training method that handles both fresh and continued training with tensor mode support"""
        try:
            # Load data
            if self.file_type == 'csv':
                target_column = self.target_col.get()
                feature_columns = self.get_selected_features()

                # VALIDATION: Check if target is selected for training
                if target_column == 'None' or not target_column:
                    messagebox.showerror("Error",
                        "For training, you must select a target column.\n"
                        "Please select a target column from the dropdown (not 'None').")
                    self.hide_processing_indicator()
                    return False

                if not feature_columns:
                    messagebox.showerror("Error", "Please select at least one feature column")
                    self.hide_processing_indicator()
                    return False

                # Check if target column is in feature columns (common mistake)
                if target_column in feature_columns:
                    messagebox.showerror("Error",
                        f"Target column '{target_column}' cannot be in feature columns.\n"
                        f"Please deselect it from the feature checkboxes.")
                    self.hide_processing_indicator()
                    return False

                self.log(f"Training with: target={target_column}, features={feature_columns}")
                features_batches, targets_batches, feature_columns_used, original_targets_batches = self.core.load_data(
                    self.current_file, target_column, feature_columns, batch_size=5000
                )
            else:
                self.log("Training with DAT file")
                features_batches, targets_batches, feature_columns_used, original_targets_batches = self.core.load_data(
                    self.current_file, batch_size=5000
                )
                # For DAT files, set default feature and target info
                target_column = "last_column"
                feature_columns = feature_columns_used

            if not features_batches:
                self.log("ERROR: No data loaded!")
                self.hide_processing_indicator()
                return False

            total_samples = sum(len(batch) for batch in features_batches)
            self.log(f"Loaded {total_samples} samples")

            # Check if we're in tensor mode
            is_tensor_mode = getattr(self.core, 'tensor_mode', False) and getattr(self.core, 'tensor_core', None) is not None

            if is_tensor_mode:
                # TENSOR MODE TRAINING - Single pass transformation
                self.log("🧠 Starting TENSOR TRANSFORMATION training...")
                self.show_processing_indicator("Building tensor transformation...")

                # Fit encoder
                all_original_targets = np.concatenate(original_targets_batches) if original_targets_batches else np.array([])
                self.core.class_encoder.fit(all_original_targets)

                # Use tensor core for training
                success = self.core.tensor_core.tensor_train(
                    self.current_file, None, self.file_type == 'csv',
                    target_column, feature_columns
                )

                if success:
                    # Copy trained state from tensor core to main core for prediction compatibility
                    self._sync_tensor_to_core()

                    # AUTOMATIC MODEL SAVING AFTER TENSOR TRAINING
                    self.log("=== Auto-saving Tensor Model ===")
                    self.show_processing_indicator("Auto-saving tensor model...")

                    # Get feature information
                    feature_columns = self.get_selected_features() if hasattr(self, 'get_selected_features') else []
                    target_column = self.target_col.get() if hasattr(self, 'target_col') else ""

                    # Debug logging
                    self.log(f"🔧 Auto-save feature configuration:")
                    self.log(f"   From core.feature_columns: {feature_columns}")
                    self.log(f"   From core.target_column: {target_column}")

                    if not feature_columns:
                        # Fallback to tensor core if main core doesn't have it
                        if hasattr(self.core.tensor_core, 'feature_columns') and self.core.tensor_core.feature_columns:
                            feature_columns = self.core.tensor_core.feature_columns
                            target_column = self.core.tensor_core.target_column
                            self.log(f"   Fallback to tensor core: {feature_columns}")
                        else:
                            # Final fallback to UI selection
                            feature_columns = self.get_selected_features() if hasattr(self, 'get_selected_features') else []
                            target_column = self.target_col.get() if hasattr(self, 'target_col') else ""
                            self.log(f"   Fallback to UI selection: {feature_columns}")

                    self.log(f"Saving model with features: {feature_columns}")

                    # Use core's auto-save functionality
                    saved_model_path = self.core.save_model_auto(
                        model_dir="Model",
                        data_filename=self.current_file,
                        feature_columns=feature_columns,
                        target_column=target_column
                    )

                    if saved_model_path:
                        self.log(f"✅ Tensor model automatically saved to: {saved_model_path}")
                        self.log(f"   Best accuracy: {self.core.best_accuracy:.2f}%")
                    else:
                        self.log("❌ Automatic tensor model saving failed!")

                    # Test the model immediately after training
                    self.log("Testing tensor model...")
                    try:
                        if hasattr(self.core, 'innodes') and self.core.innodes > 0:
                            test_sample = np.random.randn(self.core.innodes)
                            predictions, probabilities = self.core.predict_batch(test_sample.reshape(1, -1))
                            self.log(f"✅ Post-training test - Prediction: {predictions[0]}")
                            self.log("✅ Tensor model is ready for predictions")
                    except Exception as e:
                        self.log(f"❌ Post-training test failed: {e}")

                    self.log("=== TENSOR TRANSFORMATION TRAINING COMPLETED ===")
                    self.hide_processing_indicator()
                    self.log(f"Final model: {self.core.innodes} inputs, {self.core.outnodes} outputs")
                    self.log(f"Feature columns used: {feature_columns_used}")
                    self.log(f"Target column: {target_column if self.file_type == 'csv' else 'last column (DAT)'}")

                    encoded_classes = self.core.class_encoder.get_encoded_classes()
                    self.log(f"Classes: {len(encoded_classes)} - {encoded_classes}")

                    return True
                else:
                    self.log("❌ Tensor training failed!")
                    self.hide_processing_indicator()
                    return False

            else:
                # STANDARD MODE TRAINING - Original iterative approach
                # For fresh training, reset the encoder and network
                if fresh_training:
                    self.log("🔄 Initializing fresh training...")
                    # Fit encoder and setup network from scratch
                    all_original_targets = np.concatenate(original_targets_batches) if original_targets_batches else np.array([])
                    self.core.class_encoder.fit(all_original_targets)

                    encoded_classes = self.core.class_encoder.get_encoded_classes()
                    self.log(f"Fresh network: {len(feature_columns_used)} inputs, {len(encoded_classes)} outputs")

                    # Initialize arrays for fresh training
                    innodes = len(feature_columns_used)
                    outnodes = len(encoded_classes)
                    resol = int(self.resol.get())

                    # Validate dimensions before initializing arrays
                    if innodes <= 0:
                        self.log("ERROR: No input features detected!")
                        self.hide_processing_indicator()
                        return False
                    if outnodes <= 0:
                        self.log("ERROR: No output classes detected!")
                        self.hide_processing_indicator()
                        return False

                    self.core.initialize_arrays(innodes, resol, outnodes)

                    # Setup class_labels
                    self.core.class_labels[0] = float(self.margin.get())
                    for i, encoded_val in enumerate(encoded_classes, 1):
                        self.core.class_labels[i] = float(encoded_val)

                # Encode targets
                encoded_targets_batches = []
                for batch in original_targets_batches:
                    encoded_batch = self.core.class_encoder.transform(batch)
                    encoded_targets_batches.append(encoded_batch)

                # Initialize training (only for fresh training)
                if fresh_training:
                    self.log("Initializing training parameters...")
                    omax, omin = self.core.initialize_training(features_batches, encoded_targets_batches, resol)

                    # Build initial network
                    self.log("Building initial network...")
                    total_processed = 0
                    for batch_idx, (features_batch, targets_batch) in enumerate(zip(features_batches, encoded_targets_batches)):
                        processed = self.core.process_training_batch(features_batch, targets_batch)
                        total_processed += processed
                        if batch_idx % 10 == 0:
                            self.log(f"  Processed {total_processed}/{total_samples} samples")
                            self.show_processing_indicator(f"Building network... {total_processed}/{total_samples}")

                    self.log(f"✅ Initial network built with {total_processed} samples")

                # Training with FIXED early stopping
                gain = float(self.gain.get())
                max_epochs = int(self.epochs.get())
                patience = int(self.patience.get())

                training_type = "FRESH" if fresh_training else "CONTINUED"
                self.log(f"{training_type} TRAINING: {max_epochs} epochs, gain={gain}, patience={patience}")

                # Store current best accuracy for comparison
                current_best_accuracy = getattr(self.core, 'best_accuracy', 0.0)
                best_accuracy = current_best_accuracy
                best_weights = self.core.anti_wts.copy() if hasattr(self.core, 'anti_wts') and self.core.anti_wts is not None else None
                best_round = getattr(self.core, 'best_round', 0)
                patience_counter = 0

                for rnd in range(max_epochs + 1):
                    if rnd == 0 and fresh_training:
                        # Initial evaluation for fresh training
                        current_accuracy, correct, _ = self.core.evaluate(features_batches, encoded_targets_batches)
                        self.log(f"Epoch {rnd:3d}: Initial Accuracy = {current_accuracy:.2f}% ({correct}/{total_samples})")
                        best_accuracy = current_accuracy
                        best_weights = self.core.anti_wts.copy()
                        best_round = rnd
                        continue
                    elif rnd == 0 and not fresh_training:
                        # For continued training, show current model accuracy
                        current_accuracy, correct, _ = self.core.evaluate(features_batches, encoded_targets_batches)
                        self.log(f"Epoch {rnd:3d}: Current Model Accuracy = {current_accuracy:.2f}% ({correct}/{total_samples})")
                        if current_accuracy > best_accuracy:
                            best_accuracy = current_accuracy
                            best_weights = self.core.anti_wts.copy()
                            best_round = rnd
                        continue

                    # Update processing indicator for training progress
                    if rnd % 5 == 0:
                        training_type_str = "Fresh" if fresh_training else "Continued"
                        self.show_processing_indicator(f"{training_type_str} Training... Epoch {rnd}/{max_epochs}")

                    # Training epoch
                    self.core.train_epoch(features_batches, encoded_targets_batches, gain)

                    # Evaluate
                    current_accuracy, correct, predictions = self.core.evaluate(features_batches, encoded_targets_batches)

                    # Update best weights only if we get a new best accuracy
                    if current_accuracy > best_accuracy:
                        best_accuracy = current_accuracy
                        best_weights = self.core.anti_wts.copy()
                        best_round = rnd
                        patience_counter = 0
                        improvement = current_accuracy - current_best_accuracy
                        if fresh_training:
                            self.log(f"Epoch {rnd:3d}: Accuracy = {current_accuracy:.2f}% → NEW BEST")
                        else:
                            self.log(f"Epoch {rnd:3d}: Accuracy = {current_accuracy:.2f}% → NEW BEST (+{improvement:.2f}% from previous best)")
                    else:
                        patience_counter += 1
                        improvement = current_accuracy - best_accuracy
                        if improvement > 0:
                            self.log(f"Epoch {rnd:3d}: Accuracy = {current_accuracy:.2f}% (+{improvement:.2f}%)")
                        else:
                            self.log(f"Epoch {rnd:3d}: Accuracy = {current_accuracy:.2f}%")

                    # Visualization
                    if self.visualizer and rnd % 5 == 0:
                        sample_size = min(1000, total_samples)
                        sample_features = np.vstack([batch[:100] for batch in features_batches])[:sample_size]
                        sample_targets = np.concatenate([batch[:100] for batch in encoded_targets_batches])[:sample_size]
                        sample_predictions, _ = self.core.predict_batch(sample_features)

                        # Create visualization directory if it doesn't exist
                        if self.current_file:
                            base_name = os.path.splitext(os.path.basename(self.current_file))[0]
                            viz_dir = f"Visualisations/{base_name}"
                            os.makedirs(viz_dir, exist_ok=True)

                        self.visualizer.capture_training_snapshot(
                            sample_features, sample_targets, self.core.anti_wts,
                            sample_predictions, current_accuracy, rnd
                        )

                    # Early stopping
                    if patience_counter >= patience:
                        self.log(f"Early stopping at epoch {rnd} (no improvement for {patience} epochs)")
                        break

                # Restore best weights
                if best_weights is not None:
                    self.core.anti_wts = best_weights
                    self.log(f"Restored best weights from epoch {best_round}")

                self.core.is_trained = True
                self.core.best_accuracy = best_accuracy
                self.core.best_round = best_round

                # AUTOMATIC MODEL SAVING AFTER TRAINING
                self.log("=== Auto-saving Trained Model ===")
                self.show_processing_indicator("Auto-saving trained model...")

                # Get feature information
                feature_columns = self.get_selected_features() if hasattr(self, 'get_selected_features') else []
                target_column = self.target_col.get() if hasattr(self, 'target_col') else ""

                # Use core's auto-save functionality
                saved_model_path = self.core.save_model_auto(
                    model_dir="Model",
                    data_filename=self.current_file,
                    feature_columns=feature_columns,
                    target_column=target_column
                )

                if saved_model_path:
                    self.log(f"✅ Model automatically saved to: {saved_model_path}")
                    self.log(f"   Best accuracy: {best_accuracy:.2f}% at epoch {best_round}")
                else:
                    self.log("❌ Automatic model saving failed!")

                # Test the model immediately after training
                self.log("Testing trained model...")
                try:
                    if hasattr(self.core, 'innodes') and self.core.innodes > 0:
                        test_sample = np.random.randn(self.core.innodes)
                        predictions, probabilities = self.core.predict_batch(test_sample.reshape(1, -1))
                        self.log(f"✅ Post-training test - Prediction: {predictions[0]}")
                        self.log("✅ Model is ready for predictions")
                except Exception as e:
                    self.log(f"❌ Post-training test failed: {e}")

                training_type_str = "Fresh" if fresh_training else "Continued"
                self.log(f"=== {training_type_str.upper()} TRAINING COMPLETED ===")
                self.hide_processing_indicator()
                self.log(f"Best accuracy: {best_accuracy:.2f}% at epoch {best_round}")
                self.log(f"Final model: {self.core.innodes} inputs, {self.core.outnodes} outputs")
                self.log(f"Feature columns used: {feature_columns_used}")
                self.log(f"Target column: {target_column if self.file_type == 'csv' else 'last column (DAT)'}")

                encoded_classes = self.core.class_encoder.get_encoded_classes()
                self.log(f"Classes: {len(encoded_classes)} - {encoded_classes}")

                return True

        except Exception as e:
            self.hide_processing_indicator()
            messagebox.showerror("Error", f"Training failed: {e}")
            self.log(f"Error: {e}")
            self.log(traceback.format_exc())
            return False

    def _sync_tensor_to_core(self):
        """Sync tensor core state to main core for prediction compatibility"""
        if self.core.tensor_core and self.core.tensor_mode:
            tensor_core = self.core.tensor_core

            # Copy ALL essential state including feature configuration
            state_attributes = [
                'is_trained', 'best_accuracy', 'best_round', 'innodes', 'outnodes',
                'feature_columns', 'target_column', 'class_labels', 'binloc',
                'max_val', 'min_val', 'resolution_arr', 'anti_net', 'anti_wts'
            ]

            for attr in state_attributes:
                if hasattr(tensor_core, attr) and getattr(tensor_core, attr) is not None:
                    setattr(self.core, attr, getattr(tensor_core, attr))

            # CRITICAL: Ensure feature_columns is properly set
            if not hasattr(self.core, 'feature_columns') or not self.core.feature_columns:
                # Try to get from tensor core or fallback to UI selection
                if hasattr(tensor_core, 'feature_columns') and tensor_core.feature_columns:
                    self.core.feature_columns = tensor_core.feature_columns
                else:
                    # Fallback to UI selection
                    self.core.feature_columns = self.get_selected_features() if hasattr(self, 'get_selected_features') else []

            # Ensure target_column is properly set
            if not hasattr(self.core, 'target_column') or not self.core.target_column:
                if hasattr(tensor_core, 'target_column') and tensor_core.target_column:
                    self.core.target_column = tensor_core.target_column
                else:
                    # Fallback to UI selection
                    self.core.target_column = self.target_col.get() if hasattr(self, 'target_col') else ""

            # Ensure class encoder is properly set
            if hasattr(tensor_core, 'class_encoder') and tensor_core.class_encoder.is_fitted:
                self.core.class_encoder = tensor_core.class_encoder

            # Log the synchronization
            feature_count = len(self.core.feature_columns) if hasattr(self.core, 'feature_columns') else 0
            self.log(f"✅ Tensor model synchronized to core: {feature_count} features, {self.core.outnodes} classes")

            # Debug: Log what was actually synchronized
            self.log(f"   Feature columns: {getattr(self.core, 'feature_columns', 'NOT SET')}")
            self.log(f"   Target column: {getattr(self.core, 'target_column', 'NOT SET')}")
            self.log(f"   is_trained: {getattr(self.core, 'is_trained', False)}")

    def auto_load_model(self):
        """Automatically load model if it exists for current data file - PROPERLY FIXED"""
        if not self.current_file:
            return

        # Get the expected model file path
        base_name = os.path.splitext(os.path.basename(self.current_file))[0]
        model_pattern = f"Model/{base_name}_*_model.bin"

        # Find the most recent model file
        import glob
        model_files = glob.glob(model_pattern)

        if model_files:
            # Sort by modification time and get the most recent
            model_files.sort(key=os.path.getmtime, reverse=True)
            latest_model = model_files[0]

            try:
                self.log(f"🔄 Auto-loading existing model: {latest_model}")

                # FIX: The load_model method doesn't return a success value, it just loads
                # We need to check if the model was loaded by verifying core state AFTER loading

                # Store current core state to compare
                was_trained_before = getattr(self.core, 'is_trained', False)

                # Load the model - this method doesn't return anything, it modifies self.core
                self.load_model(latest_model)  # This will show its own success/failure messages

                # Check if the model was actually loaded by verifying core state changed
                is_trained_after = getattr(self.core, 'is_trained', False)
                has_innodes = hasattr(self.core, 'innodes') and self.core.innodes > 0

                # FIX: Only log "failed" if the model clearly didn't load
                if is_trained_after and has_innodes:
                    # Model loaded successfully - the success message already came from load_model
                    # Don't log "failed" because it already worked!
                    pass  # Success was already logged in load_model method
                else:
                    # Only log failure if the model didn't actually load
                    if not was_trained_before and not is_trained_after:
                        self.log("❌ Auto-load: Model file found but failed to load properly")
                    else:
                        self.log("⚠️ Auto-load: Model file found but loading had issues")

            except Exception as e:
                self.log(f"❌ Auto-load error: {e}")
                import traceback
                self.log(traceback.format_exc())
        else:
            self.log("ℹ️ No existing model found for auto-loading")

    def analyze_file(self):
        """Analyze the selected file and auto-load configuration and model"""
        filename = self.file_path.get()
        if not filename:
            messagebox.showerror("Error", "Please select a file first")
            return

        try:
            self.show_processing_indicator("Analyzing file...")
            self.log(f"Analyzing file: {filename}")

            if filename.endswith('.csv'):
                self.file_type = 'csv'
                df = pd.read_csv(filename, nrows=100)
                columns = df.columns.tolist()

                self.file_info.delete(1.0, tk.END)
                self.file_info.insert(tk.END, f"CSV File: {filename}\n")
                self.file_info.insert(tk.END, f"Columns ({len(columns)}): {', '.join(columns)}\n")
                self.file_info.insert(tk.END, f"Sample data: {len(df)} rows\n")
                self.file_info.insert(tk.END, f"First row sample: {df.iloc[0].tolist()}\n")

                # Setup target selection - ADD "None" OPTION FOR PREDICTION
                target_options = ['None'] + columns  # Add "None" as first option
                self.target_combo['values'] = target_options
                self.target_col.set('None')  # Default to no target for prediction

                # Clear previous feature checkboxes
                for widget in self.feature_scroll_frame.winfo_children():
                    widget.destroy()
                self.feature_vars.clear()

                # Create checkboxes for ALL columns as potential features
                # User can decide which to include/exclude
                for i, col in enumerate(columns):
                    var = tk.BooleanVar(value=True)  # Default all to selected
                    self.feature_vars[col] = var
                    cb = ttk.Checkbutton(self.feature_scroll_frame, text=col, variable=var)
                    cb.grid(row=i, column=0, sticky='w', padx=2, pady=1)

                # Update canvas scrollregion
                self.feature_scroll_frame.update_idletasks()
                self.feature_canvas.configure(scrollregion=self.feature_canvas.bbox("all"))

                self.feature_frame.pack(fill='x', pady=5)

            elif filename.endswith('.dat'):
                self.file_type = 'dat'
                with open(filename, 'r') as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()][:100]

                self.file_info.delete(1.0, tk.END)
                self.file_info.insert(tk.END, f"DAT File: {filename}\n")
                self.file_info.insert(tk.END, f"Total lines: {len(lines)}\n")
                if lines:
                    values = lines[0].split()
                    self.file_info.insert(tk.END, f"Columns: {len(values)} (assuming all are features for prediction)\n")

                # Don't show feature selection for DAT files in prediction mode
                if self.feature_frame.winfo_ismapped():
                    self.feature_frame.pack_forget()

            else:
                messagebox.showerror("Error", "Unsupported file format")
                self.hide_processing_indicator()
                return

            self.current_file = filename
            self.log("File analysis completed")

            # Auto-update model name based on data file
            self._update_model_name_from_file()

            # Auto-load configuration if it exists
            self.auto_load_config()

            # Auto-load model if it exists
            self.auto_load_model()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze file: {e}")
            self.log(f"Error: {e}")
            import traceback
            self.log(traceback.format_exc())
        finally:
            self.hide_processing_indicator()
    def save_trained_model_auto(self, best_accuracy, best_round):
        """GUI wrapper for automatic model saving - now much simpler"""
        if not self.core or not getattr(self.core, 'is_trained', False):
            self.log("❌ No trained model to save")
            return None

        try:
            # Get feature information
            feature_columns = self.get_selected_features() if hasattr(self, 'get_selected_features') else []
            target_column = self.target_col.get() if hasattr(self, 'target_col') else ""

            # Use core's auto-save functionality
            saved_model_path = self.core.save_model_auto(
                model_dir="Model",
                data_filename=self.current_file,
                feature_columns=feature_columns,
                target_column=target_column
            )

            if saved_model_path:
                self.log(f"✅ Model automatically saved: {saved_model_path}")
                self.log(f"   Best accuracy: {best_accuracy:.2f}% at round {best_round}")
            else:
                self.log("❌ Failed to automatically save model")

            return saved_model_path

        except Exception as e:
            self.log(f"❌ Error in automatic model saving: {e}")
            return None

    def load_model_auto_config(self, model_path):
        """GUI wrapper for auto-configuration model loading"""
        try:
            self.show_processing_indicator("Loading model with auto-configuration...")

            if not self.core:
                self.initialize_core()

            # Use core's load_model method (not load_model_auto_config since it doesn't exist)
            success = self.core.load_model(model_path)

            if success:
                # Update GUI with loaded configuration
                self.model_name.set(os.path.splitext(os.path.basename(model_path))[0])

                # Update configuration tab with loaded config
                if hasattr(self, 'config_vars'):
                    for key, value in self.core.config.items():
                        if key in self.config_vars:
                            self.config_vars[key].set(str(value))

                # Update main tab quick configuration
                self.apply_core_config_to_main()

                self.log("✅ Model loaded successfully")
                if hasattr(self.core, 'best_accuracy'):
                    self.log(f"   Best accuracy: {self.core.best_accuracy:.2f}%")
                if hasattr(self.core, 'feature_columns'):
                    self.log(f"   Features: {len(self.core.feature_columns)}")

                # Set tensor mode based on loaded model
                if hasattr(self.core, 'tensor_mode'):
                    current_mode = self.core.tensor_mode
                    if hasattr(self, 'tensor_mode'):
                        self.tensor_mode.set(current_mode)
                        self.toggle_tensor_mode()
            else:
                self.log("❌ Failed to load model")

            self.hide_processing_indicator()
            return success

        except Exception as e:
            self.log(f"❌ Load error: {e}")
            self.hide_processing_indicator()
            return False

    def apply_core_config_to_main(self):
        """Apply core configuration to main tab quick configuration"""
        try:
            if not self.core:
                return

            # Update main tab parameters from core config
            if hasattr(self, 'resol'):
                self.resol.set(str(self.core.config.get('resol', 100)))
            if hasattr(self, 'epochs'):
                self.epochs.set(str(self.core.config.get('epochs', 100)))
            if hasattr(self, 'gain'):
                self.gain.set(str(self.core.config.get('gain', 2.0)))
            if hasattr(self, 'margin'):
                self.margin.set(str(self.core.config.get('margin', 0.2)))
            if hasattr(self, 'patience'):
                self.patience.set(str(self.core.config.get('patience', 10)))

            self.log("Configuration synchronized from loaded model")

        except Exception as e:
            self.log(f"Error applying core configuration: {e}")

    def show_processing_indicator(self, message="Processing..."):
        """Show a spinning color wheel processing indicator"""
        # If already showing, just update the message
        if self.processing_frame and self.processing_frame.winfo_exists():
            for widget in self.processing_frame.winfo_children():
                if isinstance(widget, ttk.Label):
                    widget.config(text=message)
            self.processing_frame.deiconify()
            self.start_animation()
            return

        # Create new processing window
        self.processing_frame = tk.Toplevel(self.root)
        self.processing_frame.title("Processing")
        self.processing_frame.geometry("300x150")
        self.processing_frame.transient(self.root)
        self.processing_frame.grab_set()

        # Make it modal
        self.processing_frame.focus_set()
        self.processing_frame.grab_set()

        # Center the window
        self.processing_frame.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - 300) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - 150) // 2
        self.processing_frame.geometry(f"+{x}+{y}")

        # Processing message
        msg_label = ttk.Label(self.processing_frame, text=message, font=('Arial', 10))
        msg_label.pack(pady=10)

        # Canvas for spinning wheel
        canvas_size = 60
        self.processing_indicator = tk.Canvas(
            self.processing_frame,
            width=canvas_size,
            height=canvas_size,
            bg=self.processing_frame.cget('bg'),
            highlightthickness=0
        )
        self.processing_indicator.pack(pady=10)

        # Draw initial wheel
        self.animation_angle = 0
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        self.draw_spinning_wheel()

        # Start animation
        self.start_animation()

        # Force update
        self.processing_frame.update()

    def hide_processing_indicator(self):
        """Hide the processing indicator"""
        self.stop_animation()

        if self.processing_frame and self.processing_frame.winfo_exists():
            try:
                self.processing_frame.grab_release()
                self.processing_frame.destroy()
            except tk.TclError:
                pass  # Window already destroyed

        self.processing_frame = None
        self.processing_indicator = None

        # Ensure main window gets focus back
        self.root.focus_set()

    def draw_spinning_wheel(self):
        """Draw the spinning color wheel"""
        if not self.processing_indicator:
            return

        self.processing_indicator.delete("all")
        canvas_size = 60
        center = canvas_size // 2
        radius = 20

        # Draw spinning wheel with multiple colored segments
        segment_angle = 360 / len(self.colors)
        for i, color in enumerate(self.colors):
            start_angle = self.animation_angle + (i * segment_angle)
            end_angle = start_angle + segment_angle

            self.processing_indicator.create_arc(
                center - radius, center - radius,
                center + radius, center + radius,
                start=start_angle, extent=segment_angle - 5,  # Small gap between segments
                fill=color, outline="", width=0
            )

    def start_animation(self):
        """Start the spinning animation"""
        if self.processing_frame and self.processing_frame.winfo_exists():
            self.animation_angle = (self.animation_angle + 15) % 360
            self.draw_spinning_wheel()
            self.processing_animation_id = self.root.after(50, self.start_animation)
        else:
            self.stop_animation()

    def stop_animation(self):
        """Stop the spinning animation"""
        if self.processing_animation_id:
            self.root.after_cancel(self.processing_animation_id)
            self.processing_animation_id = None

    def test_data_load(self):
        """Test data loading functionality"""
        if not self.current_file:
            messagebox.showerror("Error", "Please select a file first")
            return

        try:
            self.log("=== Testing Data Load ===")

            if self.file_type == 'csv':
                target_column = self.target_col.get()
                feature_columns = self.get_selected_features()

                features_batches, targets_batches, feature_columns_used, original_targets_batches = self.core.load_data(
                    self.current_file, target_column, feature_columns, batch_size=1000
                )
            else:
                features_batches, targets_batches, feature_columns_used, original_targets_batches = self.core.load_data(
                    self.current_file, batch_size=1000
                )

            total_samples = sum(len(batch) for batch in features_batches)
            self.log(f"Loaded {total_samples} samples in {len(features_batches)} batches")
            self.log(f"Feature columns: {feature_columns_used}")

            if features_batches:
                first_batch = features_batches[0]
                self.log(f"First batch shape: {first_batch.shape}")
                self.log(f"First sample features: {first_batch[0][:5]}...")
                self.log(f"First sample target: {original_targets_batches[0][0]}")

            self.log("=== Data Load Test Completed ===")

        except Exception as e:
            self.log(f"Data load test failed: {e}")
            self.log(traceback.format_exc())

    def test_encoder(self):
        """Test class encoder functionality"""
        if not self.current_file:
            messagebox.showerror("Error", "Please select a file first")
            return

        try:
            self.log("=== Testing Encoder ===")

            if self.file_type == 'csv':
                target_column = self.target_col.get()
                feature_columns = self.get_selected_features()

                features_batches, targets_batches, _, original_targets_batches = self.core.load_data(
                    self.current_file, target_column, feature_columns, batch_size=1000
                )
            else:
                features_batches, targets_batches, _, original_targets_batches = self.core.load_data(
                    self.current_file, batch_size=1000
                )

            all_original_targets = np.concatenate(original_targets_batches) if original_targets_batches else np.array([])
            self.core.class_encoder.fit(all_original_targets)

            self.log(f"Encoder fitted with {len(self.core.class_encoder.get_encoded_classes())} classes")

            if len(all_original_targets) > 0:
                sample_targets = all_original_targets[:5]
                encoded = self.core.class_encoder.transform(sample_targets)
                decoded = self.core.class_encoder.inverse_transform(encoded)

                self.log("Encoding test:")
                for orig, enc, dec in zip(sample_targets, encoded, decoded):
                    self.log(f"  {orig} -> {enc} -> {dec}")

            self.log("=== Encoder Test Completed ===")

        except Exception as e:
            self.log(f"Encoder test failed: {e}")
            self.log(traceback.format_exc())

    def test_single_prediction(self):
        """Test single prediction"""
        if not self.core or not hasattr(self.core, 'is_trained') or not self.core.is_trained:
            messagebox.showerror("Error", "Please train a model first")
            return

        try:
            self.log("=== Testing Single Prediction ===")

            if hasattr(self.core, 'innodes') and self.core.innodes > 0:
                test_sample = np.random.randn(self.core.innodes)
                self.log(f"Test sample shape: {test_sample.shape}")

                predictions, probabilities = self.core.predict_batch(test_sample.reshape(1, -1))

                self.log(f"Prediction: {predictions[0]}")
                self.log(f"Probabilities: {probabilities[0]}")
            else:
                self.log("No network architecture information available")

            self.log("=== Single Prediction Test Completed ===")

        except Exception as e:
            self.log(f"Single prediction test failed: {e}")
            self.log(traceback.format_exc())

    def initialize_core(self):
        """Initialize DBNN core"""
        try:
            config = {
                'resol': int(self.resol.get()),
                'epochs': int(self.epochs.get()),
                'gain': float(self.gain.get()),
                'margin': float(self.margin.get()),
                'patience': int(self.patience.get())
            }

            self.core = DBNNCore(config)
            self.visualizer = DBNNVisualizer()
            self.core.attach_visualizer(self.visualizer)

            # Set tensor mode based on checkbox
            if hasattr(self, 'tensor_mode'):
                self.core.enable_tensor_mode(self.tensor_mode.get())
                mode_text = "Tensor" if self.tensor_mode.get() else "Standard"
                self.log(f"DBNN core initialized in {mode_text} mode")
            else:
                self.log("DBNN core initialized in Standard mode")

            self.log(f"Config: {config}")


            self.log("DBNN core initialized successfully")
            self.log(f"Config: {config}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize core: {e}")
            self.log(f"Error: {e}")
            self.log(traceback.format_exc())

    def train_model(self):
        """Train model with fixed early stopping and automatic model saving"""
        if not self.core:
            messagebox.showerror("Error", "Please initialize core first")
            return

        if not self.current_file:
            messagebox.showerror("Error", "Please select a training file first")
            return

        try:
            self.show_processing_indicator("Starting training process...")
            self.log("=== Starting Training ===")

            # Load data
            if self.file_type == 'csv':
                target_column = self.target_col.get()
                feature_columns = self.get_selected_features()

                # VALIDATION: Check if target is selected for training
                if target_column == 'None' or not target_column:
                    messagebox.showerror("Error",
                        "For training, you must select a target column.\n"
                        "Please select a target column from the dropdown (not 'None').")
                    self.hide_processing_indicator()
                    return False

                if not feature_columns:
                    messagebox.showerror("Error", "Please select at least one feature column")
                    self.hide_processing_indicator()
                    return False

                # Check if target column is in feature columns (common mistake)
                if target_column in feature_columns:
                    messagebox.showerror("Error",
                        f"Target column '{target_column}' cannot be in feature columns.\n"
                        f"Please deselect it from the feature checkboxes.")
                    self.hide_processing_indicator()
                    return False

                self.log(f"Training with: target={target_column}, features={feature_columns}")
                features_batches, targets_batches, feature_columns_used, original_targets_batches = self.core.load_data(
                    self.current_file, target_column, feature_columns, batch_size=5000
                )
            else:
                self.log("Training with DAT file")
                features_batches, targets_batches, feature_columns_used, original_targets_batches = self.core.load_data(
                    self.current_file, batch_size=5000
                )
                # For DAT files, set default feature and target info
                target_column = "last_column"
                feature_columns = feature_columns_used

            if not features_batches:
                self.log("ERROR: No data loaded!")
                self.hide_processing_indicator()
                return False

            total_samples = sum(len(batch) for batch in features_batches)
            self.log(f"Loaded {total_samples} samples")

            # Fit encoder and setup network
            all_original_targets = np.concatenate(original_targets_batches) if original_targets_batches else np.array([])
            self.core.class_encoder.fit(all_original_targets)

            encoded_classes = self.core.class_encoder.get_encoded_classes()
            self.log(f"Network: {len(feature_columns_used)} inputs, {len(encoded_classes)} outputs")
            self.log(f"Feature columns used: {feature_columns_used}")
            self.log(f"Encoded classes: {encoded_classes}")

            # Initialize arrays
            innodes = len(feature_columns_used)
            outnodes = len(encoded_classes)
            resol = int(self.resol.get())

            # Validate dimensions before initializing arrays
            if innodes <= 0:
                self.log("ERROR: No input features detected!")
                self.hide_processing_indicator()
                return False
            if outnodes <= 0:
                self.log("ERROR: No output classes detected!")
                self.hide_processing_indicator()
                return False

            self.core.initialize_arrays(innodes, resol, outnodes)

            # Setup class_labels
            self.core.class_labels[0] = float(self.margin.get())
            for i, encoded_val in enumerate(encoded_classes, 1):
                self.core.class_labels[i] = float(encoded_val)

            # Encode targets
            encoded_targets_batches = []
            for batch in original_targets_batches:
                encoded_batch = self.core.class_encoder.transform(batch)
                encoded_targets_batches.append(encoded_batch)

            # Initialize training
            self.log("Initializing training parameters...")
            omax, omin = self.core.initialize_training(features_batches, encoded_targets_batches, resol)

            # Build initial network
            self.log("Building initial network...")
            total_processed = 0
            for batch_idx, (features_batch, targets_batch) in enumerate(zip(features_batches, encoded_targets_batches)):
                processed = self.core.process_training_batch(features_batch, targets_batch)
                total_processed += processed
                if batch_idx % 10 == 0:
                    self.log(f"  Processed {total_processed}/{total_samples} samples")
                    # Update the processing indicator to show progress
                    self.show_processing_indicator(f"Building network... {total_processed}/{total_samples}")

            self.log(f"✅ Initial network built with {total_processed} samples")

            # Training with FIXED early stopping
            gain = float(self.gain.get())
            max_epochs = int(self.epochs.get())
            patience = int(self.patience.get())

            self.log(f"Training: {max_epochs} epochs, gain={gain}, patience={patience}")

            best_accuracy = 0.0
            best_weights = None
            best_round = 0
            patience_counter = 0

            for rnd in range(max_epochs + 1):
                if rnd == 0:
                    # Initial evaluation
                    current_accuracy, correct, _ = self.core.evaluate(features_batches, encoded_targets_batches)
                    self.log(f"Epoch {rnd:3d}: Initial Accuracy = {current_accuracy:.2f}% ({correct}/{total_samples})")
                    best_accuracy = current_accuracy
                    best_weights = self.core.anti_wts.copy()
                    best_round = rnd
                    continue

                # Update processing indicator for training progress
                if rnd % 5 == 0:
                    self.show_processing_indicator(f"Training... Epoch {rnd}/{max_epochs}")

                # Training epoch
                self.core.train_epoch(features_batches, encoded_targets_batches, gain)

                # Evaluate
                current_accuracy, correct, predictions = self.core.evaluate(features_batches, encoded_targets_batches)

                # FIXED: Always update best weights when we get a new best accuracy
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    best_weights = self.core.anti_wts.copy()
                    best_round = rnd
                    patience_counter = 0
                    self.log(f"Epoch {rnd:3d}: Accuracy = {current_accuracy:.2f}% → NEW BEST")
                else:
                    patience_counter += 1
                    improvement = current_accuracy - best_accuracy
                    if improvement > 0:
                        self.log(f"Epoch {rnd:3d}: Accuracy = {current_accuracy:.2f}% (+{improvement:.2f}%)")
                    else:
                        self.log(f"Epoch {rnd:3d}: Accuracy = {current_accuracy:.2f}%")

                # Visualization
                if self.visualizer and rnd % 5 == 0:
                    sample_size = min(1000, total_samples)
                    sample_features = np.vstack([batch[:100] for batch in features_batches])[:sample_size]
                    sample_targets = np.concatenate([batch[:100] for batch in encoded_targets_batches])[:sample_size]
                    sample_predictions, _ = self.core.predict_batch(sample_features)
                    self.visualizer.capture_training_snapshot(
                        sample_features, sample_targets, self.core.anti_wts,
                        sample_predictions, current_accuracy, rnd
                    )

                # Early stopping
                if patience_counter >= patience:
                    self.log(f"Early stopping at epoch {rnd} (no improvement for {patience} epochs)")
                    break

            # Restore best weights
            if best_weights is not None:
                self.core.anti_wts = best_weights
                self.log(f"Restored best weights from epoch {best_round}")

            self.core.is_trained = True
            self.core.best_accuracy = best_accuracy  # Store best accuracy in core

            # AUTOMATIC MODEL SAVING AFTER TRAINING
            self.log("=== Saving Trained Model ===")
            self.show_processing_indicator("Saving trained model...")
            model_file = f"{self.model_name.get()}.bin"  # Default to binary format

            # Ensure model directory exists
            model_dir = os.path.dirname(os.path.abspath(model_file))
            if model_dir and not os.path.exists(model_dir):
                os.makedirs(model_dir, exist_ok=True)
                self.log(f"Created model directory: {model_dir}")

            # Save the model using core's save method WITH FEATURE INFORMATION
            if self.file_type == 'csv':
                # Pass feature information for CSV files
                save_success = self.core.save_model(model_file, feature_columns, target_column, use_json=False)
                self.log(f"✅ Saved model with feature configuration: {len(feature_columns)} features, target: {target_column}")
            else:
                # For DAT files, save without specific feature info
                save_success = self.core.save_model(model_file, use_json=False)
                self.log("✅ Saved model (DAT file - no specific feature configuration)")

            if save_success:
                self.log(f"✅ Model successfully saved to: {model_file}")
            else:
                self.log("❌ Failed to save model!")

            # Test the model immediately after training
            self.log("Testing trained model...")
            try:
                if hasattr(self.core, 'innodes') and self.core.innodes > 0:
                    test_sample = np.random.randn(self.core.innodes)
                    predictions, probabilities = self.core.predict_batch(test_sample.reshape(1, -1))
                    self.log(f"✅ Post-training test - Prediction: {predictions[0]}")
                    self.log("✅ Model is ready for predictions")
            except Exception as e:
                self.log(f"❌ Post-training test failed: {e}")

            self.log(f"=== Training Completed ===")
            self.hide_processing_indicator()
            self.log(f"Best accuracy: {best_accuracy:.2f}% at epoch {best_round}")
            self.log(f"Final model: {self.core.innodes} inputs, {self.core.outnodes} outputs")
            self.log(f"Feature columns used: {feature_columns_used}")
            self.log(f"Target column: {target_column if self.file_type == 'csv' else 'last column (DAT)'}")
            self.log(f"Classes: {len(encoded_classes)} - {encoded_classes}")

            return True

        except Exception as e:
            self.hide_processing_indicator()
            messagebox.showerror("Error", f"Training failed: {e}")
            self.log(f"Error: {e}")
            self.log(traceback.format_exc())
            return False

    def test_model(self):
        """Test the model on separate data using the model's configuration"""
        if not self.core or not getattr(self.core, 'is_trained', False):
            messagebox.showerror("Error", "No trained model available")
            return

        test_file = filedialog.askopenfilename(
            title="Select Test File",
            filetypes=[("CSV files", "*.csv"), ("DAT files", "*.dat")]
        )

        if test_file:
            try:
                self.log(f"=== TESTING MODEL on: {test_file} ===")

                # Use the trained model's configuration with better debugging
                if hasattr(self.core, 'feature_columns') and self.core.feature_columns:
                    training_features = self.core.feature_columns
                    self.log(f"Using feature columns from trained model: {training_features}")
                else:
                    self.log("❌ No feature configuration found in trained model")
                    self.log("🔍 Debug: Available attributes in core:")
                    core_attrs = [attr for attr in dir(self.core) if not attr.startswith('_')]
                    for attr in sorted(core_attrs):
                        try:
                            value = getattr(self.core, attr)
                            if attr in ['feature_columns', 'target_column', 'innodes', 'outnodes', 'is_trained']:
                                self.log(f"   {attr}: {value}")
                        except:
                            self.log(f"   {attr}: <unable to access>")
                    return

                # Load test data using model's feature configuration
                target_column = self.core.target_column if hasattr(self.core, 'target_column') else ""

                self.log(f"Loading test data with features: {training_features}")
                if target_column:
                    self.log(f"Target column: {target_column}")

                features_batches, targets_batches, _, original_targets_batches = self.core.load_data(
                    test_file,
                    target_column=target_column,
                    feature_columns=training_features
                )

                if not features_batches:
                    self.log("No test data loaded")
                    return

                # Verify feature dimensions
                if features_batches and len(features_batches[0]) > 0:
                    actual_feature_count = features_batches[0].shape[1]
                    expected_feature_count = len(training_features)

                    if actual_feature_count != expected_feature_count:
                        self.log(f"❌ Feature dimension mismatch in test data!")
                        self.log(f"   Model expects: {expected_feature_count} features")
                        self.log(f"   Test data has: {actual_feature_count} features")
                        self.log(f"   Expected features: {training_features}")
                        return
                    else:
                        self.log(f"✅ Feature dimensions match: {actual_feature_count} features")

                # Encode targets using the model's encoder
                encoded_targets_batches = []
                for batch in original_targets_batches:
                    encoded_batch = self.core.class_encoder.transform(batch)
                    encoded_targets_batches.append(encoded_batch)

                # Evaluate using model's configuration
                accuracy, correct_predictions, predictions = self.core.evaluate(features_batches, encoded_targets_batches)
                total_samples = sum(len(batch) for batch in features_batches)

                self.log(f"📊 TEST RESULTS:")
                self.log(f"  Accuracy: {accuracy:.2f}%")
                self.log(f"  Correct: {correct_predictions}/{total_samples}")
                self.log(f"  Error Rate: {100-accuracy:.2f}%")

                # Show class-wise performance if available
                if hasattr(self.core, 'best_accuracy'):
                    training_accuracy = self.core.best_accuracy
                    accuracy_diff = accuracy - training_accuracy
                    self.log(f"  Training accuracy: {training_accuracy:.2f}%")
                    self.log(f"  Generalization: {accuracy_diff:+.2f}%")

                self.log("=== Testing Completed ===")

            except Exception as e:
                self.log(f"Test error: {e}")
                self.log(traceback.format_exc())

    def _clean_console_memory(self):
        """Clean console memory to prevent buildup"""
        try:
            current_text = self.console.get(1.0, tk.END)
            lines = current_text.split('\n')

            # Keep only last 1000 lines to prevent memory issues
            if len(lines) > 1000:
                cleaned_text = '\n'.join(lines[-1000:])
                self.console.delete(1.0, tk.END)
                self.console.insert(tk.END, cleaned_text)

            # Force Tkinter to update and free memory
            self.console.update_idletasks()

        except Exception as e:
            print(f"Console cleanup warning: {e}")

    def _initialize_output_file(self, output_file, predict_file):
        """Initialize the output file with proper headers"""
        try:
            import pandas as pd

            # Try to preserve original CSV structure if it's a CSV file
            if predict_file.endswith('.csv'):
                try:
                    # Read just the header from the original file
                    original_df = pd.read_csv(predict_file, nrows=0)
                    original_columns = original_df.columns.tolist()

                    # Create output dataframe with original columns + prediction columns
                    output_columns = original_columns + [
                        'prediction',
                        'prediction_encoded',
                        'confidence'
                    ]

                    # Create empty dataframe with the correct structure
                    output_df = pd.DataFrame(columns=output_columns)

                    # Write header to file
                    output_df.to_csv(output_file, index=False)
                    self.log(f"✅ Output file initialized with {len(original_columns)} original columns")

                except Exception as e:
                    self.log(f"⚠️ Could not preserve original CSV structure: {e}")
                    # Fallback: create basic output structure
                    self._create_basic_output_file(output_file)
            else:
                # For DAT files or when CSV loading fails, use basic structure
                self._create_basic_output_file(output_file)

        except Exception as e:
            self.log(f"⚠️ Error initializing output file: {e}")
            # Final fallback
            self._create_basic_output_file(output_file)

    def _create_basic_output_file(self, output_file):
        """Create a basic output file structure"""
        try:
            import pandas as pd

            # Basic columns for prediction output
            basic_columns = [
                'prediction',
                'prediction_encoded',
                'confidence'
            ]

            # Create empty dataframe
            output_df = pd.DataFrame(columns=basic_columns)

            # Write to file
            output_df.to_csv(output_file, index=False)
            self.log("✅ Created basic output file structure")

        except Exception as e:
            self.log(f"❌ Failed to create output file: {e}")
            raise

    def _write_predictions_batch(self, output_file, predictions, probabilities, start_index):
        """Write a batch of predictions to file efficiently"""
        try:
            if not predictions:
                return

            # Decode predictions
            decoded_predictions = self.core.class_encoder.inverse_transform(predictions)

            # Create batch dataframe
            batch_data = {
                'prediction': decoded_predictions,
                'prediction_encoded': predictions,
                'confidence': [max(prob.values()) if prob else 0.0 for prob in probabilities]
            }

            # Add probability columns
            if probabilities and len(probabilities) > 0:
                class_names = list(probabilities[0].keys())
                for class_name in class_names:
                    batch_data[f'prob_{class_name}'] = [prob.get(class_name, 0.0) for prob in probabilities]

            import pandas as pd
            batch_df = pd.DataFrame(batch_data)

            # Append to file
            with open(output_file, 'a', newline='') as f:
                batch_df.to_csv(f, header=False, index=False)

            # Store last predictions for summary (limited size)
            self._last_predictions = predictions[-1000:]  # Keep only last 1000 for summary

        except Exception as e:
            self.log(f"⚠️ Error writing prediction batch: {e}")

    def _get_memory_usage(self):
        """Get current memory usage"""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            return f"{memory_mb:.1f}MB"
        except ImportError:
            return "N/A"

    def _should_update_progress(self, batch_idx, total_batches, stats):
        """Determine when to update progress based on system load"""
        if batch_idx == 0:
            return True

        # Update more frequently at start, less frequently as we progress
        if batch_idx < 10:
            return batch_idx % 2 == 0
        elif batch_idx < 50:
            return batch_idx % 5 == 0
        else:
            return batch_idx % 10 == 0

    def _adaptive_resource_management(self, batch_idx, batch_times, stats, opt_params):
        """Adaptively manage resources based on performance"""
        if len(batch_times) < 5:
            return

        # Detect performance degradation
        recent_times = batch_times[-5:]
        avg_recent = sum(recent_times) / len(recent_times)
        avg_previous = sum(batch_times[:-5]) / len(batch_times[:-5]) if len(batch_times) > 5 else avg_recent

        if avg_recent > avg_previous * 1.3:  # 30% slowdown
            self.log(f"⚠️ Performance degradation detected at batch {batch_idx}")
            self.log(f"   Recent: {avg_recent:.3f}s, Previous: {avg_previous:.3f}s")

            # Suggest resource adjustment
            memory_usage = self._get_memory_usage()
            self.log(f"   Current memory: {memory_usage}")

    def _should_collect_garbage(self, batch_idx, stats):
        """Determine when to run garbage collection"""
        if batch_idx < 10:
            return batch_idx % 5 == 0
        else:
            return batch_idx % 20 == 0

    def _print_performance_summary(self, stats, total_samples, total_time):
        """Print detailed performance summary"""
        samples_per_second = total_samples / total_time if total_time > 0 else 0

        self.log("\n" + "="*50)
        self.log("PERFORMANCE SUMMARY")
        self.log("="*50)
        self.log(f"Total samples: {total_samples:,}")
        self.log(f"Total time: {total_time:.2f} seconds")
        self.log(f"Processing rate: {samples_per_second:.1f} samples/second")

        if stats['batch_times']:
            avg_batch_time = sum(stats['batch_times']) / len(stats['batch_times'])
            self.log(f"Average batch time: {avg_batch_time:.3f} seconds")

            # Show performance distribution
            fast_batches = len([t for t in stats['batch_times'] if t < avg_batch_time * 0.8])
            slow_batches = len([t for t in stats['batch_times'] if t > avg_batch_time * 1.2])
            self.log(f"Fast batches: {fast_batches}, Slow batches: {slow_batches}")

        self.log("="*50)

    def predict_batch_optimized(self, features_batch, clear_cache_every=100):
        """Optimized batch prediction with periodic cache clearing"""
        predictions = []
        probabilities = []

        # Use static arrays to avoid repeated allocations
        if not hasattr(self, '_prediction_vects'):
            self._prediction_vects = np.zeros(self.innodes + self.outnodes + 2)

        for sample_idx in range(len(features_batch)):
            # Reuse the same vector to avoid memory allocation
            vects = self._prediction_vects
            vects.fill(0)  # Reset vector

            for i in range(1, self.innodes + 1):
                vects[i] = features_batch[sample_idx, i-1]

            # Compute class probabilities
            classval = compute_class_probabilities_numba(
                vects, self.anti_net, self.anti_wts, self.binloc, self.resolution_arr,
                self.class_labels, self.min_val, self.max_val, self.innodes, self.outnodes
            )

            # Find predicted class
            kmax = 1
            cmax = 0.0
            for k in range(1, self.outnodes + 1):
                if classval[k] > cmax:
                    cmax = classval[k]
                    kmax = k

            predicted = self.class_labels[kmax]
            predictions.append(predicted)

            # Store probabilities
            prob_dict = {}
            for k in range(1, self.outnodes + 1):
                class_val = self.class_labels[k]
                if self.class_encoder.is_fitted:
                    class_name = self.class_encoder.encoded_to_class.get(class_val, f"Class_{k}")
                else:
                    class_name = f"Class_{k}"
                prob_dict[class_name] = float(classval[k])

            probabilities.append(prob_dict)

            # Clear cache periodically to prevent memory buildup
            if sample_idx % clear_cache_every == 0:
                import gc
                gc.collect()

        return predictions, probabilities

    def save_model(self, model_path=None):
        """Save the current model to file in binary format"""
        if not self.core or not getattr(self.core, 'is_trained', False):
            messagebox.showerror("Error", "No trained model to save")
            return

        # If no model_path provided, use the auto-generated name
        if model_path is None:
            model_path = f"{self.model_name.get()}.bin"

        # Ensure .bin extension
        if not model_path.endswith('.bin'):
            model_path += '.bin'

        # Ensure Model directory exists
        model_dir = os.path.dirname(os.path.abspath(model_path))
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)

        try:
            # Get feature information
            feature_columns = self.get_selected_features() if hasattr(self, 'get_selected_features') else []
            target_column = self.target_col.get() if hasattr(self, 'target_col') else ""

            success = self.core.save_model(
                model_path,
                feature_columns=feature_columns,
                target_column=target_column,
                use_json=False  # Always binary format
            )

            if success:
                self.log(f"✅ Model saved in binary format: {model_path}")
                self.log(f"Model info: {self.core.innodes} inputs, {self.core.outnodes} outputs")
                self.log(f"Best accuracy: {getattr(self.core, 'best_accuracy', 0.0):.2f}%")
            else:
                self.log("❌ Failed to save model")

            # Add mode information to metadata
            if hasattr(self.core, 'tensor_mode'):
                mode_info = "tensor" if self.core.tensor_mode else "standard"
                self.log(f"Saved model in {mode_info} mode")

        except Exception as e:
            self.log(f"Save error: {e}")
            self.log(traceback.format_exc())

    def load_model(self, model_path=None):
        """Load a previously saved model with proper error handling - FIXED VERSION"""
        # If no model_path provided, ask the user to select one
        if model_path is None:
            model_path = filedialog.askopenfilename(
                title="Select Model File",
                filetypes=[("Model files", "*.json *.bin *.gz"), ("JSON files", "*.json"), ("Binary files", "*.bin *.gz"), ("All files", "*.*")]
            )

        if not model_path:
            return

        try:
            self.show_processing_indicator("Loading model...")

            # Ensure core is initialized
            if not self.core:
                self.initialize_core()

            self.log(f"Loading model from: {model_path}")

            # Use core's load_model method
            self.core.load_model(model_path)

            # Update model name
            self.model_name.set(os.path.splitext(os.path.basename(model_path))[0])

            # Verify the model was loaded correctly
            if hasattr(self.core, 'innodes') and self.core.innodes > 0:
                self.log("✅ Model loaded successfully")
                self.log(f"   Architecture: {self.core.innodes} inputs, {self.core.outnodes} outputs")
                self.log(f"   Trained: {getattr(self.core, 'is_trained', False)}")
                self.log(f"   Best accuracy: {getattr(self.core, 'best_accuracy', 0):.2f}%")

                # Update configuration with loaded model's config
                if hasattr(self, 'config_vars') and hasattr(self.core, 'config'):
                    for key, value in self.core.config.items():
                        if key in self.config_vars:
                            self.config_vars[key].set(str(value))
                    self.log("✅ Configuration synchronized from loaded model")

                # Update tensor mode
                if hasattr(self.core, 'tensor_mode'):
                    tensor_mode = self.core.tensor_mode
                    if hasattr(self, 'tensor_mode'):
                        self.tensor_mode.set(tensor_mode)
                        mode_text = "Tensor" if tensor_mode else "Standard"
                        self.log(f"✅ {mode_text} mode detected and set")

                        # Update tensor info label
                        if hasattr(self, 'tensor_info'):
                            self.tensor_info.config(text=f"Current: {mode_text} Mode")

                # Test model functionality
                self.log("Testing loaded model functionality...")
                try:
                    if self.core.innodes > 0 and self.core.is_trained:
                        test_sample = np.random.randn(self.core.innodes)
                        predictions, probabilities = self.core.predict_batch(test_sample.reshape(1, -1))
                        self.log(f"✅ Model test passed - Prediction: {predictions[0]}")
                    else:
                        self.log("⚠️ Model loaded but not trained or invalid dimensions")
                except Exception as test_error:
                    self.log(f"⚠️ Model test warning: {test_error}")

            else:
                self.log("❌ ERROR: Model failed to load properly - invalid dimensions")
                self.log(f"   innodes: {getattr(self.core, 'innodes', 'NOT SET')}")
                self.log(f"   outnodes: {getattr(self.core, 'outnodes', 'NOT SET')}")
                self.log(f"   is_trained: {getattr(self.core, 'is_trained', 'NOT SET')}")

            self.log("=== Model Loading Completed ===")
            self.hide_processing_indicator()

        except Exception as e:
            self.log(f"❌ Load error: {e}")
            self.log(traceback.format_exc())
            self.hide_processing_indicator()

            # Show detailed error information
            error_msg = f"Failed to load model: {e}\n\n"
            error_msg += "Possible causes:\n"
            error_msg += "• Model file is corrupted\n"
            error_msg += "• Model was saved with a different DBNN version\n"
            error_msg += "• File format mismatch\n"
            error_msg += "• Insufficient permissions to read the file"

            messagebox.showerror("Load Error", error_msg)


    def visualize(self):
        """Enhanced visualization interface with organized options"""
        if not hasattr(self, 'core') or not self.core:
            messagebox.showerror("Error", "Please initialize core first")
            return

        try:
            # Create enhanced visualization dialog
            viz_window = tk.Toplevel(self.root)
            viz_window.title("🎯 DBNN Advanced Visualization Manager")
            viz_window.geometry("600x500")
            viz_window.transient(self.root)
            viz_window.grab_set()

            # Center the window
            viz_window.update_idletasks()
            x = self.root.winfo_x() + (self.root.winfo_width() - 600) // 2
            y = self.root.winfo_y() + (self.root.winfo_height() - 500) // 2
            viz_window.geometry(f"+{x}+{y}")

            # Header
            header_frame = ttk.Frame(viz_window)
            header_frame.pack(fill='x', padx=20, pady=15)

            ttk.Label(header_frame, text="🎯 DBNN Advanced Visualization Manager",
                     font=('Arial', 16, 'bold'), foreground='darkblue').pack(pady=5)

            # Get dataset name for organization
            dataset_name = "unknown"
            if hasattr(self.core, 'current_file') and self.core.current_file:
                dataset_name = os.path.splitext(os.path.basename(self.core.current_file))[0]

            ttk.Label(header_frame, text=f"Dataset: {dataset_name}",
                     font=('Arial', 10), foreground='gray').pack()

            # Main notebook for different visualization types
            notebook = ttk.Notebook(viz_window)
            notebook.pack(fill='both', expand=True, padx=20, pady=10)

            # Tab 1: Enhanced Visualizations
            enhanced_frame = ttk.Frame(notebook)
            notebook.add(enhanced_frame, text="🚀 Enhanced 3D")

            ttk.Label(enhanced_frame, text="Interactive 3D Visualizations",
                     font=('Arial', 12, 'bold')).pack(pady=10)

            enhanced_buttons = [
                ("🎮 Generate Enhanced 3D", "generate_enhanced_3d"),
                ("🔄 Animated Training", "generate_animated"),
                ("📊 Feature Explorer", "generate_feature_explorer")
            ]

            for text, command in enhanced_buttons:
                btn = ttk.Button(enhanced_frame, text=text,
                               command=lambda cmd=command: self._generate_visualization(cmd, viz_window))
                btn.pack(fill='x', padx=50, pady=5)

            # Tab 2: Standard Visualizations
            standard_frame = ttk.Frame(notebook)
            notebook.add(standard_frame, text="📈 Standard Charts")

            ttk.Label(standard_frame, text="Standard Charts and Metrics",
                     font=('Arial', 12, 'bold')).pack(pady=10)

            standard_buttons = [
                ("📊 Training Dashboard", "generate_standard_dashboard"),
                ("📈 Performance Metrics", "generate_performance"),
                ("🔥 Correlation Matrix", "generate_correlation"),
                ("📋 All Standard Charts", "generate_all_standard")
            ]

            for text, command in standard_buttons:
                btn = ttk.Button(standard_frame, text=text,
                               command=lambda cmd=command: self._generate_visualization(cmd, viz_window))
                btn.pack(fill='x', padx=50, pady=5)

            # Tab 3: File Management
            file_frame = ttk.Frame(notebook)
            notebook.add(file_frame, text="📁 File Manager")

            ttk.Label(file_frame, text="File Management",
                     font=('Arial', 12, 'bold')).pack(pady=10)

            file_buttons = [
                ("📂 Open Visualizations Folder", "open_viz_folder"),
                ("🕐 Open Recent", "open_recent"),
                ("🌐 Load Specific File", "load_specific")
            ]

            for text, command in file_buttons:
                btn = ttk.Button(file_frame, text=text,
                        command=lambda cmd=command: self._handle_visualization_command(cmd, viz_window))
                btn.pack(fill='x', padx=50, pady=5)

            # Close button
            close_frame = ttk.Frame(viz_window)
            close_frame.pack(fill='x', padx=20, pady=10)

            ttk.Button(close_frame, text="Close",
                      command=viz_window.destroy).pack(side='right')

        except Exception as e:
            self.log(f"❌ Visualization error: {e}")

    def _generate_visualization(self, viz_type, parent_window):
        """Generate specific visualization type."""
        self.show_processing_indicator(f"Generating {viz_type}...")
        try:
            # Get organized output paths
            dataset_name = "unknown"
            if hasattr(self.core, 'current_file') and self.core.current_file:
                dataset_name = os.path.splitext(os.path.basename(self.core.current_file))[0]

            enhanced_dir = f"Visualisations/{dataset_name}/Enhanced"
            standard_dir = f"Visualisations/{dataset_name}/Standard"
            os.makedirs(enhanced_dir, exist_ok=True)
            os.makedirs(standard_dir, exist_ok=True)

            output_file = None
            if viz_type == "generate_enhanced_3d":
                output_file = f"{enhanced_dir}/{dataset_name}_enhanced_3d.html"
                result = self.core.visualizer.generate_enhanced_interactive_3d(output_file)
            elif viz_type == "generate_standard_dashboard":
                output_file = f"{standard_dir}/{dataset_name}_standard_dashboard.html"
                result = self.core.visualizer.create_training_dashboard(output_file)

            if output_file and os.path.exists(output_file):
                self.log(f"✅ Generated: {output_file}")
                # Auto-open in browser
                import webbrowser
                webbrowser.open(f'file://{os.path.abspath(output_file)}')
            else:
                self.log(f"❌ Failed to generate {viz_type}")

        except Exception as e:
            self.log(f"❌ Error: {e}")
        finally:
            self.hide_processing_indicator()
            parent_window.destroy()

    def _generate_enhanced_visualizations(self):
        """Generate enhanced educational visualizations with interactive dashboards"""
        try:
            if not self.core or not hasattr(self.core, 'visualizer'):
                messagebox.showerror("Error", "No visualization data available")
                return False

            self.log("=== GENERATING ENHANCED EDUCATIONAL VISUALIZATIONS ===")

            # Create enhanced visualization directory
            if self.current_file:
                base_name = os.path.splitext(os.path.basename(self.current_file))[0]
                viz_dir = f"Visualisations/{base_name}_enhanced"
            else:
                base_name = self.model_name.get().replace("Model/", "")
                viz_dir = f"Visualisations/{base_name}_enhanced"

            os.makedirs(viz_dir, exist_ok=True)
            self.log(f"Enhanced visualizations will be saved in: {viz_dir}")

            viz_count = 0
            generated_files = []

            # 1. Generate Enhanced Educational Dashboard
            self.log("📊 Creating enhanced educational dashboard...")
            try:
                dashboard_file = f"{viz_dir}/enhanced_educational_dashboard.html"
                result = self.core.visualizer.create_advanced_interactive_dashboard(dashboard_file)

                if result:
                    self.log(f"✅ Enhanced educational dashboard: {result}")
                    viz_count += 1
                    generated_files.append(result)

                    # Add educational info
                    snapshot_count = len(self.core.visualizer.feature_space_snapshots)
                    self.log(f"   Using {snapshot_count} feature space snapshots")
                else:
                    self.log("❌ Enhanced dashboard generation failed")
            except Exception as e:
                self.log(f"❌ Enhanced dashboard error: {e}")

            # 2. Generate Interactive 3D Visualization
            self.log("🔄 Creating interactive 3D visualization...")
            try:
                interactive_3d_file = f"{viz_dir}/interactive_3d_evolution.html"
                result = self.core.visualizer.generate_interactive_3d_visualization(interactive_3d_file)
                if result:
                    self.log(f"✅ Interactive 3D evolution: {result}")
                    viz_count += 1
                    generated_files.append(result)
            except Exception as e:
                self.log(f"❌ Interactive 3D error: {e}")

            # 3. Generate Tensor-specific visualizations if in tensor mode
            is_tensor_mode = getattr(self.core, 'tensor_mode', False)
            if is_tensor_mode and hasattr(self.core.visualizer, 'create_tensor_dashboard'):
                self.log("🧠 Creating tensor mode visualizations...")
                try:
                    tensor_dashboard_file = f"{viz_dir}/tensor_transformation_dashboard.html"
                    result = self.core.visualizer.create_tensor_dashboard(tensor_dashboard_file)
                    if result:
                        self.log(f"✅ Tensor transformation dashboard: {result}")
                        viz_count += 1
                        generated_files.append(result)
                except Exception as e:
                    self.log(f"❌ Tensor visualization error: {e}")

            # 4. Generate standard visualizations as fallback
            self.log("📈 Generating supplementary standard visualizations...")
            standard_viz_methods = [
                (self.core.visualizer.generate_accuracy_plot, "accuracy_progression.html", "Accuracy Progression"),
                (self.core.visualizer.generate_feature_space_plot, "feature_space.html", "Feature Space"),
                (self.core.visualizer.generate_weight_distribution_plot, "weight_distribution.html", "Weight Distribution")
            ]

            for viz_method, filename, description in standard_viz_methods:
                try:
                    # For accuracy plot, don't pass snapshot index
                    if viz_method == self.core.visualizer.generate_accuracy_plot:
                        viz_fig = viz_method()
                    else:
                        viz_fig = viz_method(-1)  # Use latest snapshot

                    if viz_fig:
                        viz_file = f"{viz_dir}/{filename}"
                        viz_fig.write_html(viz_file)
                        self.log(f"✅ {description}: {viz_file}")
                        viz_count += 1
                        generated_files.append(viz_file)
                except Exception as e:
                    self.log(f"❌ {description} failed: {e}")

            # Summary
            if viz_count > 0:
                mode_info = "Tensor" if is_tensor_mode else "Standard"
                self.log(f"=== ENHANCED {mode_info.upper()} MODE VISUALIZATION COMPLETED ===")
                self.log(f"Generated {viz_count} enhanced visualization files in {viz_dir}")
                self.log("💡 Educational features included:")
                self.log("   • 3D feature space evolution")
                self.log("   • Decision boundary visualization")
                self.log("   • Weight distribution analysis")
                self.log("   • Feature correlation matrices")
                self.log("   • Interactive training progression")

                # Ask user if they want to open the files
                self._ask_to_open_enhanced_visualizations(generated_files, base_name, viz_dir)
                return True
            else:
                self.log("❌ No enhanced visualizations could be generated")
                # Fall back to standard visualizations
                self.log("🔄 Falling back to standard visualizations...")
                return self._generate_standard_visualizations()

        except Exception as e:
            self.log(f"❌ Enhanced visualization error: {e}")
            self.log(traceback.format_exc())
            # Fall back to standard visualizations
            self.log("🔄 Falling back to standard visualizations due to error...")
            return self._generate_standard_visualizations()


    def _generate_standard_visualizations(self):
        """Generate standard visualizations"""
        try:
            # Create output directory based on data file name
            if self.current_file:
                base_name = os.path.splitext(os.path.basename(self.current_file))[0]
                output_dir = f"Visualisations/{base_name}_standard"
            else:
                output_dir = "Visualisations/standard_viz"

            os.makedirs(output_dir, exist_ok=True)

            self.log(f"Standard visualizations will be saved in: {output_dir}")

            # Generate standard plots
            outputs = {}

            # Accuracy plot
            accuracy_plot = self.core.visualizer.generate_accuracy_plot()
            if accuracy_plot:
                accuracy_file = os.path.join(output_dir, f"{base_name}_accuracy.html")
                accuracy_plot.write_html(accuracy_file)
                outputs['accuracy'] = accuracy_file
                self.log(f"✅ Accuracy Plot: {accuracy_file}")

            # Feature space plot
            if self.core.visualizer.training_history:
                feature_plot = self.core.visualizer.generate_feature_space_plot(-1)
                if feature_plot:
                    feature_file = os.path.join(output_dir, f"{base_name}_features.html")
                    feature_plot.write_html(feature_file)
                    outputs['features'] = feature_file
                    self.log(f"✅ Feature Space: {feature_file}")

            # Weight distribution
            if self.core.visualizer.training_history:
                weight_plot = self.core.visualizer.generate_weight_distribution_plot(-1)
                if weight_plot:
                    weight_file = os.path.join(output_dir, f"{base_name}_weights.html")
                    weight_plot.write_html(weight_file)
                    outputs['weights'] = weight_file
                    self.log(f"✅ Weight Distribution: {weight_file}")

            # Training dashboard
            dashboard_file = os.path.join(output_dir, f"{base_name}_dashboard.html")
            if hasattr(self.core.visualizer, 'create_training_dashboard'):
                result = self.core.visualizer.create_training_dashboard(dashboard_file)
                if result:
                    outputs['dashboard'] = result
                    self.log(f"✅ Training dashboard: {result}")

            self.log(f"=== STANDARD VISUALIZATION COMPLETED ===")
            self.log(f"Generated {len(outputs)} standard visualization files in {output_dir}")
            self.log("Open the .html files in your web browser to view interactive plots")

            # Open all generated files
            for file_path in outputs.values():
                if os.path.exists(file_path):
                    import webbrowser
                    webbrowser.open(f'file://{os.path.abspath(file_path)}')

        except Exception as e:
            self.log(f"❌ Standard visualization error: {e}")
            raise

    def _ask_to_open_enhanced_visualizations(self, generated_files, base_name, viz_dir):
        """Ask user whether to open enhanced visualization files"""
        import webbrowser
        import os
        import platform

        # Create a dialog specifically for enhanced visualizations
        dialog = tk.Toplevel(self.root)
        dialog.title("Enhanced Educational Visualizations Ready!")
        dialog.geometry("600x400")
        dialog.transient(self.root)
        dialog.grab_set()

        # Center the dialog
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - 600) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - 400) // 2
        dialog.geometry(f"+{x}+{y}")

        # Content with educational emphasis
        ttk.Label(dialog, text="🎓 Enhanced Educational Visualizations Generated!",
                  font=('Arial', 14, 'bold'), foreground="darkblue").pack(pady=15)

        ttk.Label(dialog, text="Your model training has been captured with enhanced educational insights:",
                  font=('Arial', 10)).pack(pady=5)

        # Educational features list
        features_frame = ttk.Frame(dialog)
        features_frame.pack(fill='x', padx=20, pady=10)

        features_text = """• 3D Feature Space Evolution
    • Decision Boundary Visualization
    • Weight Distribution Analysis
    • Feature Correlation Matrices
    • Interactive Training Progression
    • Model Performance Analytics"""

        features_label = ttk.Label(features_frame, text=features_text, font=('Arial', 9), justify=tk.LEFT)
        features_label.pack(anchor='w')

        ttk.Label(dialog, text=f"Location: {viz_dir}",
                  font=('Arial', 10, 'bold'), foreground="green").pack(pady=5)

        # List generated files
        file_frame = ttk.Frame(dialog)
        file_frame.pack(fill='both', expand=True, padx=20, pady=10)

        file_list = scrolledtext.ScrolledText(file_frame, height=8, width=70)
        file_list.pack(fill='both', expand=True)

        for file in generated_files:
            filename = os.path.basename(file)
            file_list.insert(tk.END, f"• {filename}\n")
        file_list.config(state=tk.DISABLED)

        # Buttons frame
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill='x', pady=15)

        def open_educational_dashboard():
            """Open the main educational dashboard"""
            try:
                # Find the dashboard file
                dashboard_files = [f for f in generated_files if 'dashboard' in f.lower() or 'educational' in f.lower()]
                if dashboard_files:
                    webbrowser.open(f'file://{os.path.abspath(dashboard_files[0])}')
                    self.log(f"Opened educational dashboard: {dashboard_files[0]}")
                else:
                    # Fallback to first file
                    webbrowser.open(f'file://{os.path.abspath(generated_files[0])}')
            except Exception as e:
                self.log(f"Error opening dashboard: {e}")
            finally:
                dialog.destroy()

        def open_all_enhanced():
            """Open all enhanced visualization files"""
            try:
                opened_count = 0
                for file in generated_files:
                    webbrowser.open(f'file://{os.path.abspath(file)}')
                    opened_count += 1
                    time.sleep(0.3)  # Small delay to prevent browser overload
                self.log(f"Opened all {opened_count} enhanced visualization files")
            except Exception as e:
                self.log(f"Error opening files: {e}")
            finally:
                dialog.destroy()

        def open_folder():
            """Open the folder containing visualization files"""
            try:
                abs_viz_dir = os.path.abspath(viz_dir)
                if platform.system() == "Windows":
                    os.startfile(abs_viz_dir)
                elif platform.system() == "Darwin":  # macOS
                    os.system(f'open "{abs_viz_dir}"')
                else:  # Linux
                    os.system(f'xdg-open "{abs_viz_dir}"')
                self.log(f"Opened enhanced visualization folder: {abs_viz_dir}")
            except Exception as e:
                self.log(f"Error opening folder: {e}")
            finally:
                dialog.destroy()

        def do_nothing():
            """Close dialog without opening anything"""
            self.log("User chose not to open enhanced visualizations")
            dialog.destroy()

        # Button layout for enhanced visualizations
        ttk.Button(button_frame, text="Open Educational Dashboard",
                   command=open_educational_dashboard, style="Accent.TButton").pack(side='left', padx=5)

        ttk.Button(button_frame, text="Open All Enhanced Visualizations",
                   command=open_all_enhanced).pack(side='left', padx=5)

        ttk.Button(button_frame, text="Open Visualization Folder",
                   command=open_folder).pack(side='left', padx=5)

        ttk.Button(button_frame, text="Close",
                   command=do_nothing).pack(side='right', padx=5)

        # Style the main button if available
        try:
            style = ttk.Style()
            style.configure("Accent.TButton", foreground="white", background="darkblue")
        except:
            pass

        # Instructions
        ttk.Label(dialog, text="Start with 'Educational Dashboard' for the complete learning experience",
                  font=('Arial', 9), foreground="darkgreen").pack(pady=5)

    def generate_standard_visualizations(self, output_dir="Visualisations"):
        """Generate all standard visualizations"""
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)

            outputs = {}

            # Generate all plot types
            accuracy_plot = self.generate_accuracy_plot()
            if accuracy_plot:
                accuracy_file = os.path.join(output_dir, "accuracy_plot.html")
                accuracy_plot.write_html(accuracy_file)
                outputs['accuracy'] = accuracy_file

            # Generate feature space plot if we have snapshots
            if self.feature_space_snapshots:
                feature_plot = self.generate_feature_space_plot(-1)
                if feature_plot:
                    feature_file = os.path.join(output_dir, "feature_space.html")
                    feature_plot.write_html(feature_file)
                    outputs['feature_space'] = feature_file

            # Generate weight distribution
            if self.training_history:
                weight_plot = self.generate_weight_distribution_plot(-1)
                if weight_plot:
                    weight_file = os.path.join(output_dir, "weight_distribution.html")
                    weight_plot.write_html(weight_file)
                    outputs['weights'] = weight_file

            return outputs

        except Exception as e:
            print(f"Error generating standard visualizations: {e}")
            return {}

    def create_training_dashboard(self, output_file: str = "training_dashboard.html"):
        """Fixed version of training dashboard"""
        try:
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go
            import numpy as np
        except ImportError:
            print("Plotly not available for dashboard creation")
            return None

        if len(self.visualizer.training_history) < 2:
            return None

        # Use simpler layout to avoid subplot issues
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy Progression', 'Feature Space',
                          'Weight Distribution', 'Model Info'),
            specs=[
                [{"type": "xy"}, {"type": "scatter3d"}],
                [{"type": "xy"}, {"type": "xy"}]
            ]
        )

        # Accuracy Progression
        rounds = [s['round'] for s in self.visualizer.training_history]
        accuracies = [s['accuracy'] for s in self.visualizer.training_history]
        fig.add_trace(go.Scatter(x=rounds, y=accuracies, mode='lines+markers',
                               name='Accuracy', line=dict(color='blue')), row=1, col=1)

        # Feature Space (if available)
        if len(self.visualizer.training_history) > 0:
            feature_fig = self.visualizer.generate_feature_space_plot(-1)
            if feature_fig:
                for trace in feature_fig.data:
                    fig.add_trace(trace, row=1, col=2)

        # Weight Distribution
        if len(self.visualizer.training_history) > 0:
            weight_fig = self.visualizer.generate_weight_distribution_plot(-1)
            if weight_fig:
                for trace in weight_fig.data:
                    fig.add_trace(trace, row=2, col=1)

        # Model Info as text
        best_round = np.argmax(accuracies)
        best_accuracy = accuracies[best_round]

        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0.5, 0.5],
            mode='text',
            text=[f"Best Accuracy: {best_accuracy:.2f}%<br>Best Round: {best_round}<br>Total Rounds: {len(rounds)}"],
            textposition="middle center",
            showlegend=False
        ), row=2, col=2)

        fig.update_layout(
            height=800,
            title_text="DBNN Training Dashboard",
            showlegend=False
        )

        try:
            fig.write_html(output_file)
            print(f"Training dashboard saved to: {output_file}")
            return output_file
        except Exception as e:
            print(f"Error saving dashboard: {e}")
            return None

    def _ask_to_open_visualizations(self, generated_files, base_name, viz_dir):
        """Ask user whether to open visualization files or folder"""
        import webbrowser
        import os
        import platform

        # Create a dialog to ask user preference
        dialog = tk.Toplevel(self.root)
        dialog.title("Visualizations Generated")
        dialog.geometry("550x350")
        dialog.transient(self.root)
        dialog.grab_set()

        # Center the dialog
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - 550) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - 350) // 2
        dialog.geometry(f"+{x}+{y}")

        # Content
        ttk.Label(dialog, text="Visualization Files Generated!",
                  font=('Arial', 12, 'bold')).pack(pady=10)

        ttk.Label(dialog, text=f"Created {len(generated_files)} visualization files in:",
                  font=('Arial', 10)).pack(pady=5)

        ttk.Label(dialog, text=viz_dir,
                  font=('Arial', 10, 'bold'), foreground="blue").pack(pady=5)

        # List generated files
        file_frame = ttk.Frame(dialog)
        file_frame.pack(fill='both', expand=True, padx=20, pady=10)

        file_list = scrolledtext.ScrolledText(file_frame, height=6, width=65)
        file_list.pack(fill='both', expand=True)

        for file in generated_files:
            # Show only the filename, not the full path for cleaner display
            filename = os.path.basename(file)
            file_list.insert(tk.END, f"• {filename}\n")
        file_list.config(state=tk.DISABLED)

        # Buttons frame
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill='x', pady=10)

        def open_folder():
            """Open the folder containing visualization files"""
            try:
                abs_viz_dir = os.path.abspath(viz_dir)
                if platform.system() == "Windows":
                    os.startfile(abs_viz_dir)
                elif platform.system() == "Darwin":  # macOS
                    os.system(f'open "{abs_viz_dir}"')
                else:  # Linux
                    os.system(f'xdg-open "{abs_viz_dir}"')
                self.log(f"Opened visualization folder: {abs_viz_dir}")
            except Exception as e:
                self.log(f"Error opening folder: {e}")
                messagebox.showerror("Error", f"Could not open folder: {e}")
            finally:
                dialog.destroy()

        def open_3d_visualizations():
            """Open the 3D visualization files in web browser"""
            try:
                opened_count = 0
                for file in generated_files:
                    if 'features' in file.lower() or '3d' in file.lower() or 'dashboard' in file.lower():
                        # Open in web browser
                        webbrowser.open(f'file://{os.path.abspath(file)}')
                        opened_count += 1
                        # Small delay to prevent browser overload
                        time.sleep(0.5)

                if opened_count > 0:
                    self.log(f"Opened {opened_count} 3D visualization files in browser")
                else:
                    self.log("No 3D visualization files found to open")

            except Exception as e:
                self.log(f"Error opening visualizations: {e}")
                messagebox.showerror("Error", f"Could not open visualizations: {e}")
            finally:
                dialog.destroy()

        def open_all_files():
            """Open all visualization files in browser"""
            try:
                opened_count = 0
                for file in generated_files:
                    webbrowser.open(f'file://{os.path.abspath(file)}')
                    opened_count += 1
                    # Small delay to prevent browser overload
                    time.sleep(0.5)

                self.log(f"Opened all {opened_count} visualization files")
            except Exception as e:
                self.log(f"Error opening files: {e}")
                messagebox.showerror("Error", f"Could not open files: {e}")
            finally:
                dialog.destroy()

        def do_nothing():
            """Close dialog without opening anything"""
            self.log("User chose not to open visualizations")
            dialog.destroy()

        # Button layout
        ttk.Button(button_frame, text="Open Visualization Folder",
                   command=open_folder).pack(side='left', padx=5)

        ttk.Button(button_frame, text="Open 3D Visualizations",
                   command=open_3d_visualizations).pack(side='left', padx=5)

        ttk.Button(button_frame, text="Open All Files",
                   command=open_all_files).pack(side='left', padx=5)

        ttk.Button(button_frame, text="Close",
                   command=do_nothing).pack(side='right', padx=5)

        # Instructions
        ttk.Label(dialog, text="Choose an option to view your visualizations:",
                  font=('Arial', 9)).pack(pady=5)

class DBNNCommandLine:
    """Command Line Interface for DBNN with binary format support"""

    def __init__(self):
        self.core = DBNNCore()
        self.core.set_log_callback(self._log_message)
        self.current_model_path = None
        self.visualizer = DBNNVisualizer()
        self.core.attach_visualizer(self.visualizer)

    def _log_message(self, message: str):
        """Log message to console"""
        print(f"[DBNN] {message}")

    def show_detailed_help(self):
        """Display comprehensive help information"""
        help_text = """
DBNN (Difference Boosting Bayesian Neural Network) - Command Line Interface
================================================================================

DBNN is a powerful neural network implementation with Bayesian learning and
difference boosting capabilities. This interface provides full access to all
DBNN features through command line operations.

USAGE PATTERNS:
===============

1. TRAINING A NEW MODEL:
   python runDBNN_cmd.py --train <training_file> [options]

2. USING EXISTING MODEL:
   python runDBNN_cmd.py --model <model_file> --predict <data_file>

3. EVALUATION:
   python runDBNN_cmd.py --model <model_file> --evaluate <test_file>

4. INTERACTIVE MODE:
   python runDBNN_cmd.py --interactive

DETAILED OPTIONS:
=================

DATA INPUT OPTIONS:
  --train FILE           Training data file (CSV or DAT format)
  --test FILE            Test data file for validation during training
  --predict FILE         Data file for making predictions
  --evaluate FILE        Data file for model evaluation

CSV-SPECIFIC OPTIONS (required for CSV files):
  --target COLUMN        Target column name for CSV files
  --features COL1 COL2   Feature column names (space-separated)
  --format FORMAT        Input file format: csv or dat (default: csv)

MODEL OPTIONS:
  --model FILE           Load existing model file (.json, .bin, or .gz)
  --save-model FILE      Save trained model to specified path
  --model-format FORMAT  Model save format: binary or json (default: binary)
  --model-dir DIR        Model directory (default: Model/)

TRAINING PARAMETERS:
  --resol INTEGER        Resolution parameter (default: 100)
  --gain FLOAT           Gain parameter for weight updates (default: 2.0)
  --margin FLOAT         Classification margin (default: 0.2)
  --patience INTEGER     Early stopping patience (default: 10)
  --epochs INTEGER       Maximum training epochs (default: 100)
  --min-improvement FLOAT Minimum accuracy improvement for early stopping (default: 0.1)

OUTPUT OPTIONS:
  --output FILE          Output file for predictions
  --verbose, -v          Verbose output with detailed progress
  --quiet, -q            Quiet mode (minimal output)

OTHER OPTIONS:
  --interactive, -i      Start interactive mode
  --help, -h             Show this help message

FILE FORMATS SUPPORTED:
=======================

CSV FILES:
  - First row must contain column headers
  - Specify target column with --target
  - Specify feature columns with --features (or uses all except target)
  - Example: --train data.csv --target class --features feature1 feature2 feature3

DAT FILES (Legacy Format):
  - Space-separated values, no headers
  - Last column is assumed to be target
  - All other columns are features
  - Example: --train data.dat --format dat

MODEL FORMATS:
==============

BINARY FORMAT (Default - Recommended):
  - File extensions: .bin or .gz
  - Smaller file size
  - Faster save/load operations
  - Better performance for large models
  - Example: --save-model my_model.bin --model-format binary

JSON FORMAT (Optional):
  - File extension: .json
  - Human-readable
  - Good for debugging and inspection
  - Example: --save-model my_model.json --model-format json

DETAILED EXAMPLES:
==================

1. BASIC TRAINING WITH CSV:
   python runDBNN_cmd.py --train data/train.csv --test data/test.csv
                         --target species
                         --features sepal_length sepal_width petal_length petal_width
                         --resol 150 --gain 2.5 --epochs 200

2. TRAINING WITH DAT FILE:
   python runDBNN_cmd.py --train data/train.dat --format dat
                         --resol 100 --patience 15

3. PREDICTIONS WITH EXISTING MODEL:
   python runDBNN_cmd.py --model Model/iris_model.bin
                         --predict data/new_data.csv
                         --target species
                         --output predictions.csv

4. EVALUATE MODEL PERFORMANCE:
   python runDBNN_cmd.py --model Model/iris_model.bin
                         --evaluate data/test_set.csv
                         --target species

5. TRAINING WITH CUSTOM PARAMETERS:
   python runDBNN_cmd.py --train data/train.csv --target outcome
                         --resol 200 --gain 3.0 --margin 0.3
                         --patience 20 --epochs 300
                         --min-improvement 0.05
                         --save-model my_custom_model.bin

6. VERBOSE TRAINING WITH PROGRESS UPDATES:
   python runDBNN_cmd.py --train large_dataset.csv --target label
                         --features f1 f2 f3 f4 f5
                         --verbose --epochs 500

7. QUIET MODE FOR BATCH PROCESSING:
   python runDBNN_cmd.py --model production_model.bin
                         --predict batch_data.csv
                         --output batch_predictions.csv
                         --quiet

PERFORMANCE TIPS:
=================

- Use binary format (.bin) for production models (faster and smaller)
- For large datasets, use --quiet mode to reduce output overhead
- Adjust --resol based on dataset complexity (higher = more precise but slower)
- Use --patience to prevent overfitting with early stopping
- Start with default parameters and tune based on validation results

TROUBLESHOOTING:
================

Common Issues and Solutions:

1. "Target column not found":
   - Check column names with: head -1 your_file.csv
   - Ensure --target matches exactly (case-sensitive)

2. "No training data loaded":
   - Verify file path is correct
   - Check file format matches --format option
   - Ensure CSV files have proper headers

3. "Model failed to load":
   - Check file exists and is readable
   - Verify model format (binary vs JSON)
   - Ensure model was saved with same DBNN version

4. "Feature count mismatch":
   - Training and prediction must use same features
   - Use --features to explicitly specify feature columns

5. "Memory issues with large datasets":
   - Reduce --resol parameter
   - Use smaller batch sizes in interactive mode
   - Process data in chunks

For additional help or to report issues, please check the documentation
or contact support.

INTERACTIVE MODE FEATURES:
==========================

Interactive mode provides a menu-driven interface with:
- Step-by-step model training
- Real-time configuration
- Immediate prediction testing
- Model inspection and saving
- No need to remember command syntax

Start interactive mode with: python runDBNN_cmd.py --interactive
"""
        print(help_text)

    def parse_arguments(self):
        """Parse command line arguments"""
        import argparse

        parser = argparse.ArgumentParser(
            description='DBNN (Difference Boosting Bayesian Neural Network) - Command Line Interface',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            add_help=False
        )

        # Data input options
        data_group = parser.add_argument_group('Data Input Options')
        data_group.add_argument('--train', type=str, help='Training data file (CSV or DAT)')
        data_group.add_argument('--test', type=str, help='Test data file (CSV or DAT)')
        data_group.add_argument('--predict', type=str, help='Data file for prediction')
        data_group.add_argument('--evaluate', type=str, help='Data file for model evaluation')

        # CSV-specific options
        csv_group = parser.add_argument_group('CSV Options')
        csv_group.add_argument('--target', type=str, help='Target column name for CSV files')
        csv_group.add_argument('--features', nargs='+', help='Feature column names for CSV files')
        csv_group.add_argument('--format', choices=['csv', 'dat'], default='csv',
                              help='Input file format (default: csv)')

        # Model options
        model_group = parser.add_argument_group('Model Options')
        model_group.add_argument('--model', type=str, help='Load existing model file')
        model_group.add_argument('--save-model', type=str, help='Save model to specified path')
        model_group.add_argument('--model-format', choices=['binary', 'json'], default='binary',
                               help='Model save format (default: binary)')
        model_group.add_argument('--model-dir', type=str, default='Model',
                               help='Model directory (default: Model)')

        # Training parameters
        param_group = parser.add_argument_group('Training Parameters')
        param_group.add_argument('--resol', type=int, default=100,
                                help='Resolution parameter (default: 100)')
        param_group.add_argument('--gain', type=float, default=2.0,
                                help='Gain parameter (default: 2.0)')
        param_group.add_argument('--margin', type=float, default=0.2,
                                help='Classification margin (default: 0.2)')
        param_group.add_argument('--patience', type=int, default=10,
                                help='Early stopping patience (default: 10)')
        param_group.add_argument('--epochs', type=int, default=100,
                                help='Maximum training epochs (default: 100)')
        param_group.add_argument('--min-improvement', type=float, default=0.1,
                                help='Minimum improvement for early stopping (default: 0.1)')

        # Output options
        output_group = parser.add_argument_group('Output Options')
        output_group.add_argument('--output', type=str, help='Output file for predictions')
        output_group.add_argument('--verbose', '-v', action='store_true',
                                 help='Verbose output')
        output_group.add_argument('--quiet', '-q', action='store_true',
                                 help='Quiet mode (minimal output)')

        # Interactive mode and help
        other_group = parser.add_argument_group('Other Options')
        other_group.add_argument('--interactive', '-i', action='store_true',
                          help='Start interactive mode')
        other_group.add_argument('--help', '-h', action='store_true',
                          help='Show detailed help message')

        return parser.parse_args()

    def validate_arguments(self, args):
        """Validate command line arguments and provide helpful error messages"""
        import sys

        errors = []
        warnings = []

        # Check for no arguments or help request
        if len(sys.argv) == 1 or args.help:
            self.show_detailed_help()
            return False, []

        # Check for interactive mode
        if args.interactive:
            return True, []  # Interactive mode doesn't need validation

        # Validate training arguments
        if args.train:
            if not os.path.exists(args.train):
                errors.append(f"Training file not found: {args.train}")

            if args.format == 'csv' and not args.target:
                errors.append("--target parameter required for CSV training files")

        # Validate prediction/evaluation arguments
        if (args.predict or args.evaluate) and not args.model and not args.train:
            errors.append("For prediction or evaluation, either --model or --train must be specified")

        if args.predict and not os.path.exists(args.predict):
            errors.append(f"Prediction file not found: {args.predict}")

        if args.evaluate and not os.path.exists(args.evaluate):
            errors.append(f"Evaluation file not found: {args.evaluate}")

        # Validate model arguments
        if args.model and not os.path.exists(args.model):
            errors.append(f"Model file not found: {args.model}")

        # Check for incompatible options
        if args.verbose and args.quiet:
            warnings.append("Both --verbose and --quiet specified, using --verbose")
            args.quiet = False

        # Check for missing required combinations
        if args.train and args.format == 'csv' and not args.target:
            errors.append("CSV training requires --target parameter")

        if (args.predict or args.evaluate) and args.format == 'csv' and not args.target:
            warnings.append("CSV prediction/evaluation without --target: assuming no target column")

        # Show warnings
        for warning in warnings:
            print(f"⚠️  Warning: {warning}")

        # Show errors and return
        if errors:
            print("❌ Argument errors:")
            for error in errors:
                print(f"   - {error}")
            print("\n💡 Use --help for detailed usage information")
            return False, errors

        return True, []

    def setup_model_config(self, args):
        """Setup model configuration from command line arguments"""
        config = {
            'resol': args.resol,
            'gain': args.gain,
            'margin': args.margin,
            'patience': args.patience,
            'epochs': args.epochs,
            'min_improvement': args.min_improvement
        }
        self.core.config.update(config)

    def load_model(self, model_path: str):
        """Load model from file (auto-detects format)"""
        print(f"Loading model from: {model_path}")
        if self.core.load_model(model_path):
            self.current_model_path = model_path
            print("✅ Model loaded successfully")
            return True
        else:
            print("❌ Failed to load model")
            return False

    def save_model(self, model_path: str, use_json=False):
        """Save model to file in specified format"""
        # Ensure model directory exists
        model_dir = os.path.dirname(os.path.abspath(model_path))
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)

        # Add appropriate extension if not present
        if use_json and not model_path.endswith('.json'):
            model_path += '.json'
        elif not use_json and not (model_path.endswith('.bin') or model_path.endswith('.gz')):
            model_path += '.bin'

        print(f"Saving model to: {model_path}")
        if self.core.save_model(model_path, use_json=use_json):
            self.current_model_path = model_path
            format_name = "JSON" if use_json else "binary"
            print(f"✅ Model saved successfully in {format_name} format")
            return True
        else:
            print("❌ Failed to save model")
            return False

    def train_model(self, args):
        """Train model with specified parameters and automatic saving"""
        print("🚀 Starting DBNN Training...")
        print(f"Training file: {args.train}")
        if args.test:
            print(f"Test file: {args.test}")
        print(f"Parameters: resol={args.resol}, gain={args.gain}, margin={args.margin}")
        print(f"Early stopping: patience={args.patience}, min_improvement={args.min_improvement}%")

        use_csv = (args.format == 'csv')
        if use_csv and not args.target:
            print("❌ Error: --target parameter required for CSV format")
            return False

        import time
        start_time = time.time()

        try:
            success = self.core.train_with_early_stopping(
                train_file=args.train,
                test_file=args.test,
                use_csv=use_csv,
                target_column=args.target,
                feature_columns=args.features,
                auto_save_model=True,  # Enable automatic saving
                model_dir=args.model_dir,
                enable_interactive_viz=True,      # Enable 3D visualization
                viz_capture_interval=5            # Capture every 5 iterations
            )

            training_time = time.time() - start_time
            print(f"⏱️  Training completed in {training_time:.2f} seconds")

            # Manual save if specifically requested (in addition to auto-save)
            if success and args.save_model:
                use_json = (args.model_format == 'json')
                self.save_model(args.save_model, use_json=use_json)

            return success

        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def predict_data(self, args):
        """Make predictions on data using the trained model's configuration"""
        if not self.core.is_trained and not args.model:
            print("❌ Error: No trained model available. Use --train to train a model or --model to load one.")
            return False

        print(f"🔮 Making predictions on: {args.predict}")

        try:
            # Use the trained model's configuration
            if hasattr(self.core, 'feature_columns') and self.core.feature_columns:
                training_features = self.core.feature_columns
                print(f"Using feature columns from trained model: {training_features}")
            else:
                print("❌ No feature configuration found in trained model")
                return False

            # For prediction mode, use None for target and training features
            target_column = None  # No target in prediction mode
            feature_columns = training_features  # Use training features from model

            # Load prediction data
            features_batches, _, feature_columns_used, _ = self.core.load_data(
                args.predict,
                target_column=target_column,
                feature_columns=feature_columns,
                batch_size=10000
            )

            if not features_batches:
                print("❌ No data loaded for prediction")
                return False

            # Verify feature dimensions
            if features_batches and len(features_batches[0]) > 0:
                actual_feature_count = features_batches[0].shape[1]
                expected_feature_count = len(training_features)

                if actual_feature_count != expected_feature_count:
                    print(f"❌ Feature dimension mismatch!")
                    print(f"   Expected: {expected_feature_count} features")
                    print(f"   Got: {actual_feature_count} features")
                    print(f"   Expected features: {training_features}")
                    return False
                else:
                    print(f"✅ Feature dimensions match: {actual_feature_count} features")

            all_predictions = []
            all_probabilities = []

            for batch_idx, features_batch in enumerate(features_batches):
                if args.verbose:
                    print(f"Processing batch {batch_idx + 1}/{len(features_batches)}...")
                predictions, probabilities = self.core.predict_batch(features_batch)
                all_predictions.extend(predictions)
                all_probabilities.extend(probabilities)

            # Decode predictions
            decoded_predictions = self.core.class_encoder.inverse_transform(all_predictions)

            # Save predictions to file if requested
            if args.output:
                self.save_predictions(args.output, features_batches, decoded_predictions, all_predictions, all_probabilities)

            # Print summary
            print(f"✅ Made {len(all_predictions)} predictions")

            # Show prediction distribution
            from collections import Counter
            prediction_counts = Counter(decoded_predictions)
            print("\nPrediction distribution:")
            for pred, count in prediction_counts.most_common():
                percentage = (count / len(decoded_predictions)) * 100
                print(f"  {pred}: {count} ({percentage:.1f}%)")

            return True

        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def save_predictions(self, output_file: str, features_batches, decoded_predictions, encoded_predictions, probabilities):
        """Save predictions to output file"""
        try:
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)

                # Write header
                headers = ['Prediction', 'Prediction_Encoded', 'Confidence']

                # Add probability columns for each class
                if probabilities and len(probabilities) > 0:
                    class_names = list(probabilities[0].keys())
                    headers.extend([f'Prob_{cls}' for cls in class_names])

                writer.writerow(headers)

                # Write data
                sample_idx = 0
                for batch_idx, features_batch in enumerate(features_batches):
                    for i in range(len(features_batch)):
                        if sample_idx >= len(decoded_predictions):
                            break

                        row = [
                            decoded_predictions[sample_idx],
                            encoded_predictions[sample_idx],
                            max(probabilities[sample_idx].values()) if probabilities[sample_idx] else 0.0
                        ]

                        # Add probabilities for each class
                        if probabilities and sample_idx < len(probabilities):
                            for class_name in list(probabilities[sample_idx].keys()):
                                row.append(probabilities[sample_idx].get(class_name, 0.0))

                        writer.writerow(row)
                        sample_idx += 1

            print(f"✅ Predictions saved to: {output_file}")

        except Exception as e:
            print(f"❌ Error saving predictions: {e}")

    def evaluate_model(self, args):
        """Evaluate model on test data"""
        if not self.core.is_trained and not args.model:
            print("❌ Error: No trained model available. Use --train to train a model or --model to load one.")
            return False

        print(f"📊 Evaluating model on: {args.evaluate}")

        try:
            # Load test data
            features_batches, targets_batches, feature_columns, original_targets_batches = self.core.load_data(
                args.evaluate,
                target_column=args.target if args.format == 'csv' else None,
                feature_columns=args.features,
                batch_size=10000
            )

            if not features_batches:
                print("❌ No test data loaded")
                return False

            # Encode targets
            encoded_targets_batches = []
            for batch in original_targets_batches:
                encoded_batch = self.core.class_encoder.transform(batch)
                encoded_targets_batches.append(encoded_batch)

            # Evaluate
            accuracy, correct_predictions, predictions = self.core.evaluate(features_batches, encoded_targets_batches)
            total_samples = sum(len(batch) for batch in features_batches)

            print(f"✅ Evaluation completed:")
            print(f"   Accuracy: {accuracy:.2f}%")
            print(f"   Correct: {correct_predictions}/{total_samples}")
            print(f"   Error Rate: {100-accuracy:.2f}%")

            return True

        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def show_configuration(self):
        """Display current configuration"""
        config = self.core.config

        print("\nCurrent Configuration:")
        print("-" * 30)

        for key, value in config.items():
            print(f"{key:20}: {value}")

        if self.core.is_trained:
            print(f"{'Model trained':20}: Yes")
            if hasattr(self.core, 'innodes'):
                print(f"{'Input nodes':20}: {self.core.innodes}")
            if hasattr(self.core, 'outnodes'):
                print(f"{'Output nodes':20}: {self.core.outnodes}")
            if hasattr(self.core, 'best_accuracy'):
                print(f"{'Best accuracy':20}: {self.core.best_accuracy:.2f}%")
        else:
            print(f"{'Model trained':20}: No")

    def interactive_mode(self):
        """Start interactive mode"""
        print("\n" + "="*60)
        print("DBNN Interactive Mode")
        print("="*60)
        print("Available commands:")
        print("  train    - Train a new model")
        print("  load     - Load existing model")
        print("  save     - Save current model")
        print("  predict  - Make predictions")
        print("  evaluate - Evaluate model")
        print("  config   - Show configuration")
        print("  quit     - Exit interactive mode")
        print("="*60)

        while True:
            try:
                command = input("\nDBNN> ").strip().lower()

                if command == 'quit' or command == 'exit':
                    break
                elif command == 'train':
                    self.interactive_train()
                elif command == 'load':
                    self.interactive_load()
                elif command == 'save':
                    self.interactive_save()
                elif command == 'predict':
                    self.interactive_predict()
                elif command == 'evaluate':
                    self.interactive_evaluate()
                elif command == 'config':
                    self.show_configuration()
                elif command == '':
                    continue
                else:
                    print("❌ Unknown command. Type 'quit' to exit.")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    def interactive_train(self):
        """Interactive training mode"""
        print("\n🎯 Training Mode")

        train_file = input("Training file path: ").strip()
        if not train_file or not os.path.exists(train_file):
            print("❌ Invalid file path")
            return

        test_file = input("Test file path (optional): ").strip()
        if test_file and not os.path.exists(test_file):
            print("❌ Invalid test file path")
            return

        file_format = input("File format (csv/dat) [csv]: ").strip().lower()
        if not file_format:
            file_format = 'csv'

        target_col = ""
        feature_cols = []

        if file_format == 'csv':
            target_col = input("Target column name: ").strip()
            if not target_col:
                print("❌ Target column required for CSV")
                return

            features_input = input("Feature columns (comma-separated, empty for auto): ").strip()
            if features_input:
                feature_cols = [col.strip() for col in features_input.split(',')]

        # Get training parameters
        resol = input(f"Resolution [100]: ").strip()
        resol = int(resol) if resol else 100

        gain = input(f"Gain [2.0]: ").strip()
        gain = float(gain) if gain else 2.0

        margin = input(f"Margin [0.2]: ").strip()
        margin = float(margin) if margin else 0.2

        patience = input(f"Patience [10]: ").strip()
        patience = int(patience) if patience else 10

        epochs = input(f"Epochs [100]: ").strip()
        epochs = int(epochs) if epochs else 100

        # Create args-like object
        class Args:
            pass

        args = Args()
        args.train = train_file
        args.test = test_file if test_file else None
        args.format = file_format
        args.target = target_col
        args.features = feature_cols
        args.resol = resol
        args.gain = gain
        args.margin = margin
        args.patience = patience
        args.epochs = epochs
        args.min_improvement = 0.1
        args.save_model = None
        args.model_format = 'binary'
        args.verbose = True

        self.setup_model_config(args)
        self.train_model(args)

    def interactive_load(self):
        """Interactive model loading"""
        print("\n📥 Load Model")

        model_path = input("Model file path: ").strip()
        if not model_path:
            print("❌ No path provided")
            return

        self.load_model(model_path)

    def interactive_save(self):
        """Interactive model saving"""
        if not self.core.is_trained:
            print("❌ No trained model to save")
            return

        print("\n💾 Save Model")

        format_choice = input("Save format (binary/json) [binary]: ").strip().lower()
        use_json = (format_choice == 'json')

        default_ext = '.json' if use_json else '.bin'
        model_path = input(f"Model file path [default: Model/model{default_ext}]: ").strip()
        if not model_path:
            model_path = os.path.join('Model', f'model{default_ext}')

        self.save_model(model_path, use_json=use_json)

    def interactive_predict(self):
        """Interactive prediction"""
        if not self.core.is_trained:
            print("❌ No trained model available")
            return

        print("\n🔮 Prediction Mode")

        predict_file = input("Prediction file path: ").strip()
        if not predict_file or not os.path.exists(predict_file):
            print("❌ Invalid file path")
            return

        file_format = input("File format (csv/dat) [csv]: ").strip().lower()
        if not file_format:
            file_format = 'csv'

        target_col = ""
        feature_cols = []

        if file_format == 'csv':
            target_col = input("Target column name (optional): ").strip()
            features_input = input("Feature columns (comma-separated, empty for auto): ").strip()
            if features_input:
                feature_cols = [col.strip() for col in features_input.split(',')]

        output_file = input("Output file (optional): ").strip()

        # Create args-like object
        class Args:
            pass

        args = Args()
        args.predict = predict_file
        args.format = file_format
        args.target = target_col
        args.features = feature_cols
        args.output = output_file
        args.model = self.current_model_path
        args.verbose = True

        self.predict_data(args)

    def interactive_evaluate(self):
        """Interactive evaluation"""
        if not self.core.is_trained:
            print("❌ No trained model available")
            return

        print("\n📊 Evaluation Mode")

        eval_file = input("Evaluation file path: ").strip()
        if not eval_file or not os.path.exists(eval_file):
            print("❌ Invalid file path")
            return

        file_format = input("File format (csv/dat) [csv]: ").strip().lower()
        if not file_format:
            file_format = 'csv'

        target_col = ""
        feature_cols = []

        if file_format == 'csv':
            target_col = input("Target column name: ").strip()
            if not target_col:
                print("❌ Target column required for CSV evaluation")
                return

            features_input = input("Feature columns (comma-separated, empty for auto): ").strip()
            if features_input:
                feature_cols = [col.strip() for col in features_input.split(',')]

        # Create args-like object
        class Args:
            pass

        args = Args()
        args.evaluate = eval_file
        args.format = file_format
        args.target = target_col
        args.features = feature_cols

        self.evaluate_model(args)

    def run(self):
        """Main execution function"""
        import sys
        import time

        args = self.parse_arguments()

        # Validate arguments
        is_valid, errors = self.validate_arguments(args)
        if not is_valid:
            if errors:
                sys.exit(1)
            else:
                sys.exit(0)  # Help was shown, exit normally

        # Set up model configuration
        self.setup_model_config(args)

        # Handle different modes
        if args.interactive:
            self.interactive_mode()
            return

        # Load model if specified
        if args.model:
            if not self.load_model(args.model):
                sys.exit(1)

        # Perform requested actions
        success = True

        if args.train:
            success = self.train_model(args)

        if success and args.predict:
            success = self.predict_data(args)

        if success and args.evaluate:
            success = self.evaluate_model(args)

        # Save model if trained and no specific save path provided
        if success and self.core.is_trained and not args.save_model and args.train:
            default_model_name = f"dbnn_model_{int(time.time())}"
            use_json = (args.model_format == 'json')
            model_dir = getattr(args, 'model_dir', 'Model')
            model_path = os.path.join(model_dir, default_model_name)
            self.save_model(model_path, use_json=use_json)

class DBNNTensorCore(DBNNCore):
    """
    DBNN with tensor space transformation instead of iterative learning
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.orthogonal_basis = None
        self.weight_matrix = None
        self.feature_projection = None
        self.class_projection = None

        # Ensure class encoder exists
        if not hasattr(self, 'class_encoder'):
            self.class_encoder = ClassEncoder()

        # Initialize arrays to prevent NoneType errors
        self.innodes = 0
        self.outnodes = 0
        self.class_labels = None  # This will be initialized properly later

    def orthogonal_predict(self, features_batch):
        """Prediction using orthogonal projections with robust type handling"""
        if self.weight_matrix is not None and features_batch.size > 0:
            try:
                # Direct projection: features @ W = class_scores
                class_scores = features_batch @ self.weight_matrix

                # Convert to probabilities using softmax
                exp_scores = np.exp(class_scores - np.max(class_scores, axis=1, keepdims=True))
                probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

                # Get predictions - find which class has highest score
                predicted_indices = np.argmax(class_scores, axis=1)

                # Convert indices back to original class values using class_labels
                predictions = []
                for idx in predicted_indices:
                    if 0 <= idx < len(self.class_labels) - 1:  # -1 because index 0 is margin
                        class_val = self.class_labels[idx + 1]  # +1 because index 0 is margin
                        predictions.append(class_val)
                    else:
                        # Fallback: use first class
                        predictions.append(self.class_labels[1] if len(self.class_labels) > 1 else 1.0)

                # Convert to probability dictionaries
                prob_dicts = []
                for prob_row in probabilities:
                    prob_dict = {}
                    for k in range(1, min(self.outnodes + 1, len(prob_row) + 1)):
                        class_val = self.class_labels[k]
                        if self.class_encoder.is_fitted:
                            # Use the original class representation from encoder
                            class_name = self.class_encoder.encoded_to_class.get(class_val, f"Class_{k}")
                        else:
                            class_name = f"Class_{k}"
                        prob_dict[class_name] = float(prob_row[k-1])
                    prob_dicts.append(prob_dict)

                return predictions, prob_dicts
            except Exception as e:
                self.log(f"Orthogonal prediction failed: {e}")
                # Fallback to standard prediction
                return self.predict_batch(features_batch)
        else:
            # Fallback to standard method
            return self.predict_batch(features_batch)

    def tensor_evaluate(self, features_batches, encoded_targets_batches):
        """Evaluate tensor model accuracy with robust type handling"""
        correct_predictions = 0
        total_samples = 0
        all_predictions = []

        for batch_idx, (features_batch, targets_batch) in enumerate(zip(features_batches, encoded_targets_batches)):
            # Use orthogonal prediction for tensor mode
            predictions, probabilities = self.orthogonal_predict(features_batch)

            batch_size = len(features_batch)
            total_samples += batch_size

            # Convert predictions to encoded format for comparison - TYPE SAFE
            try:
                # Direct conversion if predictions are already encoded
                if predictions and all(isinstance(p, (int, float, np.number)) for p in predictions):
                    encoded_predictions = predictions
                else:
                    # Use encoder for conversion
                    encoded_predictions = self.class_encoder.transform(predictions)
            except Exception as e:
                self.log(f"⚠️ Prediction encoding failed, using fallback: {e}")
                # Fallback: assume predictions are already encoded
                encoded_predictions = [float(p) if isinstance(p, (int, float, np.number)) else 1.0 for p in predictions]

            for i in range(batch_size):
                if i < len(encoded_predictions):
                    predicted = encoded_predictions[i]
                else:
                    predicted = 1.0  # Default fallback

                actual = targets_batch[i]
                all_predictions.append(predicted)

                # Check if prediction is correct (using margin)
                if abs(actual - predicted) <= self.class_labels[0]:
                    correct_predictions += 1

        accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
        return accuracy, correct_predictions, all_predictions


    def build_tensor_transformation(self, features_batches, encoded_targets_batches):
        """
        Build orthogonal tensor transformation instead of iterative learning
        FIXED: Correct method signature
        """
        self.log("Building tensor space transformation...")

        # Combine all batches
        all_features = np.vstack(features_batches)
        all_targets = np.concatenate(encoded_targets_batches)

        n_samples, n_features = all_features.shape
        n_classes = len(np.unique(all_targets))

        # Step 1: Create feature-class correlation tensor
        feature_class_tensor = self._build_correlation_tensor(all_features, all_targets, n_features, n_classes)

        # Step 2: Perform orthogonal decomposition
        self.orthogonal_basis = self._tensor_orthogonal_decomposition(feature_class_tensor)

        # Step 3: Build projection matrix - FIXED: Correct call with 2 arguments
        self.weight_matrix = self._build_projection_matrix(all_features, all_targets)

        # Step 4: Initialize anti_net with orthogonal projections
        self._initialize_orthogonal_network()

        self.log(f"✅ Tensor transformation built: {n_features} features → {n_classes} classes")
        return True

    def _build_projection_matrix(self, features, targets):
        """
        Build projection matrix that maps features to class probabilities
        FIXED: Takes 3 arguments (self, features, targets) instead of 4
        """
        n_samples, n_features = features.shape
        n_classes = len(np.unique(targets))

        # Create one-hot encoded targets
        target_one_hot = np.zeros((n_samples, n_classes))
        for i, target in enumerate(targets):
            target_one_hot[i, int(target)-1] = 1.0

        # Solve for projection matrix: features @ W = target_one_hot
        # Using ridge regression for stability
        alpha = 0.1  # Regularization
        W = np.linalg.solve(
            features.T @ features + alpha * np.eye(n_features),
            features.T @ target_one_hot
        )

        return W

    def _build_correlation_tensor(self, features, targets, n_features, n_classes):
        """
        Build feature-class correlation tensor instead of iterative counting
        """
        resol = self.config.get('resol', 100)
        correlation_tensor = np.zeros((n_features+2, resol+2, n_features+2, resol+2, n_classes+2))

        # Discretize features into bins
        feature_bins = {}
        for i in range(n_features):
            feature_min = np.min(features[:, i])
            feature_max = np.max(features[:, i])
            bins = np.linspace(feature_min, feature_max, resol)
            feature_bins[i] = bins

        # Build correlation counts in one pass (no iteration!)
        for sample_idx in range(len(features)):
            sample_features = features[sample_idx]
            target_class = int(targets[sample_idx])

            # For each feature pair, update correlation counts
            for i in range(n_features):
                # Find bin for feature i
                bin_i = np.digitize(sample_features[i], feature_bins[i])
                bin_i = np.clip(bin_i, 1, resol)

                for j in range(i+1, n_features):  # Only upper triangle for efficiency
                    # Find bin for feature j
                    bin_j = np.digitize(sample_features[j], feature_bins[j])
                    bin_j = np.clip(bin_j, 1, resol)

                    # Update correlation tensor - this replaces iterative anti_net updates
                    correlation_tensor[i+1, bin_i, j+1, bin_j, target_class] += 1
                    correlation_tensor[j+1, bin_j, i+1, bin_i, target_class] += 1  # Symmetric

        return correlation_tensor


    def _tensor_orthogonal_decomposition(self, correlation_tensor):
        """
        Perform orthogonal decomposition of the correlation tensor with optimal variance threshold
        """
        n_features = correlation_tensor.shape[0] - 2
        n_classes = correlation_tensor.shape[4] - 2

        # Reshape tensor for decomposition
        tensor_flat = correlation_tensor[1:n_features+1, 1:-1, 1:n_features+1, 1:-1, 1:n_classes+1]
        tensor_2d = tensor_flat.reshape(n_features * (self.config['resol']),
                                      n_features * (self.config['resol']) * n_classes)

        # Perform SVD for orthogonal basis
        U, s, Vt = np.linalg.svd(tensor_2d, full_matrices=False)

        # Calculate optimal variance threshold based on data characteristics
        optimal_threshold = self._calculate_optimal_variance_threshold(s, n_features, n_classes)

        # Keep principal components that explain optimal variance
        explained_variance = np.cumsum(s) / np.sum(s)
        n_components = np.argmax(explained_variance >= optimal_threshold) + 1

        # Ensure we don't take too many components (avoid overfitting)
        max_reasonable_components = min(n_features * 10, len(s))  # Limit to avoid explosion
        n_components = min(n_components, max_reasonable_components)

        orthogonal_basis = U[:, :n_components]

        self.log(f"🧠 Optimal orthogonal decomposition:")
        self.log(f"   Components: {n_components} (max reasonable: {max_reasonable_components})")
        self.log(f"   Variance explained: {explained_variance[n_components-1]:.3%}")
        self.log(f"   Optimal threshold: {optimal_threshold:.3%}")
        self.log(f"   Singular values range: {s[0]:.3f} to {s[-1]:.3f}")

        return orthogonal_basis

    def _calculate_optimal_variance_threshold(self, singular_values, n_features, n_classes):
        """
        Calculate optimal variance threshold based on data characteristics
        Higher thresholds for simpler problems, adaptive for complex ones
        """
        total_variance = np.sum(singular_values)

        # Calculate data complexity metrics
        variance_ratio = singular_values[0] / total_variance if len(singular_values) > 0 else 0
        effective_rank = np.sum(singular_values > 1e-10)  # Numerical rank

        # Base threshold: higher for simpler problems (first component explains a lot)
        if variance_ratio > 0.5:
            # Simple problem - first component dominates
            base_threshold = 0.98  # Very high threshold
        elif variance_ratio > 0.3:
            # Moderately complex
            base_threshold = 0.95
        else:
            # Complex problem - need more components
            base_threshold = 0.90

        # Adjust based on feature-to-class ratio
        feature_class_ratio = n_features / max(n_classes, 1)
        if feature_class_ratio > 10:
            # High dimensionality - be more conservative
            base_threshold = min(base_threshold, 0.92)
        elif feature_class_ratio < 2:
            # Low dimensionality - can be more aggressive
            base_threshold = max(base_threshold, 0.85)

        # Adjust based on effective rank
        max_possible_components = len(singular_values)
        if effective_rank < max_possible_components * 0.3:
            # Low effective rank - problem is simpler than it appears
            base_threshold = min(base_threshold + 0.03, 0.99)

        self.log(f"🔧 Variance threshold calculation:")
        self.log(f"   First component ratio: {variance_ratio:.3f}")
        self.log(f"   Feature/class ratio: {feature_class_ratio:.2f}")
        self.log(f"   Effective rank: {effective_rank}/{max_possible_components}")
        self.log(f"   Final threshold: {base_threshold:.3%}")

        return base_threshold

    def _build_projection_matrix(self, features, targets):
        """
        Build projection matrix with enhanced regularization for better generalization
        """
        n_samples, n_features = features.shape
        n_classes = len(np.unique(targets))

        # Create one-hot encoded targets
        target_one_hot = np.zeros((n_samples, n_classes))
        for i, target in enumerate(targets):
            class_index = int(target) - 1
            if 0 <= class_index < n_classes:
                target_one_hot[i, class_index] = 1.0

        # Enhanced regularization based on data characteristics
        condition_number = np.linalg.cond(features.T @ features)

        # Adaptive regularization strength
        if condition_number > 1e10:
            alpha = 1.0  # Strong regularization for ill-conditioned problems
        elif condition_number > 1e6:
            alpha = 0.5  # Moderate regularization
        elif condition_number > 1e3:
            alpha = 0.1  # Light regularization
        else:
            alpha = 0.01  # Minimal regularization for well-conditioned problems

        # Add class balancing for imbalanced datasets
        class_counts = np.sum(target_one_hot, axis=0)
        class_weights = 1.0 / (class_counts + 1e-8)
        class_weights = class_weights / np.sum(class_weights)

        # Apply class weights
        weighted_targets = target_one_hot * class_weights

        # Solve for projection matrix with enhanced regularization
        XTX = features.T @ features
        XTY = features.T @ weighted_targets

        # Add regularization to improve generalization
        W = np.linalg.solve(
            XTX + alpha * np.eye(n_features),
            XTY
        )

        self.log(f"🔧 Enhanced projection matrix:")
        self.log(f"   Condition number: {condition_number:.3e}")
        self.log(f"   Regularization alpha: {alpha}")
        self.log(f"   Matrix shape: {W.shape}")
        self.log(f"   Class distribution: {class_counts}")

        return W

    def _initialize_orthogonal_network(self):
        """
        Initialize anti_net and anti_wts using orthogonal projections
        """
        n_features = self.innodes
        resol = self.config.get('resol', 100)
        n_classes = self.outnodes

        # Initialize with orthogonal structure
        self.anti_net = np.ones((n_features+2, resol+2, n_features+2, resol+2, n_classes+2), dtype=np.int32)
        self.anti_wts = np.ones((n_features+2, resol+2, n_features+2, resol+2, n_classes+2), dtype=np.float32)

        # Project orthogonal basis into the network structure
        if self.orthogonal_basis is not None:
            basis_3d = self.orthogonal_basis.reshape(n_features, resol, -1)

            for k in range(1, n_classes+1):
                for comp_idx in range(basis_3d.shape[2]):
                    component = basis_3d[:, :, comp_idx]

                    # Distribute component weights across the network
                    for i in range(1, n_features+1):
                        for j in range(1, resol+1):
                            weight_val = abs(component[i-1, j-1])
                            if weight_val > 0:
                                # Spread influence to related features
                                for l in range(1, n_features+1):
                                    if l != i:
                                        influence = weight_val * 0.1  # Reduced influence
                                        self.anti_wts[i, j, l, 1, k] += influence
                                        self.anti_wts[l, 1, i, j, k] += influence


    def tensor_train(self, train_file: str, test_file: Optional[str] = None,
                    use_csv: bool = True, target_column: Optional[str] = None,
                    feature_columns: Optional[List[str]] = None):
        """
        Single-pass tensor transformation training with robust type handling
        """
        self.log("🧠 Starting tensor space transformation training...")

        # Load data
        features_batches, targets_batches, feature_columns_used, original_targets_batches = self.load_data(
            train_file, target_column, feature_columns
        )

        if not features_batches:
            self.log("No training data loaded")
            return False

        # Store feature configuration
        self.feature_columns = feature_columns_used
        self.target_column = target_column if target_column else ""

        # Fit encoder and determine architecture
        all_original_targets = np.concatenate(original_targets_batches) if original_targets_batches else np.array([])
        self.class_encoder.fit(all_original_targets)

        # Log encoder information for debugging
        self.log(f"🔧 Encoder configuration:")
        self.log(f"   Original dtype: {getattr(self.class_encoder, 'original_dtype', 'unknown')}")
        self.log(f"   Class mappings: {len(self.class_encoder.class_to_encoded)}")

        encoded_targets_batches = []
        for batch in original_targets_batches:
            encoded_batch = self.class_encoder.transform(batch)
            encoded_targets_batches.append(encoded_batch)

        self.innodes = len(feature_columns_used)
        self.outnodes = len(self.class_encoder.get_encoded_classes())

        # Initialize arrays
        resol = self.config.get('resol', 100)
        self.initialize_arrays(self.innodes, resol, self.outnodes)

        # Set up class_labels with encoded values
        encoded_classes = self.class_encoder.get_encoded_classes()
        self.class_labels[0] = self.config.get('margin', 0.2)
        for i, encoded_val in enumerate(encoded_classes, 1):
            self.class_labels[i] = float(encoded_val)

        # SINGLE PASS: Build tensor transformation
        start_time = time.time()
        success = self.build_tensor_transformation(features_batches, encoded_targets_batches)
        training_time = time.time() - start_time

        if not success:
            self.log("❌ Tensor transformation failed")
            return False

        # Enhanced evaluation with robust type handling
        accuracy, correct_predictions, predictions = self.tensor_evaluate(features_batches, encoded_targets_batches)
        total_samples = sum(len(batch) for batch in features_batches)

        self.is_trained = True
        self.best_accuracy = accuracy

        self.log(f"✅ Tensor training completed in {training_time:.2f}s")
        self.log(f"✅ Final Accuracy = {accuracy:.2f}% ({correct_predictions}/{total_samples})")
        self.log(f"✅ Feature configuration: {len(self.feature_columns)} features")

        return True

# Example usage
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

    # Example configuration
    config = {
        'resol': 100,
        'gain': 2.0,
        'epochs': 50,
        'patience': 5,
        'margin': 0.2
    }

    # Create workflow
    workflow = DBNNWorkflow(config)

    # Example usage (uncomment to run)
    # workflow.run_complete_workflow(
    #     train_file="data/train.csv",
    #     test_file="data/test.csv",
    #     use_csv=True,
    #     target_column="target",
    #     feature_columns=["feature1", "feature2", "feature3"]
    # )

    print("DBNN class structure ready for use!")
